"""Sistema de cache com Redis e fallback local"""
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional
import redis
from collections import OrderedDict
import pickle
from typing import Dict, Any, Optional

class CacheManager:
    """Gerenciador de cache com Redis e fallback local"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurações
        self.ttl = config.system.get('cache_ttl', 60)
        self.max_local_items = 1000
        
        # Tentar conectar ao Redis
        self.redis_client = self._connect_redis()
        
        # Cache local (fallback)
        self.local_cache = OrderedDict()
        
    def _connect_redis(self) -> Optional[redis.Redis]:
        """Conecta ao Redis se disponível"""
        try:
            client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=False,
                socket_connect_timeout=1
            )
            
            # Testar conexão
            client.ping()
            self.logger.info("✅ Redis conectado")
            return client
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.logger.warning(f"⚠️ Redis conexão falhou: {e}")
            return None
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera valor do cache"""
        # Tentar Redis primeiro
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                self.logger.debug(f"Redis get error: {e}")
        
        # Fallback para cache local
        if key in self.local_cache:
            value, expiry = self.local_cache[key]
            if datetime.now() < expiry:
                # Mover para o final (LRU)
                self.local_cache.move_to_end(key)
                return value
            else:
                # Expirado
                del self.local_cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Armazena valor no cache"""
        ttl = ttl or self.ttl
        
        # Tentar Redis
        if self.redis_client:
            try:
                serialized = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                self.logger.debug(f"Redis set error: {e}")
        
        # Sempre armazenar no cache local também
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.local_cache[key] = (value, expiry)
        
        # Limitar tamanho do cache local
        if len(self.local_cache) > self.max_local_items:
            # Remover item mais antigo
            self.local_cache.popitem(last=False)
    
    def delete(self, key: str):
        """Remove valor do cache"""
        # Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception:
                pass
        
        # Local
        if key in self.local_cache:
            del self.local_cache[key]
    
    def clear(self):
        """Limpa todo o cache"""
        # Redis
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception:
                pass
        
        # Local
        self.local_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        stats = {
            'redis_connected': self.redis_client is not None,
            'local_cache_size': len(self.local_cache),
            'local_cache_items': list(self.local_cache.keys())[:10]  # Primeiros 10
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats['redis_memory'] = info.get('used_memory_human', 'N/A')
                stats['redis_keys'] = self.redis_client.dbsize()
            except Exception:
                pass
        
        return stats
