"""
Sistema de cache ultra-rápido com Redis e fallback local
"""
import time
import pickle
import redis
from typing import Any, Optional
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class UltraFastCache:
    """Cache em memória com Redis para latência < 1ms"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.use_redis
        self.local_cache = {}
        self.redis_client = None
        
        if self.enabled:
            self._init_redis()
    
    def _init_redis(self):
        """Inicializa conexão com Redis com retry"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=False,
                socket_keepalive=True,
                socket_connect_timeout=5,
                health_check_interval=30
            )
            self.redis_client.ping()
            logger.info("✅ Redis conectado com sucesso")
            self.enabled = True
        except redis.exceptions.ConnectionError as e:
            logger.warning(f"⚠️ Redis conexão falhou: {e}")
            self.enabled = False
        except redis.exceptions.TimeoutError as e:
            logger.warning(f"⚠️ Redis timeout: {e}")
            self.enabled = False
        except Exception as e:
            logger.warning(f"⚠️ Redis erro genérico: {e}")
            self.enabled = False
    
    def get(self, key: str) -> Optional[Any]:
        """Busca ultra-rápida no cache com tratamento de erros específicos"""
        try:
            if self.enabled and self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            else:
                cache_entry = self.local_cache.get(key)
                if cache_entry and time.time() < cache_entry[1]:
                    return cache_entry[0]
        except redis.exceptions.ConnectionError:
            logger.debug("Redis desconectado, usando cache local")
            self.enabled = False
            return self._get_from_local(key)
        except pickle.UnpicklingError:
            logger.debug(f"Erro ao deserializar cache key: {key}")
            return None
        except Exception as e:
            logger.debug(f"Erro no cache get: {e}")
            return None
        
        return None
    
    def _get_from_local(self, key: str) -> Optional[Any]:
        """Busca no cache local como fallback"""
        cache_entry = self.local_cache.get(key)
        if cache_entry and time.time() < cache_entry[1]:
            return cache_entry[0]
        return None
    
    def set(self, key: str, value: Any, ttl: int = 5):
        """Armazena no cache com TTL e tratamento de erros"""
        try:
            if self.enabled and self.redis_client:
                self.redis_client.setex(key, ttl, pickle.dumps(value))
            else:
                self._set_to_local(key, value, ttl)
        except redis.exceptions.ConnectionError:
            logger.debug("Redis desconectado ao salvar")
            self.enabled = False
            self._set_to_local(key, value, ttl)
        except pickle.PicklingError:
            logger.debug(f"Erro ao serializar valor para key: {key}")
        except Exception as e:
            logger.debug(f"Erro no cache set: {e}")
    
    def _set_to_local(self, key: str, value: Any, ttl: int):
        """Armazena no cache local"""
        self.local_cache[key] = (value, time.time() + ttl)
        # Limpar cache expirado se muito grande
        if len(self.local_cache) > 1000:
            self._cleanup_local_cache()
    
    def _cleanup_local_cache(self):
        """Limpa entradas expiradas do cache local"""
        try:
            now = time.time()
            self.local_cache = {
                k: v for k, v in self.local_cache.items()
                if v[1] > now
            }
            logger.debug(f"Cache local limpo: {len(self.local_cache)} entradas restantes")
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")
    
    def delete(self, key: str):
        """Remove uma chave do cache"""
        try:
            if self.enabled and self.redis_client:
                self.redis_client.delete(key)
            if key in self.local_cache:
                del self.local_cache[key]
        except Exception as e:
            logger.debug(f"Erro ao deletar cache key {key}: {e}")
    
    def clear(self):
        """Limpa todo o cache"""
        try:
            if self.enabled and self.redis_client:
                self.redis_client.flushdb()
            self.local_cache.clear()
            logger.info("Cache limpo completamente")
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do cache"""
        stats = {
            'redis_enabled': self.enabled,
            'local_cache_size': len(self.local_cache),
            'redis_connected': False
        }
        
        if self.enabled and self.redis_client:
            try:
                info = self.redis_client.info()
                stats['redis_connected'] = True
                stats['redis_memory_used'] = info.get('used_memory_human', 'N/A')
                stats['redis_keys'] = self.redis_client.dbsize()
            except:
                pass
        
        return stats
