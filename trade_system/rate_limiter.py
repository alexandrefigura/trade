"""
Sistema de rate limiting para proteção de API
"""
import time
import threading
from collections import defaultdict
from functools import wraps
from typing import Dict, Callable
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Controle de rate limiting para proteção de API"""
    
    def __init__(self, config):
        self.config = config
        self.calls = defaultdict(list)
        self.lock = threading.Lock()
        self.window = config.rate_limit_window
        self.max_calls = config.rate_limit_max_calls
    
    def check_limit(self, endpoint: str = 'default') -> bool:
        """Verifica se pode fazer chamada"""
        with self.lock:
            now = time.time()
            # Limpar chamadas antigas
            self.calls[endpoint] = [
                t for t in self.calls[endpoint] 
                if now - t < self.window
            ]
            
            # Verificar limite
            if len(self.calls[endpoint]) >= self.max_calls:
                return False
            
            # Registrar chamada
            self.calls[endpoint].append(now)
            return True
    
    def wait_if_needed(self, endpoint: str = 'default'):
        """Aguarda se necessário para respeitar rate limit"""
        while not self.check_limit(endpoint):
            time.sleep(0.1)
    
    def get_usage(self) -> Dict:
        """Retorna uso atual dos rate limits"""
        with self.lock:
            now = time.time()
            usage = {}
            for endpoint, calls in self.calls.items():
                recent_calls = [t for t in calls if now - t < self.window]
                usage[endpoint] = {
                    'calls': len(recent_calls),
                    'limit': self.max_calls,
                    'percentage': len(recent_calls) / self.max_calls * 100,
                    'time_until_reset': self.window - (now - min(recent_calls)) if recent_calls else 0
                }
            return usage
    
    def get_wait_time(self, endpoint: str = 'default') -> float:
        """Retorna tempo de espera necessário em segundos"""
        with self.lock:
            now = time.time()
            self.calls[endpoint] = [
                t for t in self.calls[endpoint] 
                if now - t < self.window
            ]
            
            if len(self.calls[endpoint]) >= self.max_calls:
                oldest_call = min(self.calls[endpoint])
                wait_time = self.window - (now - oldest_call)
                return max(0, wait_time)
            
            return 0
    
    def reset(self, endpoint: str = None):
        """Reseta contador de chamadas"""
        with self.lock:
            if endpoint:
                self.calls[endpoint] = []
            else:
                self.calls.clear()
            logger.info(f"Rate limiter resetado para: {endpoint or 'todos endpoints'}")


def rate_limited(endpoint: str = 'default', wait: bool = True):
    """
    Decorator para aplicar rate limiting
    
    Args:
        endpoint: Nome do endpoint para controle separado
        wait: Se True, aguarda automaticamente. Se False, levanta exceção
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            if hasattr(self, 'rate_limiter'):
                if wait:
                    self.rate_limiter.wait_if_needed(endpoint)
                else:
                    if not self.rate_limiter.check_limit(endpoint):
                        wait_time = self.rate_limiter.get_wait_time(endpoint)
                        raise RateLimitExceeded(
                            f"Rate limit exceeded for {endpoint}. "
                            f"Wait {wait_time:.1f}s"
                        )
            return await func(self, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            if hasattr(self, 'rate_limiter'):
                if wait:
                    self.rate_limiter.wait_if_needed(endpoint)
                else:
                    if not self.rate_limiter.check_limit(endpoint):
                        wait_time = self.rate_limiter.get_wait_time(endpoint)
                        raise RateLimitExceeded(
                            f"Rate limit exceeded for {endpoint}. "
                            f"Wait {wait_time:.1f}s"
                        )
            return func(self, *args, **kwargs)
        
        # Retornar wrapper apropriado baseado no tipo da função
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RateLimitExceeded(Exception):
    """Exceção levantada quando rate limit é excedido"""
    pass


# Importação condicional do asyncio
try:
    import asyncio
except ImportError:
    # Fallback para sistemas sem asyncio
    asyncio = None
