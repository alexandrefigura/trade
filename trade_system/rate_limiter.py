"""Sistema de rate limiting"""
import time
import logging
from collections import deque
from typing import Any, Dict, Optional
import asyncio

class RateLimiter:
    """Controla rate limits de API"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Limite padrão da Binance: 1200 requests por minuto
        self.max_requests_per_minute = config.system.get('rate_limit_per_minute', 1200)
        
        # Fila de timestamps de requests
        self.request_times = deque()
        
        # Lock para thread safety
        self.lock = asyncio.Lock()
        
    async def acquire(self, weight: int = 1) -> bool:
        """
        Adquire permissão para fazer request
        
        Args:
            weight: Peso do request (alguns endpoints custam mais)
            
        Returns:
            True se pode fazer request
        """
        async with self.lock:
            now = time.time()
            
            # Limpar requests antigos (mais de 1 minuto)
            while self.request_times and self.request_times[0] < now - 60:
                self.request_times.popleft()
            
            # Verificar se pode fazer request
            current_weight = len(self.request_times)
            
            if current_weight + weight > self.max_requests_per_minute:
                # Calcular tempo de espera
                oldest_request = self.request_times[0]
                wait_time = 60 - (now - oldest_request) + 0.1
                
                self.logger.warning(f"Rate limit atingido. Aguardando {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                
                # Tentar novamente
                return await self.acquire(weight)
            
            # Registrar request
            for _ in range(weight):
                self.request_times.append(now)
            
            return True
    
    def get_current_usage(self) -> Dict[str, int]:
        """Retorna uso atual do rate limit"""
        now = time.time()
        
        # Contar requests no último minuto
        recent_requests = sum(1 for t in self.request_times if t > now - 60)
        
        return {
            'current': recent_requests,
            'limit': self.max_requests_per_minute,
            'percentage': (recent_requests / self.max_requests_per_minute) * 100
        }
