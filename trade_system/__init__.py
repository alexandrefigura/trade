"""
Sistema de Trading Ultra-Otimizado v5.2

Um sistema de trading algorítmico de alta performance com:
- Análise técnica ultra-rápida com Numba
- WebSocket para dados em tempo real
- Machine Learning para predições
- Paper trading com dados reais
- Sistema de alertas multi-canal
- Gestão de risco avançada
"""

__version__ = "5.2.0"
__author__ = "Trading System Team"
__license__ = "MIT"

# Importações principais para facilitar uso
from trade_system.config import TradingConfig, get_config
from trade_system.logging_config import setup_logging, get_logger
from trade_system.cache import UltraFastCache
from trade_system.rate_limiter import RateLimiter, rate_limited
from trade_system.alerts import AlertSystem
from trade_system.signals import OptimizedSignalConsolidator
from trade_system.checkpoint import CheckpointManager

__all__ = [
    'UltraConfigV5',
    'get_config',
    'create_example_config',
    'setup_logging',
    'get_logger',
    'UltraFastCache',
    'RateLimiter',
    'rate_limited',
    'AlertSystem',
    'OptimizedSignalConsolidator',
    'CheckpointManager',
]
