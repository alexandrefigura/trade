"""
Módulos de análise do sistema de trading
"""
from trade_system.analysis.technical import (
    UltraFastTechnicalAnalysis,
    calculate_sma_fast,
    calculate_ema_fast,
    calculate_rsi_fast,
    calculate_bollinger_bands_fast,
    detect_patterns_fast,
    filter_low_volume_and_volatility
)

__all__ = [
    'UltraFastTechnicalAnalysis',
    'calculate_sma_fast',
    'calculate_ema_fast',
    'calculate_rsi_fast',
    'calculate_bollinger_bands_fast',
    'detect_patterns_fast',
    'filter_low_volume_and_volatility'
]
