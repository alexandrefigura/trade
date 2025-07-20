"""
Funções utilitárias genéricas
"""
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from trade_system.logging_config import logging
import logging

logger = logging.getLogger(__name__)


def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Calcula o ATR (Average True Range)
    
    Args:
        high: Array de preços máximos
        low: Array de preços mínimos
        close: Array de preços de fechamento
        period: Período para cálculo da média
        
    Returns:
        Array com valores de ATR
    """
    if len(close) < 2:
        return np.full_like(close, np.nan)
    
    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    # Média móvel simples do TR
    if len(tr) >= period:
        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        # Padronizar para o mesmo tamanho de close
        return np.concatenate([np.full(len(close) - len(atr), np.nan), atr])
    else:
        return np.full_like(close, np.nan)


def calculate_atr_from_prices(
    prices: np.ndarray,
    period: int = 14,
    volatility_factor: float = 0.5
) -> Optional[float]:
    """
    Calcula ATR aproximado usando apenas preços de fechamento
    
    Args:
        prices: Array de preços
        period: Período para cálculo
        volatility_factor: Fator para aproximar high/low
        
    Returns:
        Valor ATR aproximado ou None
    """
    if len(prices) < period:
        return None
    
    # Aproximar high/low usando volatilidade
    volatility = np.std(prices[-period:])
    
    # Criar pseudo-OHLC
    high = prices + volatility * volatility_factor
    low = prices - volatility * volatility_factor
    close = prices
    
    # Calcular ATR
    atr_values = calculate_atr(high, low, close, period)
    
    # Retornar último valor válido
    if len(atr_values) > 0 and not np.isnan(atr_values[-1]):
        return float(atr_values[-1])
    
    # Fallback: usar volatilidade * fator
    return volatility * 1.5


def format_price(price: float, decimals: int = 2) -> str:
    """Formata preço com separadores de milhares"""
    return f"${price:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Formata percentual com sinal"""
    return f"{value*100:+.{decimals}f}%"


def calculate_position_metrics(
    entry_price: float,
    current_price: float,
    quantity: float,
    side: str,
    fees: float = 0
) -> Dict[str, float]:
    """
    Calcula métricas de uma posição
    
    Returns:
        Dict com pnl, pnl_pct, value, etc
    """
    if side == 'BUY':
        pnl = (current_price - entry_price) * quantity
        pnl_pct = (current_price - entry_price) / entry_price
    else:  # SELL
        pnl = (entry_price - current_price) * quantity
        pnl_pct = (entry_price - current_price) / entry_price
    
    pnl_after_fees = pnl - fees
    current_value = current_price * quantity
    
    return {
        'pnl': pnl,
        'pnl_after_fees': pnl_after_fees,
        'pnl_pct': pnl_pct,
        'current_value': current_value,
        'fees': fees
    }


def calculate_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calcula Sharpe Ratio
    
    Args:
        returns: Array de retornos
        periods_per_year: 252 para daily, 52 para weekly, etc
        
    Returns:
        Sharpe Ratio anualizado
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * mean_return / std_return


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calcula drawdown máximo
    
    Args:
        equity_curve: Lista de valores de equity
        
    Returns:
        Drawdown máximo como percentual (0-1)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve[1:]:
        if value > peak:
            peak = value
        else:
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
    
    return max_dd


def is_market_open() -> bool:
    """Verifica se o mercado crypto está operacional (sempre True)"""
    return True


def get_time_until_next_candle(interval_minutes: int = 5) -> int:
    """
    Calcula segundos até próximo candle
    
    Args:
        interval_minutes: Intervalo do candle em minutos
        
    Returns:
        Segundos até próximo candle
    """
    now = datetime.now()
    current_minutes = now.minute
    minutes_in_interval = current_minutes % interval_minutes
    minutes_until_next = interval_minutes - minutes_in_interval
    
    return minutes_until_next * 60 - now.second


def validate_price_data(prices: np.ndarray) -> bool:
    """
    Valida array de preços
    
    Returns:
        True se dados são válidos
    """
    if len(prices) == 0:
        return False
    
    if np.any(prices <= 0):
        return False
    
    if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
        return False
    
    # Verificar variação mínima
    if np.std(prices) == 0:
        return False
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divisão segura com valor padrão"""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Limita valor entre min e max"""
    return max(min_value, min(value, max_value))
