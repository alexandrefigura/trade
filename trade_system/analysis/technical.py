"""
Análise técnica ultra-rápida com NumPy e Numba
"""
import time
import numpy as np
import numba as nb
from numba import njit
from typing import Tuple, Optional, Dict
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


# ===========================
# FUNÇÕES NUMBA ULTRA-RÁPIDAS
# ===========================

@nb.njit(cache=True, fastmath=True)
def calculate_sma_fast(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    sma = np.empty(n, dtype=np.float32)
    sma[:period-1] = np.nan
    sma[period-1] = np.mean(prices[:period])
    for i in range(period, n):
        sma[i] = sma[i-1] + (prices[i] - prices[i-period]) / period
    return sma

@nb.njit(cache=True, fastmath=True)
def calculate_ema_fast(prices: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

@nb.njit(cache=True, fastmath=True)
def calculate_rsi_fast(prices: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(prices)
    rsi = np.empty(n, dtype=np.float32)
    rsi[:period] = np.nan
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi_val = 100.0
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi_val = 100 - (100 / (1 + rs))
    rsi[period] = rsi_val.astype(np.float32)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    return rsi

@nb.njit(cache=True, fastmath=True)
def calculate_bollinger_bands_fast(
    prices: np.ndarray, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sma = calculate_sma_fast(prices, period)
    n = len(prices)
    std = np.empty(n, dtype=np.float32)
    std[:period-1] = np.nan
    for i in range(period-1, n):
        std[i] = np.std(prices[i-period+1:i+1])
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

@nb.njit(cache=True, fastmath=True)
def detect_patterns_fast(prices: np.ndarray, volumes: np.ndarray) -> int:
    if len(prices) < 200:
        return 0
    if np.std(prices[-50:]) == 0:
        return 0
    momentum = (prices[-1] - prices[-10]) / prices[-10]
    avg_volume = np.mean(volumes[-20:])
    volume_spike = volumes[-1] > avg_volume * 1.5
    high_20 = np.max(prices[-20:])
    low_20 = np.min(prices[-20:])
    if prices[-1] > high_20 * 0.995 and volume_spike and momentum > 0.001:
        return 1
    elif prices[-1] < low_20 * 1.005 and volume_spike and momentum < -0.001:
        return -1
    return 0

def filter_low_volume_and_volatility(
    prices: np.ndarray,
    volumes: np.ndarray,
    min_volume_multiplier: float,
    max_recent_volatility: float
) -> Optional[Tuple[str, float, Dict]]:
    if len(volumes) >= 20:
        avg_vol20 = np.mean(volumes[-20:])
        if volumes[-1] < avg_vol20 * min_volume_multiplier:
            logger.debug(f"Volume baixo: {volumes[-1]:.2f} < {avg_vol20 * min_volume_multiplier:.2f}")
            return 'HOLD', 0.5, {'reason': 'Filtrado por volume'}
    if len(prices) >= 50:
        recent_vol = np.std(prices[-50:]) / np.mean(prices[-50:])
        if recent_vol > max_recent_volatility:
            logger.debug(f"Volatilidade alta: {recent_vol:.4f} > {max_recent_volatility:.4f}")
            return 'HOLD', 0.5, {'reason': 'Volatilidade alta'}
    return None


class UltraFastTechnicalAnalysis:
    """Análise técnica com NumPy/Numba para máxima velocidade"""

    def __init__(self, config):
        self.config = config
        # Fallbacks para valores de config que podem não existir
        self.min_prices = getattr(config, 'min_prices', 200)
        self.ta_interval = getattr(config, 'ta_interval_ms', 500) / 1000.0
        self.min_volume_multiplier = getattr(config, 'min_volume_multiplier', 0.5)
        self.max_recent_volatility = getattr(config, 'max_recent_volatility', 0.05)
        self.sma_short_period = config.sma_short_period
        self.sma_long_period = config.sma_long_period
        self.ema_short_period = config.ema_short_period
        self.ema_long_period = config.ema_long_period
        self.rsi_period = config.rsi_period
        self.bb_period = config.bb_period
        self.bb_std_dev = config.bb_std_dev
        self.rsi_buy_threshold = config.rsi_buy_threshold
        self.rsi_sell_threshold = config.rsi_sell_threshold
        self.rsi_confidence = config.rsi_confidence
        self.sma_cross_confidence = config.sma_cross_confidence
        self.bb_confidence = config.bb_confidence
        self.pattern_confidence = config.pattern_confidence
        self.buy_threshold = getattr(config, 'buy_threshold', 0.5)
        self.sell_threshold = getattr(config, 'sell_threshold', 0.5)
        self.debug_mode = getattr(config, 'debug_mode', False)

        self.last_time = 0.0
        self._cached = ('HOLD', 0.5, {'reason': 'cached'})
        self.signal_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}

    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> Tuple[str, float, Dict]:
        now = time.time()
        # valida tamanho mínimo
        if prices.size < self.min_prices or np.std(prices[-self.min_prices:]) == 0:
            return 'HOLD', 0.5, {'reason': 'Dados insuficientes'}
        # checar dados inválidos
        if np.any(prices <= 0) or np.isnan(prices).any() or np.isinf(prices).any():
            return 'HOLD', 0.5, {'reason': 'Dados anômalos'}
        # respeitar intervalo
        if now - self.last_time < self.ta_interval:
            return self._cached
        # aplicar filtros de volume/volatilidade
        if not self.debug_mode:
            fil = filter_low_volume_and_volatility(
                prices, volumes,
                self.min_volume_multiplier,
                self.max_recent_volatility
            )
            if fil is not None:
                return fil

        start = time.perf_counter()
        sma_s = calculate_sma_fast(prices, self.sma_short_period)
        sma_l = calculate_sma_fast(prices, self.sma_long_period)
        ema_s = calculate_ema_fast(prices, self.ema_short_period)
        ema_l = calculate_ema_fast(prices, self.ema_long_period)
        rsi = calculate_rsi_fast(prices, self.rsi_period)
        bb_u, bb_m, bb_l = calculate_bollinger_bands_fast(prices, self.bb_period, self.bb_std_dev)
        patt = detect_patterns_fast(prices, volumes)

        signals, confs, reasons = [], [], []
        current_rsi = rsi[-1]
        # RSI signals
        if not np.isnan(current_rsi):
            if current_rsi < self.rsi_buy_threshold:
                signals.append(1); confs.append(self.rsi_confidence); reasons.append(f"RSI low {current_rsi:.1f}")
            elif current_rsi > self.rsi_sell_threshold:
                signals.append(-1); confs.append(self.rsi_confidence); reasons.append(f"RSI high {current_rsi:.1f}")
        # EMA cross
        if ema_s[-1] > ema_l[-1] and ema_s[-2] <= ema_l[-2]:
            signals.append(1); confs.append(self.sma_cross_confidence); reasons.append("EMA cross buy")
        elif ema_s[-1] < ema_l[-1] and ema_s[-2] >= ema_l[-2]:
            signals.append(-1); confs.append(self.sma_cross_confidence); reasons.append("EMA cross sell")
        # Bollinger
        price = prices[-1]
        if price < bb_l[-1]:
            signals.append(1); confs.append(self.bb_confidence); reasons.append("BB low")
        elif price > bb_u[-1]:
            signals.append(-1); confs.append(self.bb_confidence); reasons.append("BB high")
        # pattern
        if patt != 0:
            signals.append(patt); confs.append(self.pattern_confidence); reasons.append("Pattern")

        # consolidar
        if not signals:
            action, overall_conf = 'HOLD', 0.5
        else:
            weights = np.array(confs, dtype=np.float32)
            vals = np.array(signals, dtype=np.float32)
            avg_conf = float(weights.mean())
            agg = float((vals * weights).sum() / weights.sum()) if weights.sum() else 0.0
            action = 'BUY' if agg > self.buy_threshold else 'SELL' if agg < -self.sell_threshold else 'HOLD'
            overall_conf = avg_conf

        # log
        if action != 'HOLD':
            logger.info(f"📊 TA Sinal: {action} ({overall_conf:.2%})")
        elapsed = (time.perf_counter() - start) * 1000
        details = {
            'rsi': float(current_rsi),
            'sma_short': float(sma_s[-1]),
            'sma_long': float(sma_l[-1]),
            'ema_short': float(ema_s[-1]),
            'ema_long': float(ema_l[-1]),
            'bb_upper': float(bb_u[-1]),
            'bb_lower': float(bb_l[-1]),
            'pattern': int(patt),
            'calc_ms': elapsed,
            'reasons': reasons
        }

        self.last_time = now
        self._cached = (action, overall_conf, details)
        self.signal_count[action] += 1
        return action, overall_conf, details

    def get_signal_stats(self) -> Dict:
        total = sum(self.signal_count.values())
        return {
            'total': total,
            'buy': self.signal_count['BUY'],
            'sell': self.signal_count['SELL'],
            'hold': self.signal_count['HOLD']
        }
