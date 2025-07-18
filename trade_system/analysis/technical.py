"""
Análise técnica ultra-rápida com NumPy e Numba
"""
import time
import numpy as np
import pandas as pd
import numba as nb
from numba import njit
from typing import Tuple, Optional, Dict
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


# ===========================
# FUNÇÕES NUMBA ULTRA-RÁPIDAS
# ===========================

@njit(nopython=True, cache=True, fastmath=True)
def calculate_sma_fast(prices: np.ndarray, period: int) -> np.ndarray:
    """SMA ultra-rápida com janela deslizante."""
    n = prices.shape[0]
    sma = np.empty(n, dtype=np.float32)
    sma[:period-1] = np.nan
    # primeira média
    acc = 0.0
    for i in range(period):
        acc += prices[i]
    sma[period-1] = acc / period
    # janela deslizante
    for i in range(period, n):
        acc += prices[i] - prices[i-period]
        sma[i] = acc / period
    return sma


@njit(nopython=True, cache=True, fastmath=True)
def calculate_ema_fast(prices: np.ndarray, period: int) -> np.ndarray:
    """EMA ultra-rápida."""
    n = prices.shape[0]
    ema = np.empty(n, dtype=np.float32)
    alpha = 2.0 / (period + 1)
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema


@njit(nopython=True, cache=True, fastmath=True)
def calculate_rsi_fast(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI ultra-rápido."""
    n = prices.shape[0]
    rsi = np.empty(n, dtype=np.float32)
    rsi[:period] = np.nan

    # deltas
    deltas = np.empty(n-1, dtype=np.float32)
    for i in range(1, n):
        deltas[i-1] = prices[i] - prices[i-1]

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # médias iniciais
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    # RSI inicial
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi[period] = 100.0

    # iteração incremental
    for i in range(period+1, n):
        gain = gains[i-1]
        loss = losses[i-1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0

    return rsi


@njit(nopython=True, cache=True, fastmath=True)
def calculate_bollinger_bands_fast(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands ultra-rápidas."""
    n = prices.shape[0]
    sma = calculate_sma_fast(prices, period)
    std = np.empty(n, dtype=np.float32)
    std[:period-1] = np.nan
    for i in range(period-1, n):
        acc = 0.0
        for j in range(i-period+1, i+1):
            diff = prices[j] - sma[i]
            acc += diff * diff
        std[i] = np.sqrt(acc / period)
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


@njit(nopython=True, cache=True, fastmath=True)
def detect_patterns_fast(prices: np.ndarray, volumes: np.ndarray) -> int:
    """
    Detecção de padrões ultra-rápida.
    Retorna:  1 (compra), 0 (neutro), -1 (venda)
    """
    n = prices.shape[0]
    if n < 50:
        return 0
    # momentum e volume spike
    momentum = (prices[-1] - prices[-10]) / prices[-10]
    avg_vol = 0.0
    for i in range(n-20, n):
        avg_vol += volumes[i]
    avg_vol /= 20.0
    volume_spike = volumes[-1] > avg_vol * 1.5
    # breakout
    high20 = prices[n-20:n].max()
    low20 = prices[n-20:n].min()
    if prices[-1] > high20 * 0.995 and volume_spike and momentum > 0.001:
        return 1
    if prices[-1] < low20 * 1.005 and volume_spike and momentum < -0.001:
        return -1
    return 0


def filter_low_volume_and_volatility(
    prices: np.ndarray,
    volumes: np.ndarray,
    min_volume_multiplier: float,
    max_recent_volatility: float
) -> Optional[Tuple[str, float, Dict]]:
    """
    Cancela sinais se o volume estiver baixo ou volatilidade muito alta.
    Retorna um tuple de (action, confidence, details) ou None.
    """
    n = prices.shape[0]
    # volume
    if n >= 20:
        avg_vol20 = np.mean(volumes[-20:])
        if volumes[-1] < avg_vol20 * min_volume_multiplier:
            return 'HOLD', 0.5, {'reason': 'Low volume'}
    # volatilidade
    if n >= 50:
        recent_vol = np.std(prices[-50:]) / np.mean(prices[-50:])
        if recent_vol > max_recent_volatility:
            return 'HOLD', 0.5, {'reason': 'High volatility'}
    return None


class UltraFastTechnicalAnalysis:
    """Análise técnica com NumPy/Numba para máxima velocidade."""
    def __init__(self, config):
        self.config = config
        self._last_time = 0.0
        self._interval = config.ta_interval_ms / 1000.0
        self._cache: Tuple[str, float, Dict] = ('HOLD', 0.5, {'cached': True})
        self.stats = {'BUY':0, 'SELL':0, 'HOLD':0}

    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> Tuple[str, float, Dict]:
        now = time.time()
        # checagens rápidas
        if prices.size < self.config.min_prices or np.std(prices[-self.config.min_prices:]) == 0:
            return 'HOLD', 0.5, {'reason': 'Insufficient data'}
        if now - self._last_time < self._interval:
            return self._cache

        # filtros de segurança
        filt = None
        if not self.config.debug_mode:
            filt = filter_low_volume_and_volatility(
                prices, volumes,
                self.config.min_volume_multiplier,
                self.config.max_recent_volatility
            )
        if filt is not None:
            return filt

        start = time.perf_counter()

        # indicadores
        sma_s = calculate_sma_fast(prices, self.config.sma_short_period)
        sma_l = calculate_sma_fast(prices, self.config.sma_long_period)
        ema_s = calculate_ema_fast(prices, self.config.ema_short_period)
        ema_l = calculate_ema_fast(prices, self.config.ema_long_period)
        rsi = calculate_rsi_fast(prices, self.config.rsi_period)[-1]
        bb_u, bb_m, bb_d = calculate_bollinger_bands_fast(
            prices, self.config.bb_period, self.config.bb_std_dev
        )
        patt = detect_patterns_fast(prices, volumes)

        # colecionar signals
        signals = []
        confs = []
        reasons = []

        # RSI
        if 0 <= rsi <= 100:
            if rsi < self.config.rsi_buy_threshold:
                signals.append(1); confs.append(self.config.rsi_confidence)
                reasons.append(f"RSI {rsi:.1f} < buy_thr")
            elif rsi > self.config.rsi_sell_threshold:
                signals.append(-1); confs.append(self.config.rsi_confidence)
                reasons.append(f"RSI {rsi:.1f} > sell_thr")

        # EMA cross
        if ema_s[-2] <= ema_l[-2] and ema_s[-1] > ema_l[-1]:
            diff = (ema_s[-1]-ema_l[-1]) / (ema_l[-1] + 1e-9)
            if diff > self.config.ema_cross_strength:
                signals.append(1); confs.append(self.config.ema_confidence); reasons.append("EMA Bull")
        if ema_s[-2] >= ema_l[-2] and ema_s[-1] < ema_l[-1]:
            diff = (ema_l[-1]-ema_s[-1]) / (ema_l[-1] + 1e-9)
            if diff > self.config.ema_cross_strength:
                signals.append(-1); confs.append(self.config.ema_confidence); reasons.append("EMA Bear")

        # BB
        last_price = prices[-1]
        if last_price < bb_d[-1] * (1 - 1e-3):
            signals.append(1); confs.append(self.config.bb_confidence); reasons.append("BB Lower")
        elif last_price > bb_u[-1] * (1 + 1e-3):
            signals.append(-1); confs.append(self.config.bb_confidence); reasons.append("BB Upper")

        # pattern
        if patt != 0:
            signals.append(patt); confs.append(self.config.pattern_confidence)
            reasons.append("Pattern")

        # consolidar
        if not signals:
            action, conf = 'HOLD', 0.5
        else:
            arr_s = np.array(signals, dtype=np.float32)
            arr_c = np.array(confs, dtype=np.float32)
            weighted = np.dot(arr_s, arr_c) / (arr_c.sum() + 1e-9)
            conf = float(arr_c.mean())
            if weighted > self.config.buy_threshold:
                action = 'BUY'
            elif weighted < -self.config.sell_threshold:
                action = 'SELL'
            else:
                action = 'HOLD'

        # estatísticas e cache
        self.stats[action] += 1
        elapsed = (time.perf_counter() - start) * 1000
        details = {
            'rsi': float(rsi),
            'ema_diff': float(ema_s[-1] - ema_l[-1]),
            'bb_upper': float(bb_u[-1]),
            'bb_lower': float(bb_d[-1]),
            'pattern': int(patt),
            'reasons': reasons,
            'calc_ms': elapsed
        }
        if action != 'HOLD':
            logger.info(f"📊 TA {action} (conf={conf:.2%})")
            logger.debug(f"   {details}")

        self._last_time = now
        self._cache = (action, conf, details)
        return action, conf, details

    def get_signal_stats(self) -> Dict[str, float]:
        total = sum(self.stats.values()) + 1e-9
        return {
            'total': int(total),
            'buy': self.stats['BUY'],
            'sell': self.stats['SELL'],
            'hold': self.stats['HOLD'],
            'buy_pct': self.stats['BUY']/total*100,
            'sell_pct': self.stats['SELL']/total*100,
            'hold_pct': self.stats['HOLD']/total*100,
        }
