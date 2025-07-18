"""
Análise técnica ultra-rápida com NumPy e Numba
"""
import time
import numpy as np
from numba import njit
from typing import Tuple, Optional, Dict
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


# ===========================
# FUNÇÕES NUMBA ULTRA-RÁPIDAS
# ===========================

@njit(cache=True, fastmath=True)
def calculate_sma_fast(prices: np.ndarray, period: int) -> np.ndarray:
    n = prices.shape[0]
    sma = np.empty(n, dtype=np.float32)
    sma[:period-1] = np.nan
    # primeira média
    acc = 0.0
    for i in range(period):
        acc += prices[i]
    sma[period-1] = acc / period
    # sliding window
    for i in range(period, n):
        acc += prices[i] - prices[i-period]
        sma[i] = acc / period
    return sma


@njit(cache=True, fastmath=True)
def calculate_ema_fast(prices: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    n = prices.shape[0]
    ema = np.empty(n, dtype=np.float32)
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema


@njit(cache=True, fastmath=True)
def calculate_rsi_fast(prices: np.ndarray, period: int = 14) -> np.ndarray:
    n = prices.shape[0]
    rsi = np.empty(n, dtype=np.float32)
    rsi[:period] = np.nan
    # diferenças
    gains = np.zeros(n-1, dtype=np.float32)
    losses = np.zeros(n-1, dtype=np.float32)
    for i in range(1, n):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            gains[i-1] = diff
        else:
            losses[i-1] = -diff
    # médias iniciais
    avg_gain = np.sum(gains[:period]) / period
    avg_loss = np.sum(losses[:period]) / period
    # primeiro RSI calculado em index=period
    if avg_loss != 0.0:
        rs = avg_gain / avg_loss
        rsi_val = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi_val = 100.0
    rsi[period] = np.float32(rsi_val)
    # incremento
    for i in range(period+1, n):
        gain = gains[i-1]
        loss = losses[i-1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss != 0.0:
            rs = avg_gain / avg_loss
            rsi_val = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_val = 100.0
        rsi[i] = np.float32(rsi_val)
    return rsi


@njit(cache=True, fastmath=True)
def calculate_bollinger_bands_fast(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sma = calculate_sma_fast(prices, period)
    n = prices.shape[0]
    std = np.empty(n, dtype=np.float32)
    std[:period-1] = np.nan
    for i in range(period-1, n):
        # cálculo de desvio padrão na janela [i-period+1, i]
        acc = 0.0
        acc2 = 0.0
        for j in range(i-period+1, i+1):
            v = prices[j]
            acc += v
            acc2 += v * v
        m = acc / period
        var = (acc2 / period) - (m * m)
        std[i] = np.sqrt(var) if var > 0.0 else 0.0
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


@njit(cache=True, fastmath=True)
def detect_patterns_fast(prices: np.ndarray, volumes: np.ndarray) -> int:
    if prices.shape[0] < 200:
        return 0
    # momentum
    momentum = (prices[-1] - prices[-10]) / prices[-10]
    # volume spike
    avg_vol = 0.0
    for i in range(-20, 0):
        avg_vol += volumes[volumes.shape[0]+i]
    avg_vol /= 20.0
    spike = volumes[-1] > avg_vol * 1.5
    # breakout
    high20 = np.max(prices[-20:])
    low20 = np.min(prices[-20:])
    if prices[-1] > high20 * 0.995 and spike and momentum > 0.001:
        return 1
    if prices[-1] < low20 * 1.005 and spike and momentum < -0.001:
        return -1
    return 0


def filter_low_volume_and_volatility(
    prices: np.ndarray,
    volumes: np.ndarray,
    min_volume_multiplier: float,
    max_recent_volatility: float
) -> Optional[Tuple[str, float, Dict]]:
    # volume
    if volumes.size >= 20:
        avg_vol20 = np.mean(volumes[-20:])
        if volumes[-1] < avg_vol20 * min_volume_multiplier:
            logger.debug(f"Volume baixo: {volumes[-1]:.2f} < {avg_vol20 * min_volume_multiplier:.2f}")
            return 'HOLD', 0.5, {'reason': 'Filtrado por volume'}
    # volatilidade
    if prices.size >= 50:
        recent_vol = np.std(prices[-50:]) / np.mean(prices[-50:])
        if recent_vol > max_recent_volatility:
            logger.debug(f"Volatilidade alta: {recent_vol:.4f} > {max_recent_volatility:.4f}")
            return 'HOLD', 0.5, {'reason': 'Volatilidade alta'}
    return None


class UltraFastTechnicalAnalysis:
    """Análise técnica com NumPy/Numba para máxima velocidade"""

    def __init__(self, config):
        self.config = config
        self.last_calc = 0.0
        self.interval = config.ta_interval_ms / 1000.0
        self._cached = ('HOLD', 0.5, {'cached': True})
        self.signals_count = {'BUY':0, 'SELL':0, 'HOLD':0}

    def analyze(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> Tuple[str, float, Dict]:
        now = time.time()
        # sanity checks
        if prices.size < self.config.sma_long_period or np.std(prices[-self.config.sma_long_period:]) == 0:
            return 'HOLD', 0.5, {'reason': 'Dados insuficientes ou sem variação'}
        if now - self.last_calc < self.interval:
            return self._cached

        # filtros
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
        rsi   = calculate_rsi_fast(prices, self.config.rsi_period)
        bb_u, bb_m, bb_l = calculate_bollinger_bands_fast(
            prices, self.config.bb_period, self.config.bb_std_dev
        )
        pat = detect_patterns_fast(prices, volumes)

        signals = []
        confs   = []
        reasons = []

        # RSI
        cri = rsi[-1]
        if cri < self.config.rsi_buy_threshold:
            signals.append(1); confs.append(self.config.rsi_confidence); reasons.append(f"RSI {cri:.1f}<buy")
        elif cri > self.config.rsi_sell_threshold:
            signals.append(-1); confs.append(self.config.rsi_confidence); reasons.append(f"RSI {cri:.1f}>sell")

        # EMA cross
        if ema_s[-2] <= ema_l[-2] and ema_s[-1] > ema_l[-1]:
            signals.append(1); confs.append(self.config.sma_cross_confidence); reasons.append("EMA cross up")
        elif ema_s[-2] >= ema_l[-2] and ema_s[-1] < ema_l[-1]:
            signals.append(-1); confs.append(self.config.sma_cross_confidence); reasons.append("EMA cross down")

        # Bollinger
        price = prices[-1]
        if price < bb_l[-1] * 0.998:
            signals.append(1); confs.append(self.config.bb_confidence); reasons.append("BB lower")
        elif price > bb_u[-1] * 1.002:
            signals.append(-1); confs.append(self.config.bb_confidence); reasons.append("BB upper")

        # Pattern
        if pat != 0:
            signals.append(pat); confs.append(self.config.pattern_confidence)
            reasons.append("Pattern bullish" if pat>0 else "Pattern bearish")

        # consolidar
        if not signals:
            action = 'HOLD'; overall_conf = 0.5
        else:
            arr = np.array(signals, dtype=np.float32)
            wts = np.array(confs,   dtype=np.float32)
            score = np.average(arr, weights=wts) if wts.sum()>0 else 0.0
            overall_conf = float(wts.mean())
            action = 'BUY' if score > self.config.buy_threshold else 'SELL' if score < -self.config.sell_threshold else 'HOLD'

        # logging
        if action != 'HOLD':
            logger.info(f"📊 TA Sinal: {action} (conf {overall_conf:.2%})")
        details = {
            'rsi': float(cri),
            'sma_short': float(sma_s[-1]),
            'sma_long':  float(sma_l[-1]),
            'ema_short': float(ema_s[-1]),
            'ema_long':  float(ema_l[-1]),
            'bb_upper':  float(bb_u[-1]),
            'bb_middle': float(bb_m[-1]),
            'bb_lower':  float(bb_l[-1]),
            'pattern':   int(pat),
            'reasons':   reasons,
            'calc_ms':   (time.perf_counter()-start)*1000.0
        }

        # cache & contadores
        self.last_calc = now
        self._cached = (action, overall_conf, details)
        self.signals_count[action] += 1
        return action, overall_conf, details

    def get_signal_stats(self) -> Dict:
        total = sum(self.signals_count.values())
        return {
            'total': total,
            'buy': self.signals_count['BUY'],
            'sell': self.signals_count['SELL'],
            'hold': self.signals_count['HOLD']
        }
