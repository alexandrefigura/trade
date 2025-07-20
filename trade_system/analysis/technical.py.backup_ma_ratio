# trade_system/analysis/technical.py

"""
Análise técnica ultra-rápida com NumPy e Numba
"""

import time
from typing import Tuple, Optional, Dict

import numpy as np
from numba import njit

from trade_system.logging_config import get_logger

logger = get_logger(__name__)


# ===========================
# FUNÇÕES NUMBA ULTRA-RÁPIDAS
# ===========================

@njit(cache=True, fastmath=True)
def calculate_sma_fast(prices: np.ndarray, period: int) -> np.ndarray:
    """Calcula SMA em O(n) usando janela deslizante."""
    n = prices.shape[0]
    sma = np.empty(n, dtype=np.float32)
    sma[: period - 1] = np.nan
    acc = 0.0
    for i in range(period):
        acc += prices[i]
    sma[period - 1] = acc / period
    for i in range(period, n):
        acc += prices[i] - prices[i - period]
        sma[i] = acc / period
    return sma


@njit(cache=True, fastmath=True)
def calculate_ema_fast(prices: np.ndarray, period: int) -> np.ndarray:
    """Calcula EMA com fator de suavização exponencial."""
    alpha = 2.0 / (period + 1)
    n = prices.shape[0]
    ema = np.empty(n, dtype=np.float32)
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


@njit(cache=True, fastmath=True)
def calculate_rsi_fast(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calcula RSI clássico de Wilder."""
    n = prices.shape[0]
    rsi = np.empty(n, dtype=np.float32)
    rsi[: period] = np.nan

    gains = np.zeros(n - 1, dtype=np.float32)
    losses = np.zeros(n - 1, dtype=np.float32)
    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        gains[i - 1] = diff if diff > 0 else 0.0
        losses[i - 1] = -diff if diff < 0 else 0.0

    avg_gain = np.sum(gains[:period]) / period
    avg_loss = np.sum(losses[:period]) / period
    rs = avg_gain / avg_loss if avg_loss != 0.0 else np.inf
    rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0.0 else np.inf
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@njit(cache=True, fastmath=True)
def calculate_bollinger_bands_fast(
    prices: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula bandas de Bollinger (upper, middle=SMA, lower)."""
    sma = calculate_sma_fast(prices, period)
    n = prices.shape[0]
    std = np.empty(n, dtype=np.float32)
    std[: period - 1] = np.nan

    for i in range(period - 1, n):
        acc = 0.0
        acc2 = 0.0
        for j in range(i - period + 1, i + 1):
            v = prices[j]
            acc += v
            acc2 += v * v
        mean = acc / period
        var = (acc2 / period) - (mean * mean)
        std[i] = np.sqrt(var) if var > 0.0 else 0.0

    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


@njit(cache=True, fastmath=True)
def detect_patterns_fast(prices: np.ndarray, volumes: np.ndarray) -> int:
    """
    Detecta padrões simples de rompimento:
      +1 para rompimento de topo com volume,
      -1 para rompimento de fundo com volume,
       0 caso contrário.
    """
    n = prices.shape[0]
    if n < 200:
        return 0

    momentum = (prices[-1] - prices[-10]) / prices[-10]
    avg_vol = np.mean(volumes[-20:]) if volumes.size >= 20 else 0.0
    spike = volumes[-1] > avg_vol * 1.5

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
    max_recent_volatility: float,
) -> Optional[Tuple[str, float, Dict]]:
    """
    Filtra cenário de mercado inseguro:
      - Volume atual muito baixo em relação à média
      - Volatilidade recente acima do limite
    Retorna sinal HOLD caso filtre, senão None.
    """
    if volumes.size >= 20:
        avg_vol20 = float(np.mean(volumes[-20:]))
        if volumes[-1] < avg_vol20 * min_volume_multiplier:
            logger.debug(f"Volume baixo: {volumes[-1]:.2f} < {avg_vol20 * min_volume_multiplier:.2f}")
            return 'HOLD', 0.5, {'reason': 'Filtrado por volume baixo'}

    if prices.size >= 50:
        recent_vol = float(np.std(prices[-50:]) / np.mean(prices[-50:]))
        if recent_vol > max_recent_volatility:
            logger.debug(f"Volatilidade alta: {recent_vol:.4f} > {max_recent_volatility:.4f}")
            return 'HOLD', 0.5, {'reason': 'Filtrado por alta volatilidade'}

    return None


class UltraFastTechnicalAnalysis:
    """Análise técnica com NumPy/Numba para máxima velocidade."""

    def __init__(self, config):
        self.config = config
        self.last_calc = 0.0
        self.interval = config.ta_interval_ms / 1000.0
        self._cache: Tuple[str, float, Dict] = ('HOLD', 0.5, {'cached': True})
        self.signals_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}

    def analyze(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> Tuple[str, float, Dict]:
        """
        Executa todos os indicadores e combina sinais:
          - SMA & EMA crosses
          - RSI thresholds
          - Bollinger Bands
          - Padrões de volume/preço
        Retorna (action, confidence, details).
        """
        now = time.time()
        # Checa se há dados suficientes ou sem variação
        if (
            prices.size < self.config.sma_long_period
            or np.std(prices[-self.config.sma_long_period :]) == 0.0
        ):
            return 'HOLD', 0.5, {'reason': 'Dados insuficientes ou sem variação'}

        # Reuso de resultado dentro do intervalo
        if now - self.last_calc < self.interval:
            return self._cache

        # Filtro de segurança (volume/volatilidade) quando não está em debug
        if not self.config.debug_mode:
            filt = filter_low_volume_and_volatility(
                prices,
                volumes,
                self.config.min_volume_multiplier,
                self.config.max_recent_volatility,
            )
            if filt is not None:
                return filt

        start = time.perf_counter()

        # Calcula indicadores fundamentais
        sma_s = calculate_sma_fast(prices, self.config.sma_short_period)
        sma_l = calculate_sma_fast(prices, self.config.sma_long_period)
        ema_s = calculate_ema_fast(prices, self.config.ema_short_period)
        ema_l = calculate_ema_fast(prices, self.config.ema_long_period)
        rsi = calculate_rsi_fast(prices, self.config.rsi_period)
        bb_u, bb_m, bb_l = calculate_bollinger_bands_fast(
            prices, self.config.bb_period, self.config.bb_std_dev
        )
        pat = detect_patterns_fast(prices, volumes)

        signals, confs, reasons = [], [], []

        # RSI
        last_rsi = float(rsi[-1])
        if last_rsi < self.config.rsi_buy_threshold:
            signals.append(1)
            confs.append(self.config.rsi_confidence)
            reasons.append(f"RSI {last_rsi:.1f}<buy")
        elif last_rsi > self.config.rsi_sell_threshold:
            signals.append(-1)
            confs.append(self.config.rsi_confidence)
            reasons.append(f"RSI {last_rsi:.1f}>sell")

        # EMA cross
        if ema_s[-2] <= ema_l[-2] < ema_s[-1] > ema_l[-1]:
            signals.append(1)
            confs.append(self.config.sma_cross_confidence)
            reasons.append("EMA cross up")
        elif ema_s[-2] >= ema_l[-2] > ema_s[-1] < ema_l[-1]:
            signals.append(-1)
            confs.append(self.config.sma_cross_confidence)
            reasons.append("EMA cross down")

        # Bollinger Bands
        price = float(prices[-1])
        if price < bb_l[-1] * 0.998:
            signals.append(1)
            confs.append(self.config.bb_confidence)
            reasons.append("BB lower")
        elif price > bb_u[-1] * 1.002:
            signals.append(-1)
            confs.append(self.config.bb_confidence)
            reasons.append("BB upper")

        # Padrões detectados
        if pat != 0:
            signals.append(pat)
            confs.append(self.config.pattern_confidence)
            reasons.append("Pattern bullish" if pat > 0 else "Pattern bearish")

        # Combina sinais
        if not signals:
            action, overall_conf = 'HOLD', 0.5
        else:
            arr = np.array(signals, dtype=np.float32)
            wts = np.array(confs, dtype=np.float32)
            score = float(np.average(arr, weights=wts)) if wts.sum() > 0 else 0.0
            overall_conf = float(wts.mean())
            if score > self.config.buy_threshold:
                action = 'BUY'
            elif score < -self.config.sell_threshold:
                action = 'SELL'
            else:
                action = 'HOLD'

        # Log e estatísticas
        if action != 'HOLD':
            logger.info(f"📊 TA Sinal: {action} (conf {overall_conf:.2%})")
        self.signals_count[action] += 1

        # Monta detalhes para inspeção
        details = {
            'rsi': last_rsi,
            'sma_short': float(sma_s[-1]),
            'sma_long': float(sma_l[-1]),
            'ema_short': float(ema_s[-1]),
            'ema_long': float(ema_l[-1]),
            'bb_upper': float(bb_u[-1]),
            'bb_middle': float(bb_m[-1]),
            'bb_lower': float(bb_l[-1]),
            'pattern': int(pat),
            'reasons': reasons,
            'calc_ms': (time.perf_counter() - start) * 1000.0,
        }

        # Atualiza cache e timestamp
        self.last_calc = now
        self._cache = (action, overall_conf, details)
        return action, overall_conf, details

    def get_signal_stats(self) -> Dict[str, int]:
        """Retorna contagem de sinais gerados."""
        total = sum(self.signals_count.values())
        return {
            'total': total,
            'buy': self.signals_count['BUY'],
            'sell': self.signals_count['SELL'],
            'hold': self.signals_count['HOLD'],
        }
