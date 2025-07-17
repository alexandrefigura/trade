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

@nb.jit(nopython=True, cache=True, fastmath=True)
def calculate_sma_fast(prices: np.ndarray, period: int) -> np.ndarray:
    """SMA ultra-rápida com Numba"""
    n = len(prices)
    sma = np.empty(n, dtype=np.float32)
    sma[:period-1] = np.nan
    
    # Primeira média
    sma[period-1] = np.mean(prices[:period])
    
    # Sliding window otimizado
    for i in range(period, n):
        sma[i] = sma[i-1] + (prices[i] - prices[i-period]) / period
    
    return sma


@nb.jit(nopython=True, cache=True, fastmath=True)
def calculate_ema_fast(prices: np.ndarray, period: int) -> np.ndarray:
    """EMA ultra-rápida com Numba"""
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema


@nb.jit(nopython=True, cache=True, fastmath=True)
def calculate_rsi_fast(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI ultra-rápido com Numba"""
    n = len(prices)
    rsi = np.empty(n, dtype=np.float32)
    rsi[:period] = np.nan
    
    # Calcular diferenças
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)
    
    # Médias iniciais
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # RSI inicial
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    else:
        rsi[period] = 100
    
    # Cálculo incremental
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    return rsi


@nb.jit(nopython=True, cache=True, fastmath=True)
def calculate_bollinger_bands_fast(
    prices: np.ndarray, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands ultra-rápidas"""
    sma = calculate_sma_fast(prices, period)
    
    # Desvio padrão rolling
    n = len(prices)
    std = np.empty(n, dtype=np.float32)
    std[:period-1] = np.nan
    
    for i in range(period-1, n):
        std[i] = np.std(prices[i-period+1:i+1])
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower


@nb.jit(nopython=True, cache=True, fastmath=True)
def detect_patterns_fast(prices: np.ndarray, volumes: np.ndarray) -> int:
    """
    Detecção de padrões ultra-rápida
    Retorna: -1 (venda), 0 (neutro), 1 (compra)
    """
    if len(prices) < 200:
        return 0
    
    # Validar dados
    if np.std(prices[-50:]) == 0:
        return 0
    
    # Momentum
    momentum = (prices[-1] - prices[-10]) / prices[-10]
    
    # Volume anormal
    avg_volume = np.mean(volumes[-20:])
    volume_spike = volumes[-1] > avg_volume * 1.5
    
    # Breakout detection
    high_20 = np.max(prices[-20:])
    low_20 = np.min(prices[-20:])
    
    if prices[-1] > high_20 * 0.995 and volume_spike and momentum > 0.001:
        return 1  # Sinal de compra
    elif prices[-1] < low_20 * 1.005 and volume_spike and momentum < -0.001:
        return -1  # Sinal de venda
    
    return 0  # Neutro


def filter_low_volume_and_volatility(
    prices: np.ndarray,
    volumes: np.ndarray,
    min_volume_multiplier: float,
    max_recent_volatility: float
) -> Optional[Tuple[str, float, Dict]]:
    """
    Cancela sinais se o volume estiver baixo ou a volatilidade muito alta.
    Retorna ('HOLD', 0.5, {...}) para cancelar, ou None para continuar.
    """
    # Filtro de volume
    if len(volumes) >= 20:
        avg_vol20 = np.mean(volumes[-20:])
        if volumes[-1] < avg_vol20 * min_volume_multiplier:
            logger.debug(f"Volume baixo: {volumes[-1]:.2f} < {avg_vol20 * min_volume_multiplier:.2f}")
            return 'HOLD', 0.5, {'reason': 'Filtrado por volume'}
    
    # Filtro de volatilidade recente
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
        self.last_calculation_time = 0
        self.calculation_interval = config.ta_interval_ms / 1000.0
        self._cached_signal = ('HOLD', 0.5, {'cached': True})
        self.signal_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> Tuple[str, float, Dict]:
        """Análise técnica completa ultra-rápida"""
        now = time.time()
        
        # Validações
        if len(prices) < 200:
            return 'HOLD', 0.5, {'reason': 'Dados insuficientes'}
        
        if np.std(prices[-50:]) == 0:
            return 'HOLD', 0.5, {'reason': 'Dados inválidos - sem variação'}
        
        if np.any(prices <= 0) or np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return 'HOLD', 0.5, {'reason': 'Dados inválidos - valores anormais'}
        
        # Respeita intervalo mínimo entre cálculos
        if now - self.last_calculation_time < self.calculation_interval:
            return self._cached_signal
        
        # Filtros (se não estiver em debug)
        if not self.config.debug_mode:
            filter_result = filter_low_volume_and_volatility(
                prices,
                volumes,
                self.config.min_volume_multiplier,
                self.config.max_recent_volatility
            )
            if filter_result is not None:
                return filter_result
        
        start_ts = time.perf_counter()
        
        # Calcular indicadores
        sma_short = calculate_sma_fast(prices, self.config.sma_short_period)
        sma_long = calculate_sma_fast(prices, self.config.sma_long_period)
        ema_short = calculate_ema_fast(prices, self.config.ema_short_period)
        ema_long = calculate_ema_fast(prices, self.config.ema_long_period)
        rsi_vals = calculate_rsi_fast(prices, self.config.rsi_period)
        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands_fast(
            prices, self.config.bb_period, self.config.bb_std_dev
        )
        pattern_signal = detect_patterns_fast(prices, volumes)
        
        # Gerar sinais
        signals = []
        confs = []
        signal_reasons = []
        
        # RSI
        current_rsi = rsi_vals[-1]
        if not np.isnan(current_rsi) and 0 <= current_rsi <= 100:
            if current_rsi < self.config.rsi_buy_threshold:
                signals.append(1)
                confs.append(self.config.rsi_confidence)
                signal_reasons.append(f"RSI oversold: {current_rsi:.1f}")
            elif current_rsi > self.config.rsi_sell_threshold:
                signals.append(-1)
                confs.append(self.config.rsi_confidence)
                signal_reasons.append(f"RSI overbought: {current_rsi:.1f}")
        
        # EMA Cross
        if not np.isnan(ema_short[-1]) and not np.isnan(ema_long[-1]):
            if ema_short[-1] > ema_long[-1] and ema_short[-2] <= ema_long[-2]:
                cross_strength = abs(ema_short[-1] - ema_long[-1]) / ema_long[-1]
                if cross_strength > 0.0002:
                    signals.append(1)
                    confs.append(self.config.sma_cross_confidence)
                    signal_reasons.append("EMA bullish cross")
            elif ema_short[-1] < ema_long[-1] and ema_short[-2] >= ema_long[-2]:
                cross_strength = abs(ema_long[-1] - ema_short[-1]) / ema_long[-1]
                if cross_strength > 0.0002:
                    signals.append(-1)
                    confs.append(self.config.sma_cross_confidence)
                    signal_reasons.append("EMA bearish cross")
        
        # Bollinger Bands
        price = prices[-1]
        if not np.isnan(bb_lower[-1]) and not np.isnan(bb_upper[-1]):
            if price < bb_lower[-1] * 0.998:
                signals.append(1)
                confs.append(self.config.bb_confidence)
                signal_reasons.append("BB oversold")
            elif price > bb_upper[-1] * 1.002:
                signals.append(-1)
                confs.append(self.config.bb_confidence)
                signal_reasons.append("BB overbought")
        
        # Pattern
        if pattern_signal != 0:
            signals.append(pattern_signal)
            confs.append(self.config.pattern_confidence)
            signal_reasons.append(f"Pattern: {'Bullish' if pattern_signal > 0 else 'Bearish'}")
        
        # Consolidar
        if not signals:
            action = 'HOLD'
            overall_conf = 0.5
        else:
            signals_array = np.array(signals, dtype=np.float32)
            confs_array = np.array(confs, dtype=np.float32)
            
            if np.sum(confs_array) > 0:
                weighted = np.average(signals_array, weights=confs_array)
                overall_conf = float(np.mean(confs_array))
            else:
                weighted = 0
                overall_conf = 0.5
            
            if weighted > self.config.buy_threshold:
                action = 'BUY'
            elif weighted < -self.config.sell_threshold:
                action = 'SELL'
            else:
                action = 'HOLD'
        
        # Atualizar contadores
        self.signal_count[action] += 1
        
        # Log se significativo
        if action != 'HOLD':
            logger.info(f"📊 TA Sinal: {action} (conf: {overall_conf:.2%}) - RSI: {current_rsi:.1f}")
            logger.debug(f"   Razões: {', '.join(signal_reasons)}")
        
        # Detalhes
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        details = {
            'rsi': float(current_rsi) if not np.isnan(current_rsi) else 50.0,
            'sma_short': float(sma_short[-1]) if not np.isnan(sma_short[-1]) else 0.0,
            'sma_long': float(sma_long[-1]) if not np.isnan(sma_long[-1]) else 0.0,
            'ema_short': float(ema_short[-1]) if not np.isnan(ema_short[-1]) else 0.0,
            'ema_long': float(ema_long[-1]) if not np.isnan(ema_long[-1]) else 0.0,
            'bb_upper': float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else 0.0,
            'bb_lower': float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else 0.0,
            'bb_position': ((price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) 
                           if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]) 
                           and bb_upper[-1] > bb_lower[-1] else 0.5),
            'pattern': pattern_signal,
            'calc_time_ms': elapsed_ms,
            'current_price': float(price),
            'reasons': signal_reasons
        }
        
        self.last_calculation_time = now
        self._cached_signal = (action, overall_conf, details)
        return action, overall_conf, details
    
    def get_signal_stats(self) -> Dict:
        """Retorna estatísticas dos sinais"""
        total = sum(self.signal_count.values())
        return {
            'total_signals': total,
            'buy_signals': self.signal_count['BUY'],
            'sell_signals': self.signal_count['SELL'],
            'hold_signals': self.signal_count['HOLD'],
            'buy_percentage': self.signal_count['BUY'] / total * 100 if total > 0 else 0,
            'sell_percentage': self.signal_count['SELL'] / total * 100 if total > 0 else 0
        }
