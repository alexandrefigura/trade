"""
Análise técnica ultra-rápida com NumPy e Numba - Versão Balanceada
"""
import time
import numpy as np
import numba as nb
from numba import njit
from typing import Tuple, Optional, Dict
from collections import deque
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    Detecção de padrões balanceada
    Retorna: -1 (venda), 0 (neutro), 1 (compra)
    """
    if len(prices) < 200:
        return 0
    
    # Validar dados
    if np.std(prices[-50:]) == 0:
        return 0
    
    # Calcular indicadores de padrão
    # 1. Momentum de curto prazo
    momentum_10 = (prices[-1] - prices[-10]) / prices[-10]
    momentum_20 = (prices[-1] - prices[-20]) / prices[-20]
    
    # 2. Tendência de médio prazo
    sma_50 = np.mean(prices[-50:])
    sma_100 = np.mean(prices[-100:])
    trend = (sma_50 - sma_100) / sma_100
    
    # 3. Volume
    avg_volume = np.mean(volumes[-20:])
    volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
    
    # 4. Volatilidade
    volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
    
    # Score balanceado
    buy_score = 0
    sell_score = 0
    
    # Momentum
    if momentum_10 > 0.005 and momentum_20 > 0.01:
        buy_score += 1
    elif momentum_10 < -0.005 and momentum_20 < -0.01:
        sell_score += 1
    
    # Tendência
    if trend > 0.005 and prices[-1] > sma_50:
        buy_score += 1
    elif trend < -0.005 and prices[-1] < sma_50:
        sell_score += 1
    
    # Volume spike com direção
    if volume_ratio > 1.5:
        if momentum_10 > 0:
            buy_score += 1
        elif momentum_10 < 0:
            sell_score += 1
    
    # Breakout com confirmação
    high_20 = np.max(prices[-20:])
    low_20 = np.min(prices[-20:])
    range_20 = high_20 - low_20
    
    if range_20 > 0:
        position_in_range = (prices[-1] - low_20) / range_20
        
        if position_in_range > 0.9 and momentum_10 > 0 and volume_ratio > 1.2:
            buy_score += 2  # Breakout alta
        elif position_in_range < 0.1 and momentum_10 < 0 and volume_ratio > 1.2:
            sell_score += 2  # Breakout baixa
    
    # Decisão final balanceada
    if buy_score >= 3:
        return 1
    elif sell_score >= 3:
        return -1
    else:
        return 0


def filter_low_volume_and_volatility(
    prices: np.ndarray,
    volumes: np.ndarray,
    min_volume_multiplier: float,
    max_recent_volatility: float
) -> Optional[Tuple[str, float, Dict]]:
    """
    Filtros de qualidade de mercado
    """
    # Filtro de volume
    if len(volumes) >= 20:
        avg_vol20 = np.mean(volumes[-20:])
        if volumes[-1] < avg_vol20 * min_volume_multiplier:
            logger.debug(f"Volume baixo: {volumes[-1]:.2f} < {avg_vol20 * min_volume_multiplier:.2f}")
            return 'HOLD', 0.5, {'reason': 'Volume insuficiente'}
    
    # Filtro de volatilidade
    if len(prices) >= 50:
        recent_vol = np.std(prices[-50:]) / np.mean(prices[-50:])
        if recent_vol > max_recent_volatility:
            logger.debug(f"Volatilidade alta: {recent_vol:.4f} > {max_recent_volatility:.4f}")
            return 'HOLD', 0.5, {'reason': 'Volatilidade excessiva'}
    
    return None


class UltraFastTechnicalAnalysis:
    """Análise técnica com NumPy/Numba - Versão Balanceada"""
    
    def __init__(self, config):
        self.config = config
        self.last_calculation_time = 0
        self.calculation_interval = config.ta_interval_ms / 1000.0
        self._cached_signal = ('HOLD', 0.5, {'cached': True})
        self.signal_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Anti-bias
        self.signal_history = deque(maxlen=20)
        self.last_signal_time = {'BUY': 0, 'SELL': 0}
        self.signal_cooldown = config.signal_cooldown_ms / 1000.0 if hasattr(config, 'signal_cooldown_ms') else 10.0
    
    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> tuple[str, float, Dict]:
        """Análise técnica completa balanceada"""
        now = time.time()
        
        # Validações básicas
        if len(prices) < 200:
            return 'HOLD', 0.5, {'reason': 'Dados insuficientes'}
        
        if np.std(prices[-50:]) == 0:
            return 'HOLD', 0.5, {'reason': 'Sem variação de preço'}
        
        if np.any(prices <= 0) or np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return 'HOLD', 0.5, {'reason': 'Dados inválidos'}
        
        # Respeitar intervalo de cálculo
        if now - self.last_calculation_time < self.calculation_interval:
            return self._cached_signal
        
        # Filtros de qualidade (exceto em debug)
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
        
        # Calcular todos os indicadores
        indicators = self._calculate_all_indicators(prices, volumes)
        
        # Gerar sinais individuais
        signals, confidences, reasons = self._generate_signals(indicators, prices)
        
        # Consolidar com lógica anti-bias
        action, confidence = self._consolidate_with_antibias(signals, confidences)
        
        # Aplicar cooldown se necessário
        action, confidence = self._apply_signal_cooldown(action, confidence, now)
        
        # Atualizar estatísticas
        self.signal_count[action] += 1
        self.signal_history.append(action)
        
        # Log significativo
        if action != 'HOLD' or (self.config.debug_mode and len(reasons) > 0):
            logger.info(f"📊 TA Sinal: {action} ({confidence:.2%}) - RSI: {indicators['rsi']:.1f}")
            if reasons:
                logger.debug(f"   Sinais: {', '.join(reasons)}")
        
        # Preparar detalhes
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        details = self._prepare_details(indicators, prices[-1], elapsed_ms, reasons)
        
        self.last_calculation_time = now
        self._cached_signal = (action, confidence, details)
        
        return action, confidence, details
    
    def _calculate_all_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """Calcula todos os indicadores técnicos"""
        # Moving Averages
        sma_short = calculate_sma_fast(prices, self.config.sma_short_period)
        sma_long = calculate_sma_fast(prices, self.config.sma_long_period)
        ema_short = calculate_ema_fast(prices, self.config.ema_short_period)
        ema_long = calculate_ema_fast(prices, self.config.ema_long_period)
        
        # RSI
        rsi_vals = calculate_rsi_fast(prices, self.config.rsi_period)
        
        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands_fast(
            prices, self.config.bb_period, self.config.bb_std_dev
        )
        
        # Patterns
        pattern_signal = detect_patterns_fast(prices, volumes)
        
        # MACD (simples)
        if len(prices) >= 26:
            macd_line = ema_short[-1] - ema_long[-1]
            macd_signal = np.mean([ema_short[-i] - ema_long[-i] for i in range(1, 10)])
            macd_histogram = macd_line - macd_signal
        else:
            macd_line = macd_signal = macd_histogram = 0
        
        return {
            'sma_short': sma_short,
            'sma_long': sma_long,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'rsi': rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50.0,
            'rsi_vals': rsi_vals,
            'bb_upper': bb_upper,
            'bb_mid': bb_mid,
            'bb_lower': bb_lower,
            'pattern': pattern_signal,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    def _generate_signals(self, indicators: Dict, prices: np.ndarray) -> Tuple[list, list, list]:
        """Gera sinais balanceados de cada indicador"""
        signals = []
        confidences = []
        reasons = []
        
        current_price = prices[-1]
        
        # 1. RSI - Simétrico
        rsi = indicators['rsi']
        if self.config.rsi_buy_threshold < rsi < self.config.rsi_sell_threshold:
            # RSI neutro, não gerar sinal
            pass
        elif rsi <= self.config.rsi_buy_threshold:
            strength = (self.config.rsi_buy_threshold - rsi) / self.config.rsi_buy_threshold
            signals.append(1)
            confidences.append(self.config.rsi_confidence * (1 + strength * 0.2))
            reasons.append(f"RSI oversold: {rsi:.1f}")
        elif rsi >= self.config.rsi_sell_threshold:
            strength = (rsi - self.config.rsi_sell_threshold) / (100 - self.config.rsi_sell_threshold)
            signals.append(-1)
            confidences.append(self.config.rsi_confidence * (1 + strength * 0.2))
            reasons.append(f"RSI overbought: {rsi:.1f}")
        
        # 2. EMA Cross - Balanceado
        if not np.isnan(indicators['ema_short'][-1]) and not np.isnan(indicators['ema_long'][-1]):
            ema_diff = (indicators['ema_short'][-1] - indicators['ema_long'][-1]) / indicators['ema_long'][-1]
            
            # Só gerar sinal em cruzamentos claros
            if len(prices) > 2:
                prev_diff = (indicators['ema_short'][-2] - indicators['ema_long'][-2]) / indicators['ema_long'][-2]
                
                if prev_diff <= 0 and ema_diff > 0.0002:  # Cruzamento para cima
                    signals.append(1)
                    confidences.append(self.config.sma_cross_confidence)
                    reasons.append(f"EMA bullish cross")
                elif prev_diff >= 0 and ema_diff < -0.0002:  # Cruzamento para baixo
                    signals.append(-1)
                    confidences.append(self.config.sma_cross_confidence)
                    reasons.append(f"EMA bearish cross")
        
        # 3. Bollinger Bands - Balanceado
        bb_upper = indicators['bb_upper'][-1]
        bb_lower = indicators['bb_lower'][-1]
        bb_mid = indicators['bb_mid'][-1]
        
        if not np.isnan(bb_lower) and not np.isnan(bb_upper):
            bb_width = bb_upper - bb_lower
            
            if bb_width > 0:
                # Posição relativa nas bandas
                bb_position = (current_price - bb_lower) / bb_width
                
                if bb_position < 0.05:  # Muito próximo da banda inferior
                    signals.append(1)
                    confidences.append(self.config.bb_confidence)
                    reasons.append(f"Price at lower BB")
                elif bb_position > 0.95:  # Muito próximo da banda superior
                    signals.append(-1)
                    confidences.append(self.config.bb_confidence)
                    reasons.append(f"Price at upper BB")
        
        # 4. MACD - Se disponível
        if indicators['macd_histogram'] != 0:
            if indicators['macd_histogram'] > 0 and indicators['macd_line'] > 0:
                signals.append(1)
                confidences.append(0.6)
                reasons.append("MACD bullish")
            elif indicators['macd_histogram'] < 0 and indicators['macd_line'] < 0:
                signals.append(-1)
                confidences.append(0.6)
                reasons.append("MACD bearish")
        
        # 5. Pattern Detection
        if indicators['pattern'] != 0:
            signals.append(indicators['pattern'])
            confidences.append(self.config.pattern_confidence)
            reasons.append(f"Pattern: {'Bullish' if indicators['pattern'] > 0 else 'Bearish'}")
        
        return signals, confidences, reasons
    
    def _consolidate_with_antibias(self, signals: list, confidences: list) -> Tuple[str, float]:
        """Consolida sinais com lógica anti-bias"""
        if not signals:
            return 'HOLD', 0.5
        
        # Converter para arrays
        signals_array = np.array(signals, dtype=np.float32)
        confs_array = np.array(confidences, dtype=np.float32)
        
        # Score ponderado
        if np.sum(confs_array) > 0:
            weighted = np.average(signals_array, weights=confs_array)
            avg_confidence = np.mean(confs_array)
        else:
            weighted = 0
            avg_confidence = 0.5
        
        # Verificar histórico recente para anti-bias
        if len(self.signal_history) >= 10:
            recent = list(self.signal_history)[-10:]
            buy_count = recent.count('BUY')
            sell_count = recent.count('SELL')
            
            # Se muito enviesado, ajustar threshold
            if buy_count > 7:  # Muitos BUYs
                buy_threshold = self.config.buy_threshold * 1.5
                sell_threshold = self.config.sell_threshold * 0.7
            elif sell_count > 7:  # Muitos SELLs
                buy_threshold = self.config.buy_threshold * 0.7
                sell_threshold = self.config.sell_threshold * 1.5
            else:
                buy_threshold = self.config.buy_threshold
                sell_threshold = self.config.sell_threshold
        else:
            buy_threshold = self.config.buy_threshold
            sell_threshold = self.config.sell_threshold
        
        # Decisão final
        if weighted > buy_threshold:
            action = 'BUY'
        elif weighted < -sell_threshold:
            action = 'SELL'
        else:
            action = 'HOLD'
            avg_confidence *= 0.7  # Reduzir confiança em HOLD
        
        return action, avg_confidence
    
    def _apply_signal_cooldown(self, action: str, confidence: float, current_time: float) -> Tuple[str, float]:
        """Aplica cooldown entre sinais do mesmo tipo"""
        if action in ['BUY', 'SELL']:
            last_time = self.last_signal_time.get(action, 0)
            time_since_last = current_time - last_time
            
            if time_since_last < self.signal_cooldown:
                # Ainda em cooldown
                remaining = self.signal_cooldown - time_since_last
                logger.debug(f"Sinal {action} em cooldown por mais {remaining:.1f}s")
                return 'HOLD', confidence * 0.5
            else:
                # Atualizar tempo
                self.last_signal_time[action] = current_time
        
        return action, confidence
    
    def _prepare_details(self, indicators: Dict, current_price: float, elapsed_ms: float, reasons: list) -> Dict:
        """Prepara detalhes para retorno"""
        return {
            'rsi': float(indicators['rsi']),
            'sma_short': float(indicators['sma_short'][-1]) if not np.isnan(indicators['sma_short'][-1]) else 0.0,
            'sma_long': float(indicators['sma_long'][-1]) if not np.isnan(indicators['sma_long'][-1]) else 0.0,
            'ema_short': float(indicators['ema_short'][-1]) if not np.isnan(indicators['ema_short'][-1]) else 0.0,
            'ema_long': float(indicators['ema_long'][-1]) if not np.isnan(indicators['ema_long'][-1]) else 0.0,
            'bb_upper': float(indicators['bb_upper'][-1]) if not np.isnan(indicators['bb_upper'][-1]) else 0.0,
            'bb_lower': float(indicators['bb_lower'][-1]) if not np.isnan(indicators['bb_lower'][-1]) else 0.0,
            'bb_position': self._calculate_bb_position(current_price, indicators),
            'pattern': indicators['pattern'],
            'macd_histogram': float(indicators['macd_histogram']),
            'calc_time_ms': elapsed_ms,
            'current_price': float(current_price),
            'reasons': reasons,
            'signal_balance': self._calculate_signal_balance()
        }
    
    def _calculate_bb_position(self, price: float, indicators: Dict) -> float:
        """Calcula posição relativa nas Bollinger Bands"""
        bb_upper = indicators['bb_upper'][-1]
        bb_lower = indicators['bb_lower'][-1]
        
        if not np.isnan(bb_upper) and not np.isnan(bb_lower) and bb_upper > bb_lower:
            return (price - bb_lower) / (bb_upper - bb_lower)
        return 0.5
    
    def _calculate_signal_balance(self) -> float:
        """Calcula balanço de sinais (1.0 = perfeito, 0 = totalmente enviesado)"""
        total = sum(self.signal_count.values())
        if total == 0:
            return 1.0
        
        buy_ratio = self.signal_count['BUY'] / total
        sell_ratio = self.signal_count['SELL'] / total
        
        if buy_ratio + sell_ratio == 0:
            return 0.0
        
        # Quanto mais próximo de 0.5/0.5, melhor
        balance = 1.0 - abs(buy_ratio - sell_ratio)
        return balance
    
    def get_signal_stats(self) -> Dict:
        """Retorna estatísticas dos sinais"""
        total = sum(self.signal_count.values())
        balance = self._calculate_signal_balance()
        
        return {
            'total_signals': total,
            'buy_signals': self.signal_count['BUY'],
            'sell_signals': self.signal_count['SELL'],
            'hold_signals': self.signal_count['HOLD'],
            'buy_percentage': self.signal_count['BUY'] / total * 100 if total > 0 else 0,
            'sell_percentage': self.signal_count['SELL'] / total * 100 if total > 0 else 0,
            'signal_balance': balance,
            'balance_status': 'Balanceado' if balance > 0.7 else 'Enviesado',
            'recent_signals': list(self.signal_history)[-10:] if self.signal_history else []
        }