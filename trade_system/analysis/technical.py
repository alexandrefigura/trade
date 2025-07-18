"""Análise técnica otimizada com Numba"""
import numpy as np
import pandas as pd
import talib
from numba import jit, prange
import logging
from typing import Dict, Any, Tuple, Optional

@jit(nopython=True)
def calculate_rsi_fast(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI otimizado com Numba para performance máxima"""
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)
    
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        rs = 100
    else:
        rs = up / down
    
    rsi = np.zeros_like(prices)
    rsi[:period] = np.nan
    rsi[period] = 100. - 100. / (1. + rs)
    
    for i in range(period + 1, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        
        if down == 0:
            rs = 100
        else:
            rs = up / down
            
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi

@jit(nopython=True)
def calculate_bollinger_bands_fast(prices: np.ndarray, period: int = 20, 
                                  std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands otimizado com Numba"""
    middle = np.zeros_like(prices)
    upper = np.zeros_like(prices)
    lower = np.zeros_like(prices)
    
    for i in range(period - 1, len(prices)):
        slice_prices = prices[i - period + 1:i + 1]
        mean = np.mean(slice_prices)
        std = np.std(slice_prices)
        
        middle[i] = mean
        upper[i] = mean + (std_dev * std)
        lower[i] = mean - (std_dev * std)
    
    return upper, middle, lower

@jit(nopython=True)
def calculate_support_resistance_fast(highs: np.ndarray, lows: np.ndarray, 
                                    lookback: int = 20) -> Tuple[float, float]:
    """Calcula suporte e resistência de forma otimizada"""
    if len(highs) < lookback:
        return lows[-1], highs[-1]
    
    recent_lows = lows[-lookback:]
    recent_highs = highs[-lookback:]
    
    # Método simples mas eficaz
    support = np.percentile(recent_lows, 25)
    resistance = np.percentile(recent_highs, 75)
    
    return support, resistance

class TechnicalAnalyzer:
    """Análise técnica de alta performance"""
    
    def __init__(self, config: Any):
        self.config = config.ta if hasattr(config, 'ta') else config.get('ta', {})
        self.logger = logging.getLogger(__name__)
        self.indicators_cache = {}
        
    def analyze(self, candles: pd.DataFrame) -> Dict[str, Any]:
        """
        Análise técnica completa
        
        Args:
            candles: DataFrame com OHLCV
            
        Returns:
            Dict com sinal, confiança e indicadores
        """
        try:
            if len(candles) < 50:
                return self._empty_analysis()
            
            # Extrair arrays numpy para performance
            close = candles['close'].values
            high = candles['high'].values
            low = candles['low'].values
            volume = candles['volume'].values
            
            # Calcular todos os indicadores
            indicators = self._calculate_indicators(close, high, low, volume)
            
            # Gerar sinal baseado nos indicadores
            signal, confidence = self._generate_signal(indicators, close[-1])
            
            self.logger.info(f"📊 TA Sinal: {signal} (conf {confidence:.2%})")
            
            return {
                'signal': signal,
                'confidence': confidence,
                'indicators': indicators,
                'price': close[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise técnica: {e}")
            return self._empty_analysis()
    
    def _calculate_indicators(self, close: np.ndarray, high: np.ndarray, 
                            low: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """Calcula todos os indicadores técnicos"""
        indicators = {}
        
        # RSI
        rsi_period = self.config.get('rsi_period', 14)
        indicators['rsi'] = calculate_rsi_fast(close, rsi_period)[-1]
        
        # MACD
        macd_fast = self.config.get('macd_fast', 12)
        macd_slow = self.config.get('macd_slow', 26)
        macd_signal = self.config.get('macd_signal', 9)
        
        macd_line, signal_line, macd_hist = talib.MACD(
            close, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
        )
        indicators['macd'] = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
        indicators['macd_signal'] = signal_line[-1] if not np.isnan(signal_line[-1]) else 0
        indicators['macd_hist'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
        
        # Bollinger Bands
        bb_period = self.config.get('bb_period', 20)
        bb_std = self.config.get('bb_std', 2.0)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands_fast(close, bb_period, bb_std)
        
        indicators['bb_upper'] = bb_upper[-1]
        indicators['bb_middle'] = bb_middle[-1]
        indicators['bb_lower'] = bb_lower[-1]
        indicators['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
        
        # EMAs
        ema_short = self.config.get('ema_short', 9)
        ema_long = self.config.get('ema_long', 21)
        indicators['ema_short'] = talib.EMA(close, timeperiod=ema_short)[-1]
        indicators['ema_long'] = talib.EMA(close, timeperiod=ema_long)[-1]
        indicators['ema_diff'] = (indicators['ema_short'] - indicators['ema_long']) / indicators['ema_long']
        
        # Momentum (feature importante que estava faltando!)
        momentum_period = self.config.get('momentum_period', 10)
        indicators['momentum'] = talib.MOM(close, timeperiod=momentum_period)[-1]
        
        # ATR (Volatilidade)
        atr_period = self.config.get('atr_period', 14)
        indicators['atr'] = talib.ATR(high, low, close, timeperiod=atr_period)[-1]
        indicators['volatility'] = indicators['atr'] / close[-1]
        
        # Support & Resistance
        support, resistance = calculate_support_resistance_fast(high, low)
        indicators['support'] = support
        indicators['resistance'] = resistance
        
        # Volume indicators
        indicators['volume_ratio'] = volume[-1] / np.mean(volume[-20:])
        indicators['volume_sma'] = np.mean(volume[-20:])
        
        # Price action
        indicators['price_change'] = (close[-1] - close[-2]) / close[-2]
        indicators['high_low_ratio'] = (high[-1] - low[-1]) / close[-1]
        
        # Trend strength
        indicators['trend_strength'] = self._calculate_trend_strength(close)
        
        # Volume profile
        indicators['volume_profile'] = self._calculate_volume_profile(volume, close)
        
        return indicators
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calcula força da tendência usando regressão linear"""
        if len(prices) < 20:
            return 0.5
        
        # Usar últimos 20 períodos
        y = prices[-20:]
        x = np.arange(len(y))
        
        # Regressão linear simples
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalizar pela volatilidade
        normalized_slope = np.tanh(slope / np.std(y) * 10)
        
        # Converter para 0-1
        return (normalized_slope + 1) / 2
    
    def _calculate_volume_profile(self, volume: np.ndarray, prices: np.ndarray) -> float:
        """Calcula perfil de volume (VWAP simplificado)"""
        if len(volume) < 20:
            return 0.5
        
        # VWAP dos últimos 20 períodos
        vwap = np.sum(volume[-20:] * prices[-20:]) / np.sum(volume[-20:])
        current_price = prices[-1]
        
        # Diferença normalizada
        diff = (current_price - vwap) / vwap
        
        # Converter para score 0-1
        return np.tanh(diff * 100) / 2 + 0.5
    
    def _generate_signal(self, indicators: Dict[str, float], current_price: float) -> Tuple[str, float]:
        """Gera sinal de trading baseado nos indicadores"""
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        # RSI
        rsi = indicators['rsi']
        rsi_buy = self.config.get('rsi_buy_threshold', 30)
        rsi_sell = self.config.get('rsi_sell_threshold', 70)
        
        if rsi < rsi_buy:
            buy_score += 0.25
        elif rsi > rsi_sell:
            sell_score += 0.25
        total_weight += 0.25
        
        # MACD
        if indicators['macd'] > indicators['macd_signal'] and indicators['macd'] < 0:
            buy_score += 0.20
        elif indicators['macd'] < indicators['macd_signal'] and indicators['macd'] > 0:
            sell_score += 0.20
        total_weight += 0.20
        
        # Bollinger Bands
        if current_price < indicators['bb_lower']:
            buy_score += 0.15
        elif current_price > indicators['bb_upper']:
            sell_score += 0.15
        total_weight += 0.15
        
        # EMA Cross
        if indicators['ema_diff'] > 0.001:
            buy_score += 0.15
        elif indicators['ema_diff'] < -0.001:
            sell_score += 0.15
        total_weight += 0.15
        
        # Momentum
        if indicators['momentum'] > 0:
            buy_score += 0.10
        else:
            sell_score += 0.10
        total_weight += 0.10
        
        # Trend
        trend = indicators['trend_strength']
        if trend > 0.6:
            buy_score += 0.10
        elif trend < 0.4:
            sell_score += 0.10
        total_weight += 0.10
        
        # Volume
        if indicators['volume_ratio'] > 1.5 and indicators['price_change'] > 0:
            buy_score += 0.05
        elif indicators['volume_ratio'] > 1.5 and indicators['price_change'] < 0:
            sell_score += 0.05
        total_weight += 0.05
        
        # Calcular confiança final
        buy_confidence = buy_score / total_weight if total_weight > 0 else 0
        sell_confidence = sell_score / total_weight if total_weight > 0 else 0
        
        # Determinar sinal
        if buy_confidence > sell_confidence and buy_confidence > 0.5:
            return 'BUY', buy_confidence
        elif sell_confidence > buy_confidence and sell_confidence > 0.5:
            return 'SELL', sell_confidence
        else:
            return 'HOLD', max(buy_confidence, sell_confidence)
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Retorna análise vazia em caso de erro"""
        return {
            'signal': 'HOLD',
            'confidence': 0.0,
            'indicators': {},
            'price': 0.0
        }
