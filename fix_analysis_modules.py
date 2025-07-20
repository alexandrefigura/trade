import os
import re

print("🔧 CORRIGINDO MÓDULOS DE ANÁLISE")
print("=" * 60)

# 1. Verificar classes em cada arquivo de análise
analysis_dir = 'trade_system/analysis'
analysis_files = ['technical.py', 'orderbook.py', 'ml.py']

print("\n🔍 Verificando classes existentes...")

found_classes = {}
for file in analysis_files:
    filepath = os.path.join(analysis_dir, file)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Encontrar classes
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            found_classes[file] = classes
            print(f"\n{file}:")
            for cls in classes:
                print(f"  - {cls}")
                
        except Exception as e:
            print(f"❌ Erro ao ler {file}: {e}")

# 2. Se orderbook.py não tem OrderbookAnalyzer, criar
if 'orderbook.py' not in found_classes or 'OrderbookAnalyzer' not in found_classes.get('orderbook.py', []):
    print("\n📝 Criando OrderbookAnalyzer...")
    
    orderbook_code = '''"""Análise do livro de ofertas (orderbook)"""
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

class OrderbookAnalyzer:
    """Analisa o livro de ofertas para detectar pressão de compra/venda"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, orderbook: Optional[Dict]) -> Dict[str, Any]:
        """
        Analisa orderbook e retorna métricas
        
        Args:
            orderbook: Dict com bids e asks
            
        Returns:
            Dict com análise do orderbook
        """
        if not orderbook:
            return self._empty_analysis()
        
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return self._empty_analysis()
            
            # Converter para arrays numpy
            bids = np.array(bids[:20], dtype=float)  # Top 20 bids
            asks = np.array(asks[:20], dtype=float)  # Top 20 asks
            
            # Calcular métricas
            metrics = {
                **self._calculate_spread(bids, asks),
                **self._calculate_pressure(bids, asks),
                **self._calculate_imbalance(bids, asks)
            }
            
            # Gerar sinal
            signal = self._generate_signal(metrics)
            metrics['signal'] = signal
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar orderbook: {e}")
            return self._empty_analysis()
    
    def _calculate_spread(self, bids: np.ndarray, asks: np.ndarray) -> Dict[str, float]:
        """Calcula spread"""
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100
        spread_bps = spread_pct * 100
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_pct': spread_pct,
            'spread_bps': spread_bps
        }
    
    def _calculate_pressure(self, bids: np.ndarray, asks: np.ndarray) -> Dict[str, float]:
        """Calcula pressão compradora/vendedora"""
        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        total_volume = bid_volume + ask_volume
        
        buy_pressure = bid_volume / total_volume if total_volume > 0 else 0.5
        
        return {
            'buy_pressure': buy_pressure,
            'sell_pressure': 1 - buy_pressure,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume
        }
    
    def _calculate_imbalance(self, bids: np.ndarray, asks: np.ndarray) -> Dict[str, float]:
        """Calcula desequilíbrio no topo do book"""
        top_bid_vol = bids[0][1]
        top_ask_vol = asks[0][1]
        
        total = top_bid_vol + top_ask_vol
        imbalance = (top_bid_vol - top_ask_vol) / total if total > 0 else 0
        
        return {
            'imbalance': imbalance,
            'top_bid_volume': top_bid_vol,
            'top_ask_volume': top_ask_vol
        }
    
    def _generate_signal(self, metrics: Dict[str, float]) -> str:
        """Gera sinal baseado nas métricas"""
        buy_pressure = metrics.get('buy_pressure', 0.5)
        imbalance = metrics.get('imbalance', 0)
        
        if buy_pressure > 0.65 and imbalance > 0.2:
            return 'STRONG_BUY'
        elif buy_pressure > 0.55:
            return 'BUY'
        elif buy_pressure < 0.35 and imbalance < -0.2:
            return 'STRONG_SELL'
        elif buy_pressure < 0.45:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Retorna análise vazia"""
        return {
            'buy_pressure': 0.5,
            'sell_pressure': 0.5,
            'spread_bps': 0,
            'imbalance': 0,
            'signal': 'NEUTRAL'
        }
'''
    
    orderbook_file = os.path.join(analysis_dir, 'orderbook.py')
    with open(orderbook_file, 'w', encoding='utf-8') as f:
        f.write(orderbook_code)
    
    print("✅ OrderbookAnalyzer criado!")

# 3. Verificar technical.py
if 'technical.py' not in found_classes or 'TechnicalAnalyzer' not in found_classes.get('technical.py', []):
    print("\n📝 Criando TechnicalAnalyzer...")
    
    technical_code = '''"""Análise técnica com indicadores"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

class TechnicalAnalyzer:
    """Análise técnica básica"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, candles: pd.DataFrame) -> Dict[str, Any]:
        """Analisa dados e retorna sinais"""
        try:
            if len(candles) < 20:
                return self._empty_analysis()
            
            # Calcular indicadores
            indicators = self._calculate_indicators(candles)
            
            # Gerar sinal
            signal, confidence = self._generate_signal(indicators)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'indicators': indicators
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise técnica: {e}")
            return self._empty_analysis()
    
    def _calculate_indicators(self, candles: pd.DataFrame) -> Dict[str, float]:
        """Calcula indicadores técnicos"""
        close = candles['close'].values
        
        # RSI simplificado
        rsi = self._calculate_rsi(close)
        
        # Médias móveis
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
        
        # Momentum
        momentum = (close[-1] - close[-10]) / close[-10] * 100 if len(close) >= 10 else 0
        
        return {
            'rsi': rsi,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'momentum': momentum,
            'price': close[-1],
            'volatility': np.std(close[-20:]) / np.mean(close[-20:])
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calcula RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _generate_signal(self, indicators: Dict[str, float]) -> Tuple[str, float]:
        """Gera sinal de trading"""
        rsi = indicators['rsi']
        momentum = indicators['momentum']
        
        buy_score = 0
        sell_score = 0
        
        # RSI
        if rsi < 30:
            buy_score += 0.5
        elif rsi > 70:
            sell_score += 0.5
        
        # Momentum
        if momentum > 0:
            buy_score += 0.3
        else:
            sell_score += 0.3
        
        # Determinar sinal
        if buy_score > sell_score and buy_score > 0.5:
            return 'BUY', buy_score
        elif sell_score > buy_score and sell_score > 0.5:
            return 'SELL', sell_score
        else:
            return 'HOLD', max(buy_score, sell_score)
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Retorna análise vazia"""
        return {
            'signal': 'HOLD',
            'confidence': 0.0,
            'indicators': {}
        }
'''
    
    technical_file = os.path.join(analysis_dir, 'technical.py')
    with open(technical_file, 'w', encoding='utf-8') as f:
        f.write(technical_code)
    
    print("✅ TechnicalAnalyzer criado!")

# 4. Atualizar __init__.py
print("\n📝 Atualizando __init__.py...")

init_content = '''"""Módulos de análise"""

from trade_system.analysis.technical import TechnicalAnalyzer
from trade_system.analysis.orderbook import OrderbookAnalyzer

__all__ = ['TechnicalAnalyzer', 'OrderbookAnalyzer']

# Tentar importar ML se disponível
try:
    from trade_system.analysis.ml import MLPredictor
    __all__.append('MLPredictor')
except ImportError:
    pass
'''

init_file = os.path.join(analysis_dir, '__init__.py')
with open(init_file, 'w', encoding='utf-8') as f:
    f.write(init_content)

print("✅ __init__.py atualizado!")

print("\n✅ Correção completa!")
print("\n🚀 Execute: python run_trading.py")
