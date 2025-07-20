"""Análise do livro de ofertas (orderbook)"""
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

ParallelOrderbookAnalyzer = OrderbookAnalyzer
