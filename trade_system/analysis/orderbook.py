"""
Análise paralela de orderbook para detecção de imbalances
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Tuple, Dict
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class ParallelOrderbookAnalyzer:
    """Análise de orderbook com processamento paralelo"""
    
    def __init__(self, config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.imbalance_history = deque(maxlen=100)
        self.pressure_history = deque(maxlen=50)
        self.large_orders = deque(maxlen=20)
    
    def analyze(self, bids: np.ndarray, asks: np.ndarray, cache) -> Tuple[str, float, Dict]:
        """
        Análise paralela do orderbook
        
        Args:
            bids: Array de bids [[price, volume], ...]
            asks: Array de asks [[price, volume], ...]
            cache: Cache para acesso rápido
            
        Returns:
            Tupla (action, confidence, details)
        """
        # Verificar cache primeiro
        cached_imbalance = cache.get('orderbook_imbalance')
        if cached_imbalance is not None:
            return self._quick_decision(cached_imbalance)
        
        # Validar dados
        if len(bids) == 0 or len(asks) == 0:
            return 'HOLD', 0.5, {'reason': 'Orderbook vazio'}
        
        # Cálculos paralelos
        future_bid = self.executor.submit(self._analyze_bid_side, bids)
        future_ask = self.executor.submit(self._analyze_ask_side, asks)
        
        bid_strength = future_bid.result()
        ask_strength = future_ask.result()
        
        # Calcular imbalance
        total = bid_strength + ask_strength
        if total > 0:
            imbalance = (bid_strength - ask_strength) / total
        else:
            imbalance = 0
        
        # Registrar histórico
        self.imbalance_history.append(imbalance)
        
        # Analisar microestrutura
        spread = asks[0, 0] - bids[0, 0] if asks[0, 0] > 0 and bids[0, 0] > 0 else 0
        spread_bps = (spread / bids[0, 0]) * 10000 if bids[0, 0] > 0 else 0
        
        # Profundidade do book
        bid_depth = np.sum(bids[:5, 1])
        ask_depth = np.sum(asks[:5, 1])
        depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 1
        
        # Detectar ordens grandes
        self._detect_large_orders(bids, asks)
        
        # Análise de pressão
        pressure = self._calculate_pressure(bids, asks)
        self.pressure_history.append(pressure)
        
        # Decisão baseada em múltiplos fatores
        action, confidence = self._make_decision(
            imbalance, spread_bps, depth_ratio, pressure
        )
        
        # Detalhes para log
        details = {
            'imbalance': float(imbalance),
            'spread_bps': float(spread_bps),
            'bid_strength': float(bid_strength),
            'ask_strength': float(ask_strength),
            'bid_depth': float(bid_depth),
            'ask_depth': float(ask_depth),
            'depth_ratio': float(depth_ratio),
            'best_bid': float(bids[0, 0]) if bids[0, 0] > 0 else 0,
            'best_ask': float(asks[0, 0]) if asks[0, 0] > 0 else 0,
            'pressure': float(pressure),
            'large_orders': len(self.large_orders)
        }
        
        # Cache resultado
        cache.set('orderbook_analysis', {
            'action': action,
            'confidence': confidence,
            'details': details
        }, ttl=2)
        
        return action, confidence, details
    
    def _analyze_bid_side(self, bids: np.ndarray) -> float:
        """Analisa força do lado comprador"""
        if len(bids) == 0:
            return 0.0
        
        # Peso por proximidade do preço
        weights = np.exp(-np.arange(len(bids)) * 0.1)
        weighted_volume = np.sum(bids[:, 1] * weights[:len(bids)])
        
        # Considerar concentração de ordens
        top_5_volume = np.sum(bids[:5, 1]) if len(bids) >= 5 else np.sum(bids[:, 1])
        total_volume = np.sum(bids[:, 1])
        concentration = top_5_volume / total_volume if total_volume > 0 else 0
        
        return weighted_volume * (1 + concentration * 0.2)
    
    def _analyze_ask_side(self, asks: np.ndarray) -> float:
        """Analisa força do lado vendedor"""
        if len(asks) == 0:
            return 0.0
        
        weights = np.exp(-np.arange(len(asks)) * 0.1)
        weighted_volume = np.sum(asks[:, 1] * weights[:len(asks)])
        
        top_5_volume = np.sum(asks[:5, 1]) if len(asks) >= 5 else np.sum(asks[:, 1])
        total_volume = np.sum(asks[:, 1])
        concentration = top_5_volume / total_volume if total_volume > 0 else 0
        
        return weighted_volume * (1 + concentration * 0.2)
    
    def _calculate_pressure(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """Calcula pressão de compra/venda"""
        if len(bids) < 10 or len(asks) < 10:
            return 0.0
        
        # Pressão nos primeiros 10 níveis
        bid_pressure = np.sum(bids[:10, 1] * np.exp(-np.arange(10) * 0.05))
        ask_pressure = np.sum(asks[:10, 1] * np.exp(-np.arange(10) * 0.05))
        
        total_pressure = bid_pressure + ask_pressure
        if total_pressure > 0:
            return (bid_pressure - ask_pressure) / total_pressure
        
        return 0.0
    
    def _detect_large_orders(self, bids: np.ndarray, asks: np.ndarray):
        """Detecta ordens grandes no book"""
        # Calcular tamanho médio
        all_volumes = np.concatenate([bids[:, 1], asks[:, 1]])
        if len(all_volumes) == 0:
            return
        
        avg_volume = np.mean(all_volumes)
        std_volume = np.std(all_volumes)
        threshold = avg_volume + 2 * std_volume
        
        # Detectar ordens grandes
        for i, (price, volume) in enumerate(bids):
            if volume > threshold:
                self.large_orders.append({
                    'side': 'BID',
                    'price': price,
                    'volume': volume,
                    'level': i,
                    'timestamp': time.time()
                })
        
        for i, (price, volume) in enumerate(asks):
            if volume > threshold:
                self.large_orders.append({
                    'side': 'ASK',
                    'price': price,
                    'volume': volume,
                    'level': i,
                    'timestamp': time.time()
                })
    
    def _make_decision(
        self,
        imbalance: float,
        spread_bps: float,
        depth_ratio: float,
        pressure: float
    ) -> Tuple[str, float]:
        """Toma decisão baseada em múltiplos fatores"""
        # Pontuação composta
        score = 0.0
        confidence_factors = []
        
        # Fator 1: Imbalance
        if abs(imbalance) > 0.6:
            score += imbalance * 0.4
            confidence_factors.append(min(abs(imbalance), 0.9))
        
        # Fator 2: Spread (menor é melhor)
        if spread_bps < 10:
            score += np.sign(imbalance) * 0.2
            confidence_factors.append(0.8)
        elif spread_bps > 20:
            score *= 0.5  # Reduz confiança com spread alto
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.7)
        
        # Fator 3: Depth ratio
        if depth_ratio > 1.5:
            score += 0.2
            confidence_factors.append(0.7)
        elif depth_ratio < 0.67:
            score -= 0.2
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Fator 4: Pressão
        score += pressure * 0.3
        if abs(pressure) > 0.3:
            confidence_factors.append(min(abs(pressure) + 0.5, 0.9))
        else:
            confidence_factors.append(0.5)
        
        # Decisão final
        if score > 0.4:
            action = 'BUY'
        elif score < -0.4:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Confiança média ponderada
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Ajustar confiança para HOLD
        if action == 'HOLD':
            confidence *= 0.7
        
        return action, confidence
    
    def _quick_decision(self, imbalance: float) -> Tuple[str, float, Dict]:
        """Decisão rápida baseada em cache"""
        if imbalance > 0.6:
            return 'BUY', 0.7 + imbalance * 0.3, {'cached': True, 'imbalance': imbalance}
        elif imbalance < -0.6:
            return 'SELL', 0.7 + abs(imbalance) * 0.3, {'cached': True, 'imbalance': imbalance}
        else:
            return 'HOLD', 0.5, {'cached': True, 'imbalance': imbalance}
    
    def get_market_pressure(self) -> str:
        """Retorna pressão geral do mercado"""
        if len(self.imbalance_history) < 10:
            return "Neutro"
        
        avg_imbalance = np.mean(list(self.imbalance_history)[-20:])
        
        if avg_imbalance > 0.3:
            return "Pressão Compradora"
        elif avg_imbalance < -0.3:
            return "Pressão Vendedora"
        else:
            return "Neutro"
    
    def get_orderbook_stats(self) -> Dict:
        """Retorna estatísticas do orderbook"""
        if not self.imbalance_history:
            return {}
        
        recent_imbalances = list(self.imbalance_history)[-50:]
        recent_pressures = list(self.pressure_history)[-50:] if self.pressure_history else []
        
        stats = {
            'avg_imbalance': np.mean(recent_imbalances),
            'std_imbalance': np.std(recent_imbalances),
            'trend': 'UP' if recent_imbalances[-1] > recent_imbalances[0] else 'DOWN',
            'large_orders_count': len(self.large_orders),
            'market_pressure': self.get_market_pressure()
        }
        
        if recent_pressures:
            stats['avg_pressure'] = np.mean(recent_pressures)
            stats['pressure_trend'] = 'INCREASING' if recent_pressures[-1] > recent_pressures[0] else 'DECREASING'
        
        return stats


# Importações necessárias
import time
