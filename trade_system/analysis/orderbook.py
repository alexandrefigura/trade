"""
Análise paralela de orderbook para detecção de imbalances
"""
import time
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
        self.bid_ask_ratio_history = deque(maxlen=50)
    
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
        cached_result = cache.get('orderbook_analysis')
        if cached_result and time.time() - cached_result.get('timestamp', 0) < 0.5:
            return cached_result['action'], cached_result['confidence'], cached_result['details']
        
        # Validar dados
        if len(bids) == 0 or len(asks) == 0:
            return 'HOLD', 0.5, {'reason': 'Orderbook vazio'}
        
        # Garantir que são arrays 2D
        if bids.ndim == 1:
            bids = bids.reshape(-1, 2)
        if asks.ndim == 1:
            asks = asks.reshape(-1, 2)
        
        # Cálculos paralelos
        future_bid = self.executor.submit(self._analyze_bid_side, bids)
        future_ask = self.executor.submit(self._analyze_ask_side, asks)
        
        bid_strength = future_bid.result()
        ask_strength = future_ask.result()
        
        # Calcular métricas principais
        imbalance = self._calculate_imbalance(bid_strength, ask_strength)
        bid_ask_ratio = self._calculate_bid_ask_ratio(bids, asks)
        spread_bps = self._calculate_spread(bids, asks)
        
        # Analisar profundidade
        depth_analysis = self._analyze_depth(bids, asks)
        
        # Detectar padrões
        wall_detection = self._detect_walls(bids, asks)
        momentum_signal = self._analyze_momentum()
        
        # Tomar decisão
        action, confidence = self._make_orderbook_decision(
            imbalance=imbalance,
            bid_ask_ratio=bid_ask_ratio,
            spread_bps=spread_bps,
            depth_analysis=depth_analysis,
            wall_detection=wall_detection,
            momentum_signal=momentum_signal
        )
        
        # Preparar detalhes
        details = {
            'imbalance': float(imbalance),
            'bid_ask_ratio': float(bid_ask_ratio),
            'spread_bps': float(spread_bps),
            'bid_strength': float(bid_strength),
            'ask_strength': float(ask_strength),
            'bid_depth': float(depth_analysis['bid_depth']),
            'ask_depth': float(depth_analysis['ask_depth']),
            'best_bid': float(bids[0, 0]) if bids[0, 0] > 0 else 0,
            'best_ask': float(asks[0, 0]) if asks[0, 0] > 0 else 0,
            'wall_detection': wall_detection,
            'momentum': momentum_signal
        }
        
        # Cache resultado
        cache.set('orderbook_analysis', {
            'action': action,
            'confidence': confidence,
            'details': details,
            'timestamp': time.time()
        }, ttl=1)
        
        # Log se significativo
        if confidence > 0.7 and action != 'HOLD':
            logger.info(f"📊 Orderbook sinal: {action} (conf: {confidence:.2%}) - Imbalance: {imbalance:.2f}")
        
        return action, confidence, details
    
    def _analyze_bid_side(self, bids: np.ndarray) -> float:
        """Analisa força do lado comprador"""
        if len(bids) == 0:
            return 0.0
        
        # Peso por proximidade do preço (exponencial decay)
        weights = np.exp(-np.arange(len(bids)) * 0.1)
        
        # Volume ponderado
        volumes = bids[:, 1]
        weighted_volume = np.sum(volumes * weights[:len(volumes)])
        
        # Considerar concentração
        if len(bids) >= 5:
            top_5_volume = np.sum(bids[:5, 1])
            total_volume = np.sum(bids[:, 1])
            concentration = top_5_volume / total_volume if total_volume > 0 else 0
            
            # Boost se houver concentração no topo
            if concentration > 0.7:
                weighted_volume *= 1.2
        
        return weighted_volume
    
    def _analyze_ask_side(self, asks: np.ndarray) -> float:
        """Analisa força do lado vendedor"""
        if len(asks) == 0:
            return 0.0
        
        weights = np.exp(-np.arange(len(asks)) * 0.1)
        volumes = asks[:, 1]
        weighted_volume = np.sum(volumes * weights[:len(volumes)])
        
        if len(asks) >= 5:
            top_5_volume = np.sum(asks[:5, 1])
            total_volume = np.sum(asks[:, 1])
            concentration = top_5_volume / total_volume if total_volume > 0 else 0
            
            if concentration > 0.7:
                weighted_volume *= 1.2
        
        return weighted_volume
    
    def _calculate_imbalance(self, bid_strength: float, ask_strength: float) -> float:
        """Calcula imbalance entre compradores e vendedores"""
        total = bid_strength + ask_strength
        if total == 0:
            return 0.0
        
        imbalance = (bid_strength - ask_strength) / total
        
        # Registrar histórico
        self.imbalance_history.append(imbalance)
        
        return imbalance
    
    def _calculate_bid_ask_ratio(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """Calcula ratio entre volume de bid e ask"""
        bid_volume = np.sum(bids[:, 1]) if len(bids) > 0 else 0
        ask_volume = np.sum(asks[:, 1]) if len(asks) > 0 else 0
        
        if ask_volume == 0:
            return 2.0  # Máximo
        
        ratio = bid_volume / ask_volume
        self.bid_ask_ratio_history.append(ratio)
        
        return min(ratio, 2.0)  # Cap em 2.0
    
    def _calculate_spread(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """Calcula spread em basis points"""
        if len(bids) == 0 or len(asks) == 0:
            return 100.0  # Alto spread default
        
        best_bid = bids[0, 0]
        best_ask = asks[0, 0]
        
        if best_bid <= 0 or best_ask <= 0:
            return 100.0
        
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000
        
        return spread_bps
    
    def _analyze_depth(self, bids: np.ndarray, asks: np.ndarray) -> Dict:
        """Analisa profundidade do orderbook"""
        levels = [5, 10, 20]
        depth_info = {}
        
        for level in levels:
            bid_depth = np.sum(bids[:level, 1]) if len(bids) >= level else np.sum(bids[:, 1])
            ask_depth = np.sum(asks[:level, 1]) if len(asks) >= level else np.sum(asks[:, 1])
            
            depth_info[f'bid_depth_{level}'] = bid_depth
            depth_info[f'ask_depth_{level}'] = ask_depth
            depth_info[f'depth_ratio_{level}'] = bid_depth / ask_depth if ask_depth > 0 else 1.0
        
        # Resumo
        depth_info['bid_depth'] = depth_info.get('bid_depth_10', 0)
        depth_info['ask_depth'] = depth_info.get('ask_depth_10', 0)
        depth_info['depth_ratio'] = depth_info.get('depth_ratio_10', 1.0)
        
        return depth_info
    
    def _detect_walls(self, bids: np.ndarray, asks: np.ndarray) -> Dict:
        """Detecta paredes de compra/venda"""
        wall_info = {
            'bid_wall': False,
            'ask_wall': False,
            'bid_wall_price': 0,
            'ask_wall_price': 0,
            'bid_wall_size': 0,
            'ask_wall_size': 0
        }
        
        # Calcular volume médio
        all_volumes = np.concatenate([bids[:, 1], asks[:, 1]])
        if len(all_volumes) == 0:
            return wall_info
        
        avg_volume = np.mean(all_volumes)
        wall_threshold = avg_volume * 3  # 3x média = parede
        
        # Detectar bid wall
        for i, (price, volume) in enumerate(bids):
            if volume > wall_threshold:
                wall_info['bid_wall'] = True
                wall_info['bid_wall_price'] = price
                wall_info['bid_wall_size'] = volume
                break
        
        # Detectar ask wall
        for i, (price, volume) in enumerate(asks):
            if volume > wall_threshold:
                wall_info['ask_wall'] = True
                wall_info['ask_wall_price'] = price
                wall_info['ask_wall_size'] = volume
                break
        
        return wall_info
    
    def _analyze_momentum(self) -> float:
        """Analisa momentum baseado no histórico de imbalance"""
        if len(self.imbalance_history) < 10:
            return 0.0
        
        recent = list(self.imbalance_history)[-10:]
        older = list(self.imbalance_history)[-20:-10] if len(self.imbalance_history) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        momentum = recent_avg - older_avg
        
        return momentum
    
    def _make_orderbook_decision(
        self,
        imbalance: float,
        bid_ask_ratio: float,
        spread_bps: float,
        depth_analysis: Dict,
        wall_detection: Dict,
        momentum_signal: float
    ) -> Tuple[str, float]:
        """Toma decisão baseada em análise do orderbook"""
        score = 0.0
        confidence_factors = []
        reasons = []
        
        # 1. Imbalance (peso: 35%)
        if abs(imbalance) > 0.3:
            score += imbalance * 0.35
            confidence_factors.append(min(abs(imbalance) * 1.5, 0.9))
            reasons.append(f"Imbalance: {imbalance:.2f}")
        
        # 2. Bid/Ask Ratio (peso: 25%)
        if bid_ask_ratio > 1.3:
            score += 0.25
            confidence_factors.append(0.7)
            reasons.append(f"High bid/ask ratio: {bid_ask_ratio:.2f}")
        elif bid_ask_ratio < 0.7:
            score -= 0.25
            confidence_factors.append(0.7)
            reasons.append(f"Low bid/ask ratio: {bid_ask_ratio:.2f}")
        
        # 3. Spread (peso: 15%)
        if spread_bps < 10:
            # Spread apertado é bom para trading
            confidence_factors.append(0.8)
        elif spread_bps > 30:
            # Spread largo, reduzir confiança
            score *= 0.7
            confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.6)
        
        # 4. Depth Ratio (peso: 15%)
        depth_ratio = depth_analysis.get('depth_ratio', 1.0)
        if depth_ratio > 1.2:
            score += 0.15
            confidence_factors.append(0.6)
        elif depth_ratio < 0.8:
            score -= 0.15
            confidence_factors.append(0.6)
        
        # 5. Wall Detection (peso: 10%)
        if wall_detection['bid_wall'] and not wall_detection['ask_wall']:
            score += 0.1
            confidence_factors.append(0.8)
            reasons.append("Bid wall detected")
        elif wall_detection['ask_wall'] and not wall_detection['bid_wall']:
            score -= 0.1
            confidence_factors.append(0.8)
            reasons.append("Ask wall detected")
        
        # 6. Momentum bonus
        if abs(momentum_signal) > 0.1:
            score += momentum_signal * 0.2
            confidence_factors.append(0.7)
        
        # Decisão final
        if score > 0.2:
            action = 'BUY'
        elif score < -0.2:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Confiança média
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Ajustar confiança para HOLD
        if action == 'HOLD':
            confidence *= 0.6
        
        # Debug log
        if self.config.debug_mode and reasons:
            logger.debug(f"Orderbook decision: {action} (score: {score:.2f}) - {', '.join(reasons)}")
        
        return action, confidence
    
    def get_market_pressure(self) -> str:
        """Retorna pressão geral do mercado"""
        if len(self.imbalance_history) < 10:
            return "Neutro"
        
        avg_imbalance = np.mean(list(self.imbalance_history)[-20:])
        
        if avg_imbalance > 0.2:
            return "Pressão Compradora"
        elif avg_imbalance < -0.2:
            return "Pressão Vendedora"
        else:
            return "Neutro"
    
    def get_orderbook_stats(self) -> Dict:
        """Retorna estatísticas do orderbook"""
        if not self.imbalance_history:
            return {}
        
        recent_imbalances = list(self.imbalance_history)[-50:]
        recent_ratios = list(self.bid_ask_ratio_history)[-50:] if self.bid_ask_ratio_history else []
        
        stats = {
            'avg_imbalance': np.mean(recent_imbalances),
            'std_imbalance': np.std(recent_imbalances),
            'trend': 'UP' if recent_imbalances[-1] > recent_imbalances[0] else 'DOWN',
            'large_orders_count': len(self.large_orders),
            'market_pressure': self.get_market_pressure()
        }
        
        if recent_ratios:
            stats['avg_bid_ask_ratio'] = np.mean(recent_ratios)
            stats['ratio_trend'] = 'INCREASING' if recent_ratios[-1] > recent_ratios[0] else 'DECREASING'
        
        return stats