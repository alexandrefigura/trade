"""
Sistema de consolidação de sinais otimizado
"""
import time
import numpy as np
from collections import deque
from typing import List, Tuple, Dict
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class OptimizedSignalConsolidator:
    """Consolidação ultra-rápida de sinais"""
    
    def __init__(self):
        self.weights = {
            'technical': 0.35,
            'orderbook': 0.35,
            'ml': 0.30
        }
        self.signal_history = deque(maxlen=100)
        self.performance_by_source = {
            'technical': {'correct': 0, 'total': 0},
            'orderbook': {'correct': 0, 'total': 0},
            'ml': {'correct': 0, 'total': 0}
        }
    
    def consolidate(self, signals: List[Tuple[str, str, float]]) -> Tuple[str, float]:
        """
        Consolida sinais com votação ponderada otimizada
        
        Args:
            signals: Lista de tuplas (source, action, confidence)
            
        Returns:
            Tupla (action, confidence)
        """
        if not signals:
            return 'HOLD', 0.0
        
        # Vetorizar cálculos
        actions = []
        confidences = []
        weights = []
        signal_details = {}
        
        for source, action, confidence in signals:
            # Converter ação para número
            action_value = 1 if action == 'BUY' else (-1 if action == 'SELL' else 0)
            actions.append(action_value)
            confidences.append(confidence)
            
            # Peso adaptativo baseado em performance
            base_weight = self.weights.get(source, 0.25)
            adaptive_weight = self._get_adaptive_weight(source, base_weight)
            weights.append(adaptive_weight)
            
            signal_details[source] = (action, confidence)
        
        actions = np.array(actions)
        confidences = np.array(confidences)
        weights = np.array(weights)
        
        # Score ponderado
        weighted_score = np.sum(actions * confidences * weights) / np.sum(weights)
        avg_confidence = np.average(confidences, weights=weights)
        
        # Decisão final com thresholds adaptativos
        buy_threshold = 0.3
        sell_threshold = -0.3
        
        if weighted_score > buy_threshold:
            final_action = 'BUY'
        elif weighted_score < sell_threshold:
            final_action = 'SELL'
        else:
            final_action = 'HOLD'
            avg_confidence *= 0.5  # Reduzir confiança em HOLD
        
        # Registrar no histórico
        self.signal_history.append({
            'timestamp': time.time(),
            'action': final_action,
            'confidence': avg_confidence,
            'weighted_score': weighted_score,
            'signals': signal_details,
            'weights_used': dict(zip([s[0] for s in signals], weights))
        })
        
        # Log para sinais fortes
        if avg_confidence > 0.8 and final_action != 'HOLD':
            logger.info(f"🎯 Sinal forte consolidado: {final_action} ({avg_confidence:.2%})")
        
        return final_action, avg_confidence
    
    def _get_adaptive_weight(self, source: str, base_weight: float) -> float:
        """Ajusta peso baseado em performance histórica"""
        perf = self.performance_by_source.get(source, {})
        total = perf.get('total', 0)
        
        if total < 10:  # Poucos dados, usar peso base
            return base_weight
        
        # Taxa de acerto
        accuracy = perf.get('correct', 0) / total
        
        # Ajustar peso: aumentar se acima de 60%, diminuir se abaixo de 40%
        if accuracy > 0.6:
            return base_weight * 1.2
        elif accuracy < 0.4:
            return base_weight * 0.8
        else:
            return base_weight
    
    def update_performance(self, source: str, was_correct: bool):
        """Atualiza métricas de performance por fonte"""
        if source in self.performance_by_source:
            self.performance_by_source[source]['total'] += 1
            if was_correct:
                self.performance_by_source[source]['correct'] += 1
    
    def get_signal_statistics(self) -> Dict:
        """Retorna estatísticas dos sinais recentes"""
        if not self.signal_history:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'avg_confidence': 0
            }
        
        recent_signals = list(self.signal_history)
        buy_count = sum(1 for s in recent_signals if s['action'] == 'BUY')
        sell_count = sum(1 for s in recent_signals if s['action'] == 'SELL')
        hold_count = sum(1 for s in recent_signals if s['action'] == 'HOLD')
        
        return {
            'total_signals': len(recent_signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'avg_confidence': np.mean([s['confidence'] for s in recent_signals]),
            'performance_by_source': dict(self.performance_by_source)
        }
