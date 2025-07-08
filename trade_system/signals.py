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
    """Consolidação ultra-rápida de sinais com lógica balanceada"""
    
    def __init__(self):
        self.weights = {
            'technical': 0.40,
            'orderbook': 0.35,
            'ml': 0.25
        }
        self.signal_history = deque(maxlen=100)
        self.performance_by_source = {
            'technical': {'correct': 0, 'total': 0},
            'orderbook': {'correct': 0, 'total': 0},
            'ml': {'correct': 0, 'total': 0}
        }
        self.debug_mode = False
        self.last_action = 'HOLD'
        self.same_action_count = 0
    
    def set_debug_mode(self, enabled: bool):
        """Ativa/desativa modo debug"""
        self.debug_mode = enabled
        if enabled:
            logger.info("🔧 Signal Consolidator em modo DEBUG")
            # Ajustar pesos para debug
            self.weights = {
                'technical': 0.50,  # Mais peso para TA em debug
                'orderbook': 0.30,
                'ml': 0.20
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
        
        # Dicionários para contagem
        votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        confidence_sum = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        signal_details = {}
        
        # Processar cada sinal
        for source, action, confidence in signals:
            # Peso adaptativo
            base_weight = self.weights.get(source, 0.25)
            adaptive_weight = self._get_adaptive_weight(source, base_weight)
            
            # Adicionar voto ponderado
            weighted_vote = confidence * adaptive_weight
            votes[action] += weighted_vote
            confidence_sum[action] += confidence
            
            signal_details[source] = (action, confidence)
            
            # Debug log
            if self.debug_mode:
                logger.debug(f"Sinal {source}: {action} ({confidence:.2%}) peso: {adaptive_weight:.2f}")
        
        # Normalizar votos
        total_votes = sum(votes.values())
        if total_votes == 0:
            return 'HOLD', 0.0
        
        # Encontrar ação vencedora
        winning_action = max(votes.items(), key=lambda x: x[1])[0]
        winning_score = votes[winning_action] / total_votes
        
        # Calcular confiança média da ação vencedora
        if confidence_sum[winning_action] > 0:
            num_votes_for_winner = sum(1 for _, action, _ in signals if action == winning_action)
            avg_confidence = confidence_sum[winning_action] / num_votes_for_winner
        else:
            avg_confidence = 0.5
        
        # Aplicar thresholds para evitar excesso de um tipo de sinal
        final_action, final_confidence = self._apply_balance_logic(
            winning_action, winning_score, avg_confidence, votes
        )
        
        # Registrar no histórico
        self.signal_history.append({
            'timestamp': time.time(),
            'action': final_action,
            'confidence': final_confidence,
            'votes': dict(votes),
            'signals': signal_details,
            'debug_mode': self.debug_mode
        })
        
        # Log para sinais significativos
        if final_confidence > 0.7 and final_action != 'HOLD':
            logger.info(f"🎯 Sinal forte consolidado: {final_action} ({final_confidence:.2%})")
            logger.debug(f"   Votos: BUY={votes['BUY']:.2f}, SELL={votes['SELL']:.2f}, HOLD={votes['HOLD']:.2f}")
        
        return final_action, final_confidence
    
    def _apply_balance_logic(
        self, 
        action: str, 
        score: float, 
        confidence: float,
        votes: Dict[str, float]
    ) -> Tuple[str, float]:
        """Aplica lógica de balanceamento para evitar viés"""
        # Contar ações consecutivas iguais
        if action == self.last_action and action != 'HOLD':
            self.same_action_count += 1
        else:
            self.same_action_count = 0
        
        # Se muitas ações iguais consecutivas, aumentar threshold
        if self.same_action_count > 5:
            required_score = 0.6 + (self.same_action_count * 0.02)  # Aumenta 2% por sinal repetido
            if score < required_score:
                logger.debug(f"Bloqueando {action} repetido ({self.same_action_count}x). Score {score:.2f} < {required_score:.2f}")
                action = 'HOLD'
                confidence *= 0.5
        
        # Verificar se há equilíbrio entre BUY e SELL
        buy_score = votes.get('BUY', 0)
        sell_score = votes.get('SELL', 0)
        
        # Se scores muito próximos, preferir HOLD
        if abs(buy_score - sell_score) < 0.1:
            if action != 'HOLD':
                logger.debug(f"Scores muito próximos: BUY={buy_score:.2f}, SELL={sell_score:.2f} -> HOLD")
                action = 'HOLD'
                confidence *= 0.6
        
        # Threshold mínimo de confiança
        min_confidence_threshold = 0.4 if self.debug_mode else 0.5
        if confidence < min_confidence_threshold:
            action = 'HOLD'
            confidence = 0.5
        
        self.last_action = action
        
        return action, confidence
    
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
                'avg_confidence': 0,
                'balance_ratio': 0
            }
        
        recent_signals = list(self.signal_history)
        buy_count = sum(1 for s in recent_signals if s['action'] == 'BUY')
        sell_count = sum(1 for s in recent_signals if s['action'] == 'SELL')
        hold_count = sum(1 for s in recent_signals if s['action'] == 'HOLD')
        
        # Calcular ratio de balanço
        total_directional = buy_count + sell_count
        balance_ratio = 0
        if total_directional > 0:
            balance_ratio = min(buy_count, sell_count) / max(buy_count, sell_count)
        
        return {
            'total_signals': len(recent_signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'avg_confidence': np.mean([s['confidence'] for s in recent_signals]),
            'performance_by_source': dict(self.performance_by_source),
            'balance_ratio': balance_ratio,  # 1.0 = perfeito balanço, 0 = totalmente enviesado
            'consecutive_same': self.same_action_count
        }
    
    def reset_consecutive_counter(self):
        """Reseta contador de ações consecutivas"""
        self.same_action_count = 0
        logger.info("🔄 Contador de sinais consecutivos resetado")