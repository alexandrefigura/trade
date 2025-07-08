"""
Sistema de consolidação de sinais otimizado com feature flags e pesos adaptativos
"""
import time
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SignalFeatureFlags:
    """Feature flags para controlar quais sinais estão ativos"""
    enable_technical: bool = True
    enable_orderbook: bool = True
    enable_ml: bool = True
    enable_volume: bool = True
    enable_volatility: bool = True
    enable_sentiment: bool = False  # Para futuras implementações
    enable_custom: bool = False
    
    # Flags de comportamento
    enable_adaptive_weights: bool = True
    enable_balance_logic: bool = True
    enable_performance_tracking: bool = True
    enable_signal_filtering: bool = True
    
    # Configurações adicionais
    min_signals_required: int = 2  # Mínimo de sinais para tomar decisão
    require_consensus: bool = False  # Exigir consenso entre sinais
    
    def to_dict(self) -> Dict:
        """Converte flags para dicionário"""
        return {
            'technical': self.enable_technical,
            'orderbook': self.enable_orderbook,
            'ml': self.enable_ml,
            'volume': self.enable_volume,
            'volatility': self.enable_volatility,
            'sentiment': self.enable_sentiment,
            'custom': self.enable_custom,
            'adaptive_weights': self.enable_adaptive_weights,
            'balance_logic': self.enable_balance_logic,
            'performance_tracking': self.enable_performance_tracking,
            'signal_filtering': self.enable_signal_filtering,
            'min_signals': self.min_signals_required,
            'consensus': self.require_consensus
        }


@dataclass
class PerformanceMetrics:
    """Métricas de performance por fonte de sinal"""
    correct: int = 0
    total: int = 0
    profit_sum: float = 0.0
    recent_correct: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_profit: deque = field(default_factory=lambda: deque(maxlen=20))
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def accuracy(self) -> float:
        """Taxa de acerto geral"""
        return self.correct / self.total if self.total > 0 else 0.5
    
    @property
    def recent_accuracy(self) -> float:
        """Taxa de acerto recente (últimas 20 trades)"""
        if not self.recent_correct:
            return 0.5
        return sum(self.recent_correct) / len(self.recent_correct)
    
    @property
    def avg_profit(self) -> float:
        """Lucro médio por trade"""
        return self.profit_sum / self.total if self.total > 0 else 0.0
    
    @property
    def recent_avg_profit(self) -> float:
        """Lucro médio recente"""
        if not self.recent_profit:
            return 0.0
        return sum(self.recent_profit) / len(self.recent_profit)
    
    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio simplificado das trades recentes"""
        if not self.recent_profit or len(self.recent_profit) < 5:
            return 0.0
        
        profits = list(self.recent_profit)
        avg_return = np.mean(profits)
        std_return = np.std(profits)
        
        if std_return == 0:
            return 0.0
        
        return avg_return / std_return * np.sqrt(252)  # Anualizado


class AdaptiveSignalConsolidator:
    """Sistema avançado de consolidação com feature flags e pesos adaptativos"""
    
    def __init__(self, config: Optional[Dict] = None):
        # Feature flags
        self.flags = SignalFeatureFlags()
        if config and 'feature_flags' in config:
            self._update_flags(config['feature_flags'])
        
        # Pesos iniciais
        self.base_weights = {
            'technical': config.get('signal_weights', {}).get('technical', 0.40),
            'orderbook': config.get('signal_weights', {}).get('orderbook', 0.40),
            'ml': config.get('signal_weights', {}).get('ml', 0.20),
            'volume': config.get('signal_weights', {}).get('volume', 0.10),
            'volatility': config.get('signal_weights', {}).get('volatility', 0.10),
            'sentiment': config.get('signal_weights', {}).get('sentiment', 0.05),
            'custom': config.get('signal_weights', {}).get('custom', 0.05)
        }
        
        # Pesos adaptativos (começam iguais aos base)
        self.adaptive_weights = self.base_weights.copy()
        
        # Performance tracking
        self.performance = {
            source: PerformanceMetrics() for source in self.base_weights.keys()
        }
        
        # Histórico
        self.signal_history = deque(maxlen=1000)
        self.weight_history = deque(maxlen=100)
        self.trade_outcomes = deque(maxlen=100)
        
        # Estado
        self.last_action = 'HOLD'
        self.same_action_count = 0
        self.last_weight_update = datetime.now()
        self.weight_update_interval = timedelta(minutes=config.get('weight_update_minutes', 30))
        
        # Configurações de adaptação
        self.adaptation_config = {
            'learning_rate': config.get('weight_learning_rate', 0.1),
            'min_trades_for_adaptation': config.get('min_trades_adaptation', 10),
            'performance_window': config.get('performance_window', 20),
            'weight_bounds': config.get('weight_bounds', (0.05, 0.50)),
            'use_sharpe': config.get('use_sharpe_for_weights', True),
            'decay_factor': config.get('weight_decay', 0.95)
        }
        
        # Debug
        self.debug_mode = config.get('debug_mode', False)
        
        logger.info("🚀 Signal Consolidator inicializado com feature flags e pesos adaptativos")
        logger.info(f"📊 Feature flags: {self.flags.to_dict()}")
        logger.info(f"⚖️ Pesos iniciais: {self.adaptive_weights}")
    
    def _update_flags(self, flag_config: Dict):
        """Atualiza feature flags com configuração"""
        for key, value in flag_config.items():
            if hasattr(self.flags, f'enable_{key}'):
                setattr(self.flags, f'enable_{key}', value)
            elif hasattr(self.flags, key):
                setattr(self.flags, key, value)
    
    def update_feature_flags(self, flags: Dict):
        """Atualiza feature flags em runtime"""
        self._update_flags(flags)
        logger.info(f"🔄 Feature flags atualizadas: {self.flags.to_dict()}")
        
        # Recalcular pesos se alguma fonte foi desabilitada
        self._normalize_weights()
    
    def consolidate(self, signals: List[Tuple[str, str, float]]) -> Tuple[str, float]:
        """
        Consolida sinais com feature flags e pesos adaptativos
        
        Args:
            signals: Lista de tuplas (source, action, confidence)
            
        Returns:
            Tupla (action, confidence)
        """
        # Filtrar sinais baseado em feature flags
        filtered_signals = self._filter_signals_by_flags(signals)
        
        if not filtered_signals:
            return 'HOLD', 0.0
        
        # Verificar requisitos mínimos
        if len(filtered_signals) < self.flags.min_signals_required:
            if self.debug_mode:
                logger.debug(f"Sinais insuficientes: {len(filtered_signals)} < {self.flags.min_signals_required}")
            return 'HOLD', 0.0
        
        # Atualizar pesos se necessário
        if self.flags.enable_adaptive_weights:
            self._update_adaptive_weights()
        
        # Calcular votos ponderados
        votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        confidence_sum = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        signal_details = {}
        source_votes = {}  # Para verificar consenso
        
        for source, action, confidence in filtered_signals:
            # Obter peso adaptativo
            weight = self.adaptive_weights.get(source, 0.1)
            
            # Aplicar filtros de qualidade se habilitado
            if self.flags.enable_signal_filtering:
                confidence = self._filter_signal_quality(source, action, confidence)
            
            # Adicionar voto ponderado
            weighted_vote = confidence * weight
            votes[action] += weighted_vote
            confidence_sum[action] += confidence
            
            signal_details[source] = (action, confidence, weight)
            source_votes[source] = action
            
            if self.debug_mode:
                logger.debug(f"📊 {source}: {action} ({confidence:.2%}) peso: {weight:.3f}")
        
        # Verificar consenso se exigido
        if self.flags.require_consensus:
            if not self._check_consensus(source_votes):
                return 'HOLD', 0.3  # Baixa confiança quando não há consenso
        
        # Encontrar ação vencedora
        total_votes = sum(votes.values())
        if total_votes == 0:
            return 'HOLD', 0.0
        
        winning_action = max(votes.items(), key=lambda x: x[1])[0]
        winning_score = votes[winning_action] / total_votes
        
        # Calcular confiança
        num_votes_for_winner = sum(1 for _, action, _ in filtered_signals if action == winning_action)
        if num_votes_for_winner > 0 and confidence_sum[winning_action] > 0:
            avg_confidence = confidence_sum[winning_action] / num_votes_for_winner
        else:
            avg_confidence = 0.5
        
        # Aplicar lógica de balanço se habilitada
        if self.flags.enable_balance_logic:
            winning_action, avg_confidence = self._apply_balance_logic(
                winning_action, winning_score, avg_confidence, votes
            )
        
        # Registrar no histórico
        self._record_signal({
            'timestamp': time.time(),
            'action': winning_action,
            'confidence': avg_confidence,
            'votes': dict(votes),
            'signals': signal_details,
            'weights': self.adaptive_weights.copy(),
            'filtered_count': len(filtered_signals),
            'original_count': len(signals)
        })
        
        # Log para sinais importantes
        if avg_confidence > 0.7 and winning_action != 'HOLD':
            logger.info(f"🎯 Sinal consolidado: {winning_action} ({avg_confidence:.2%})")
            if self.debug_mode:
                logger.debug(f"   Votos: {votes}")
                logger.debug(f"   Pesos: {self.adaptive_weights}")
        
        return winning_action, avg_confidence
    
    def _filter_signals_by_flags(self, signals: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """Filtra sinais baseado em feature flags"""
        filtered = []
        
        for source, action, confidence in signals:
            # Verificar se a fonte está habilitada
            flag_name = f'enable_{source}'
            if hasattr(self.flags, flag_name):
                if getattr(self.flags, flag_name):
                    filtered.append((source, action, confidence))
                elif self.debug_mode:
                    logger.debug(f"❌ Sinal {source} filtrado (desabilitado)")
            else:
                # Fonte não reconhecida, incluir se custom habilitado
                if self.flags.enable_custom:
                    filtered.append((source, action, confidence))
        
        return filtered
    
    def _filter_signal_quality(self, source: str, action: str, confidence: float) -> float:
        """Filtra qualidade do sinal baseado em performance histórica"""
        perf = self.performance.get(source)
        if not perf or perf.total < 5:
            return confidence
        
        # Reduzir confiança se fonte tem performance ruim recente
        if perf.recent_accuracy < 0.4:
            confidence *= 0.8
            if self.debug_mode:
                logger.debug(f"📉 Confiança reduzida para {source}: accuracy {perf.recent_accuracy:.2%}")
        
        # Boost se performance excelente
        elif perf.recent_accuracy > 0.7:
            confidence = min(0.95, confidence * 1.1)
            if self.debug_mode:
                logger.debug(f"📈 Confiança aumentada para {source}: accuracy {perf.recent_accuracy:.2%}")
        
        return confidence
    
    def _check_consensus(self, source_votes: Dict[str, str]) -> bool:
        """Verifica se há consenso entre as fontes principais"""
        # Fontes principais para consenso
        main_sources = ['technical', 'orderbook', 'ml']
        main_votes = [source_votes.get(s) for s in main_sources if s in source_votes]
        
        if len(main_votes) < 2:
            return True  # Não há fontes suficientes para exigir consenso
        
        # Verificar se pelo menos 2/3 concordam
        from collections import Counter
        vote_counts = Counter(main_votes)
        most_common = vote_counts.most_common(1)[0]
        
        return most_common[1] >= len(main_votes) * 0.66
    
    def _update_adaptive_weights(self):
        """Atualiza pesos adaptativos baseado em performance recente"""
        # Verificar se é hora de atualizar
        if datetime.now() - self.last_weight_update < self.weight_update_interval:
            return
        
        # Verificar se há dados suficientes
        min_trades = self.adaptation_config['min_trades_for_adaptation']
        if all(perf.total < min_trades for perf in self.performance.values()):
            return
        
        logger.info("🔄 Atualizando pesos adaptativos...")
        
        # Calcular scores de performance para cada fonte
        performance_scores = {}
        
        for source, perf in self.performance.items():
            if perf.total < 5:  # Muito poucos dados
                performance_scores[source] = 0.5  # Score neutro
                continue
            
            # Combinar múltiplas métricas
            accuracy_score = perf.recent_accuracy
            
            if self.adaptation_config['use_sharpe']:
                # Usar Sharpe ratio se disponível
                sharpe = perf.sharpe_ratio
                # Normalizar Sharpe para [0, 1]
                normalized_sharpe = (sharpe + 2) / 4  # Assume Sharpe entre -2 e 2
                normalized_sharpe = np.clip(normalized_sharpe, 0, 1)
                
                # Combinar accuracy e Sharpe
                performance_scores[source] = 0.6 * accuracy_score + 0.4 * normalized_sharpe
            else:
                # Usar apenas accuracy e lucro médio
                profit_score = (perf.recent_avg_profit + 0.05) / 0.1  # Normalizar lucro
                profit_score = np.clip(profit_score, 0, 1)
                
                performance_scores[source] = 0.7 * accuracy_score + 0.3 * profit_score
        
        # Atualizar pesos usando learning rate
        learning_rate = self.adaptation_config['learning_rate']
        bounds = self.adaptation_config['weight_bounds']
        decay = self.adaptation_config['decay_factor']
        
        new_weights = {}
        for source in self.adaptive_weights:
            if source not in performance_scores:
                new_weights[source] = self.adaptive_weights[source] * decay
                continue
            
            # Calcular novo peso
            current_weight = self.adaptive_weights[source]
            performance = performance_scores[source]
            
            # Ajuste proporcional à performance
            if performance > 0.6:  # Boa performance
                adjustment = learning_rate * (performance - 0.5)
                new_weight = current_weight * (1 + adjustment)
            elif performance < 0.4:  # Performance ruim
                adjustment = learning_rate * (0.5 - performance)
                new_weight = current_weight * (1 - adjustment)
            else:  # Performance neutra
                new_weight = current_weight * decay  # Pequeno decay
            
            # Aplicar limites
            new_weights[source] = np.clip(new_weight, bounds[0], bounds[1])
        
        # Normalizar pesos para somar 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for source in new_weights:
                new_weights[source] /= total_weight
        
        # Registrar mudanças significativas
        for source, new_weight in new_weights.items():
            old_weight = self.adaptive_weights[source]
            if abs(new_weight - old_weight) > 0.05:
                logger.info(f"  {source}: {old_weight:.3f} → {new_weight:.3f} "
                          f"(perf: {performance_scores.get(source, 0):.2%})")
        
        # Atualizar pesos
        self.adaptive_weights = new_weights
        self.last_weight_update = datetime.now()
        
        # Registrar no histórico
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': new_weights.copy(),
            'performance_scores': performance_scores.copy()
        })
    
    def _normalize_weights(self):
        """Normaliza pesos para somar 1.0, considerando feature flags"""
        # Filtrar pesos de fontes desabilitadas
        active_weights = {}
        
        for source, weight in self.adaptive_weights.items():
            flag_name = f'enable_{source}'
            if hasattr(self.flags, flag_name) and getattr(self.flags, flag_name):
                active_weights[source] = weight
        
        # Normalizar
        total = sum(active_weights.values())
        if total > 0:
            for source in active_weights:
                self.adaptive_weights[source] = active_weights[source] / total
        
        # Zerar pesos de fontes desabilitadas
        for source in self.adaptive_weights:
            if source not in active_weights:
                self.adaptive_weights[source] = 0.0
    
    def _apply_balance_logic(self, action: str, score: float, 
                           confidence: float, votes: Dict[str, float]) -> Tuple[str, float]:
        """Aplica lógica de balanceamento"""
        # Contar ações consecutivas
        if action == self.last_action and action != 'HOLD':
            self.same_action_count += 1
        else:
            self.same_action_count = 0
        
        # Penalizar ações repetidas
        if self.same_action_count > 5:
            penalty = 0.05 * (self.same_action_count - 5)
            confidence *= (1 - penalty)
            
            if self.debug_mode:
                logger.debug(f"⚠️ {action} repetido {self.same_action_count}x, "
                           f"confiança reduzida para {confidence:.2%}")
        
        # Verificar equilíbrio entre BUY e SELL
        buy_score = votes.get('BUY', 0)
        sell_score = votes.get('SELL', 0)
        
        if abs(buy_score - sell_score) < 0.1 and action != 'HOLD':
            confidence *= 0.7
            if confidence < 0.5:
                action = 'HOLD'
        
        self.last_action = action
        return action, confidence
    
    def _record_signal(self, signal_data: Dict):
        """Registra sinal no histórico"""
        self.signal_history.append(signal_data)
        
        # Limpar cache antigo se necessário
        if len(self.signal_history) > 900:
            # Manter apenas últimos 900 registros
            self.signal_history = deque(list(self.signal_history)[-900:], maxlen=1000)
    
    def record_trade_outcome(self, source_signals: Dict[str, Tuple[str, float]], 
                           was_correct: bool, profit: float = 0.0):
        """
        Registra resultado de uma trade para atualizar performance
        
        Args:
            source_signals: Dict com fonte -> (action, confidence)
            was_correct: Se a trade foi bem-sucedida
            profit: Lucro/prejuízo da trade
        """
        if not self.flags.enable_performance_tracking:
            return
        
        # Atualizar performance de cada fonte
        for source, (action, confidence) in source_signals.items():
            if source in self.performance:
                perf = self.performance[source]
                
                # Atualizar contadores
                perf.total += 1
                if was_correct:
                    perf.correct += 1
                perf.profit_sum += profit
                
                # Atualizar deques recentes
                perf.recent_correct.append(1 if was_correct else 0)
                perf.recent_profit.append(profit)
                perf.last_update = datetime.now()
        
        # Registrar outcome
        self.trade_outcomes.append({
            'timestamp': datetime.now(),
            'sources': source_signals,
            'correct': was_correct,
            'profit': profit
        })
        
        if self.debug_mode:
            logger.debug(f"📊 Trade outcome registrado: {'✓' if was_correct else '✗'} "
                       f"Lucro: {profit:.2%}")
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas detalhadas do sistema"""
        recent_signals = list(self.signal_history)[-100:] if self.signal_history else []
        
        # Contagem de ações
        action_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_by_action = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        for signal in recent_signals:
            action = signal['action']
            confidence = signal['confidence']
            action_counts[action] += 1
            confidence_by_action[action].append(confidence)
        
        # Calcular métricas
        total_signals = sum(action_counts.values())
        balance_ratio = 0
        if action_counts['BUY'] + action_counts['SELL'] > 0:
            balance_ratio = min(action_counts['BUY'], action_counts['SELL']) / \
                          max(action_counts['BUY'], action_counts['SELL'])
        
        # Performance por fonte
        source_stats = {}
        for source, perf in self.performance.items():
            source_stats[source] = {
                'accuracy': perf.accuracy,
                'recent_accuracy': perf.recent_accuracy,
                'total_trades': perf.total,
                'avg_profit': perf.avg_profit,
                'recent_avg_profit': perf.recent_avg_profit,
                'sharpe_ratio': perf.sharpe_ratio,
                'weight': self.adaptive_weights.get(source, 0),
                'enabled': getattr(self.flags, f'enable_{source}', False)
            }
        
        return {
            'signal_counts': action_counts,
            'total_signals': total_signals,
            'avg_confidence': {
                action: np.mean(confs) if confs else 0 
                for action, confs in confidence_by_action.items()
            },
            'balance_ratio': balance_ratio,
            'consecutive_same': self.same_action_count,
            'source_performance': source_stats,
            'current_weights': dict(self.adaptive_weights),
            'feature_flags': self.flags.to_dict(),
            'last_weight_update': self.last_weight_update.isoformat(),
            'weight_history_size': len(self.weight_history),
            'trade_outcomes_size': len(self.trade_outcomes)
        }
    
    def reset_performance_metrics(self):
        """Reseta métricas de performance"""
        logger.warning("🔄 Resetando métricas de performance")
        
        for source in self.performance:
            self.performance[source] = PerformanceMetrics()
        
        # Resetar pesos para valores base
        self.adaptive_weights = self.base_weights.copy()
        
        # Limpar históricos
        self.trade_outcomes.clear()
        self.weight_history.clear()
        
        logger.info("✅ Métricas resetadas")
    
    def export_config(self) -> Dict:
        """Exporta configuração atual do sistema"""
        return {
            'feature_flags': self.flags.to_dict(),
            'base_weights': self.base_weights,
            'adaptive_weights': self.adaptive_weights,
            'adaptation_config': self.adaptation_config,
            'performance_summary': {
                source: {
                    'accuracy': perf.accuracy,
                    'recent_accuracy': perf.recent_accuracy,
                    'sharpe': perf.sharpe_ratio
                }
                for source, perf in self.performance.items()
            }
        }
    
    def import_config(self, config: Dict):
        """Importa configuração do sistema"""
        if 'feature_flags' in config:
            self._update_flags(config['feature_flags'])
        
        if 'adaptive_weights' in config:
            self.adaptive_weights = config['adaptive_weights'].copy()
            self._normalize_weights()
        
        if 'adaptation_config' in config:
            self.adaptation_config.update(config['adaptation_config'])
        
        logger.info("✅ Configuração importada com sucesso")


# Manter compatibilidade com código existente
class OptimizedSignalConsolidator(AdaptiveSignalConsolidator):
    """Alias para manter compatibilidade"""
    pass