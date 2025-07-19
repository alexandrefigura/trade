"""Agregador de sinais de múltiplas fontes"""
import logging
from typing import Dict, Any, List, Tuple
import numpy as np

class SignalAggregator:
    """Agrega e pondera sinais de diferentes análises"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pesos padrão para cada fonte
        self.weights = {
            'technical': 0.40,
            'ml': 0.35,
            'orderbook': 0.25
        }
        
        # Histórico de sinais
        self.signal_history = []
        
    def aggregate(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Agrega sinais de múltiplas fontes
        
        Args:
            signals: Dict com sinais de cada fonte
            
        Returns:
            Sinal agregado final
        """
        try:
            # Coletar sinais válidos
            valid_signals = self._collect_valid_signals(signals)
            
            if not valid_signals:
                return self._empty_signal()
            
            # Calcular pontuações
            buy_score, sell_score = self._calculate_scores(valid_signals)
            
            # Determinar sinal final
            signal, confidence = self._determine_final_signal(buy_score, sell_score)
            
            # Compilar resultado
            result = {
                'signal': signal,
                'confidence': confidence,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'sources': valid_signals,
                'indicators': self._merge_indicators(signals)
            }
            
            # Adicionar ao histórico
            self.signal_history.append(result)
            if len(self.signal_history) > 100:
                self.signal_history.pop(0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao agregar sinais: {e}")
            return self._empty_signal()
    
    def _collect_valid_signals(self, signals: Dict[str, Dict]) -> List[Tuple[str, str, float, float]]:
        """Coleta sinais válidos com suas fontes"""
        valid_signals = []
        
        for source, data in signals.items():
            if not isinstance(data, dict):
                continue
                
            signal = data.get('signal', 'HOLD')
            confidence = data.get('confidence', 0.0)
            
            # Validar sinal
            if signal in ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL'] and confidence > 0:
                weight = self.weights.get(source, 0.0)
                valid_signals.append((source, signal, confidence, weight))
        
        return valid_signals
    
    def _calculate_scores(self, valid_signals: List[Tuple]) -> Tuple[float, float]:
        """Calcula pontuações de compra e venda"""
        buy_score = 0.0
        sell_score = 0.0
        
        for source, signal, confidence, weight in valid_signals:
            weighted_score = confidence * weight
            
            if signal in ['BUY', 'STRONG_BUY']:
                multiplier = 1.5 if signal == 'STRONG_BUY' else 1.0
                buy_score += weighted_score * multiplier
            elif signal in ['SELL', 'STRONG_SELL']:
                multiplier = 1.5 if signal == 'STRONG_SELL' else 1.0
                sell_score += weighted_score * multiplier
        
        # Normalizar pontuações
        total_weight = sum(self.weights.values())
        buy_score = buy_score / total_weight
        sell_score = sell_score / total_weight
        
        return buy_score, sell_score
    
    def _determine_final_signal(self, buy_score: float, sell_score: float) -> Tuple[str, float]:
        """Determina sinal final baseado nas pontuações"""
        # Threshold mínimo
        min_threshold = 0.5
        
        if buy_score > sell_score and buy_score > min_threshold:
            confidence = buy_score
            signal = 'BUY'
        elif sell_score > buy_score and sell_score > min_threshold:
            confidence = sell_score
            signal = 'SELL'
        else:
            confidence = max(buy_score, sell_score)
            signal = 'HOLD'
        
        return signal, confidence
    
    def _merge_indicators(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Mescla indicadores de todas as fontes"""
        indicators = {}
        
        # Technical indicators
        if 'technical' in signals and 'indicators' in signals['technical']:
            indicators.update(signals['technical']['indicators'])
        
        # Orderbook metrics
        if 'orderbook' in signals:
            ob_data = signals['orderbook']
            for key in ['spread_bps', 'imbalance', 'buy_pressure', 'volatility']:
                if key in ob_data:
                    indicators[key] = ob_data[key]
        
        return indicators
    
    def _empty_signal(self) -> Dict[str, Any]:
        """Retorna sinal vazio"""
        return {
            'signal': 'HOLD',
            'confidence': 0.0,
            'buy_score': 0.0,
            'sell_score': 0.0,
            'sources': [],
            'indicators': {}
        }
    
    def get_signal_consistency(self) -> float:
        """Calcula consistência dos sinais recentes"""
        if len(self.signal_history) < 10:
            return 0.5
        
        recent_signals = [s['signal'] for s in self.signal_history[-10:]]
        
        # Contar sinais
        buy_count = sum(1 for s in recent_signals if s == 'BUY')
        sell_count = sum(1 for s in recent_signals if s == 'SELL')
        hold_count = sum(1 for s in recent_signals if s == 'HOLD')
        
        # Consistência é alta quando um tipo domina
        max_count = max(buy_count, sell_count, hold_count)
        consistency = max_count / len(recent_signals)
        
        return consistency
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """Atualiza pesos baseado em performance"""
        # TODO: Implementar ajuste adaptativo de pesos
        pass
