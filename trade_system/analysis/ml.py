"""
Machine Learning simplificado para predições rápidas
"""
import numpy as np
import random
from collections import deque
from typing import Dict, Tuple, List
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class SimplifiedMLPredictor:
    """ML simplificado para predições rápidas com lógica balanceada"""
    
    def __init__(self):
        # Pesos do modelo
        self.feature_weights = np.array([
            0.25,   # RSI
            0.20,   # Momentum
            0.20,   # Volume ratio
            0.20,   # Spread
            0.15    # Volatility
        ], dtype=np.float32)
        
        # Thresholds adaptativos
        self.threshold_buy = 0.3
        self.threshold_sell = -0.3
        
        # Histórico
        self.prediction_history = deque(maxlen=100)
        self.feature_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        
        # Parâmetros adaptativos
        self.learning_rate = 0.01
        self.adaptation_enabled = True
        
        # Estatísticas
        self.total_predictions = 0
        self.correct_predictions = 0
        self.prediction_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Anti-bias
        self.last_predictions = deque(maxlen=10)
        
        logger.info("🤖 ML Predictor inicializado")
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        """
        Predição ultra-rápida com lógica anti-bias
        
        Args:
            features: Dicionário com features do mercado
            
        Returns:
            Tupla (action, confidence)
        """
        # Validar features
        required_features = ['rsi', 'momentum', 'volume_ratio', 'spread_bps', 'volatility']
        for feat in required_features:
            if feat not in features:
                logger.warning(f"Feature ausente: {feat}")
                return 'HOLD', 0.5
        
        # Normalizar features
        feature_vector = self._normalize_features(features)
        
        # Score base
        base_score = np.dot(feature_vector, self.feature_weights)
        
        # Aplicar fatores não-lineares
        adjusted_score = self._apply_nonlinear_factors(base_score, features)
        
        # Aplicar lógica anti-bias
        final_score = self._apply_antibias_logic(adjusted_score)
        
        # Decisão com thresholds adaptativos
        action, confidence = self._make_balanced_decision(final_score)
        
        # Registrar
        self.total_predictions += 1
        self.prediction_counts[action] += 1
        self.last_predictions.append(action)
        self._record_prediction(features, final_score, action, confidence)
        
        # Adaptar se habilitado
        if self.adaptation_enabled and len(self.performance_history) > 20:
            self._adapt_weights()
        
        return action, confidence
    
    def _normalize_features(self, features: Dict) -> np.ndarray:
        """Normaliza features para o modelo"""
        # RSI: converter para -1 a 1
        rsi = features.get('rsi', 50)
        rsi_norm = (rsi - 50) / 50
        
        # Momentum: já está normalizado
        momentum = features.get('momentum', 0)
        momentum = np.clip(momentum, -0.1, 0.1)
        
        # Volume ratio: normalizar em torno de 1
        volume_ratio = features.get('volume_ratio', 1)
        volume_norm = np.log(volume_ratio) if volume_ratio > 0 else 0
        volume_norm = np.clip(volume_norm, -1, 1)
        
        # Spread: negativo é bom
        spread_bps = features.get('spread_bps', 10)
        spread_norm = -np.log(spread_bps + 1) / 10
        spread_norm = np.clip(spread_norm, -1, 0)
        
        # Volatilidade: normalizar
        volatility = features.get('volatility', 0.01)
        vol_norm = -np.log(volatility * 100 + 1) / 5
        vol_norm = np.clip(vol_norm, -1, 0)
        
        return np.array([
            rsi_norm,
            momentum * 10,  # Amplificar momentum
            volume_norm,
            spread_norm,
            vol_norm
        ], dtype=np.float32)
    
    def _apply_nonlinear_factors(self, score: float, features: Dict) -> float:
        """Aplica fatores não-lineares ao score"""
        # RSI extremos
        rsi = features.get('rsi', 50)
        if rsi < 25:
            score += 0.2
        elif rsi > 75:
            score -= 0.2
        elif 45 < rsi < 55:
            # RSI neutro, dar mais peso a outros fatores
            momentum = features.get('momentum', 0)
            score += momentum * 5
        
        # Volume spike
        volume_ratio = features.get('volume_ratio', 1)
        if volume_ratio > 2:
            score *= 1.2
        elif volume_ratio < 0.5:
            score *= 0.8
        
        # Volatilidade
        volatility = features.get('volatility', 0.01)
        if volatility > 0.03:
            score *= 0.7  # Reduzir em alta volatilidade
        elif volatility < 0.005:
            score *= 1.1  # Boost em baixa volatilidade
        
        # Spread penalty
        spread_bps = features.get('spread_bps', 10)
        if spread_bps > 20:
            score *= 0.8
        
        return np.clip(score, -2, 2)
    
    def _apply_antibias_logic(self, score: float) -> float:
        """Aplica lógica para evitar viés em uma direção"""
        if len(self.last_predictions) < 5:
            return score
        
        # Contar últimas predições
        recent = list(self.last_predictions)
        buy_count = recent.count('BUY')
        sell_count = recent.count('SELL')
        
        # Se muito enviesado, ajustar score
        if buy_count > 7:  # Mais de 70% BUY
            score -= 0.3  # Penalizar BUY
            logger.debug(f"Anti-bias: Muitos BUYs recentes ({buy_count}/10)")
        elif sell_count > 7:  # Mais de 70% SELL
            score += 0.3  # Penalizar SELL
            logger.debug(f"Anti-bias: Muitos SELLs recentes ({sell_count}/10)")
        
        # Verificar proporção histórica
        total_preds = sum(self.prediction_counts.values())
        if total_preds > 100:
            buy_ratio = self.prediction_counts['BUY'] / total_preds
            sell_ratio = self.prediction_counts['SELL'] / total_preds
            
            # Ajustar thresholds se muito desbalanceado
            if buy_ratio > 0.6:
                self.threshold_buy = min(0.5, self.threshold_buy + 0.05)
            elif sell_ratio > 0.6:
                self.threshold_sell = max(-0.5, self.threshold_sell - 0.05)
        
        return score
    
    def _make_balanced_decision(self, score: float) -> Tuple[str, float]:
        """Toma decisão balanceada baseada no score"""
        # Aplicar função sigmoide para confiança
        confidence = 1 / (1 + np.exp(-abs(score)))
        
        # Adicionar ruído pequeno para evitar empates
        noise = random.uniform(-0.05, 0.05)
        score_with_noise = score + noise
        
        # Decisão com thresholds dinâmicos
        if score_with_noise > self.threshold_buy:
            action = 'BUY'
            # Boost de confiança para scores muito altos
            if score > self.threshold_buy * 1.5:
                confidence = min(0.9, confidence * 1.1)
        elif score_with_noise < self.threshold_sell:
            action = 'SELL'
            if score < self.threshold_sell * 1.5:
                confidence = min(0.9, confidence * 1.1)
        else:
            action = 'HOLD'
            confidence *= 0.6  # Reduzir confiança em HOLD
        
        # Verificar se precisa forçar balanço
        if self._should_force_balance():
            needed_action = self._get_needed_action()
            if needed_action != action and confidence < 0.7:
                action = needed_action
                confidence *= 0.8
                logger.debug(f"ML forçando {action} para balancear")
        
        return action, float(confidence)
    
    def _should_force_balance(self) -> bool:
        """Verifica se deve forçar ação para balancear"""
        total = sum(self.prediction_counts.values())
        if total < 50:
            return False
        
        buy_ratio = self.prediction_counts['BUY'] / total
        sell_ratio = self.prediction_counts['SELL'] / total
        
        # Se muito desbalanceado
        return abs(buy_ratio - sell_ratio) > 0.3
    
    def _get_needed_action(self) -> str:
        """Retorna ação necessária para balancear"""
        if self.prediction_counts['BUY'] > self.prediction_counts['SELL']:
            return 'SELL'
        else:
            return 'BUY'
    
    def _record_prediction(self, features: Dict, score: float, action: str, confidence: float):
        """Registra predição no histórico"""
        record = {
            'timestamp': time.time(),
            'features': features.copy(),
            'score': score,
            'action': action,
            'confidence': confidence,
            'weights': self.feature_weights.copy(),
            'thresholds': (self.threshold_buy, self.threshold_sell)
        }
        
        self.prediction_history.append(record)
        self.feature_history.append(features)
    
    def update_performance(self, prediction_id: int, was_correct: bool, profit: float = 0):
        """Atualiza performance de uma predição"""
        self.performance_history.append({
            'prediction_id': prediction_id,
            'correct': was_correct,
            'profit': profit,
            'timestamp': time.time()
        })
        
        if was_correct:
            self.correct_predictions += 1
    
    def _adapt_weights(self):
        """Adapta pesos baseado em performance recente"""
        if not self.performance_history:
            return
        
        # Calcular taxa de acerto recente
        recent_perf = list(self.performance_history)[-20:]
        accuracy = sum(p['correct'] for p in recent_perf) / len(recent_perf)
        
        # Só adaptar se performance ruim
        if accuracy < 0.45:
            # Pequenos ajustes aleatórios
            noise = np.random.randn(len(self.feature_weights)) * self.learning_rate
            self.feature_weights += noise
            
            # Normalizar pesos
            self.feature_weights = np.abs(self.feature_weights)
            self.feature_weights /= np.sum(self.feature_weights)
            
            logger.info(f"🔧 Pesos ML adaptados (accuracy: {accuracy:.2%})")
    
    def get_feature_importance(self) -> Dict:
        """Retorna importância relativa das features"""
        total_weight = np.sum(self.feature_weights)
        return {
            'rsi': float(self.feature_weights[0] / total_weight),
            'momentum': float(self.feature_weights[1] / total_weight),
            'volume_ratio': float(self.feature_weights[2] / total_weight),
            'spread': float(self.feature_weights[3] / total_weight),
            'volatility': float(self.feature_weights[4] / total_weight)
        }
    
    def get_prediction_stats(self) -> Dict:
        """Retorna estatísticas de predição"""
        total = self.total_predictions if self.total_predictions > 0 else 1
        
        recent_predictions = list(self.prediction_history)[-50:]
        
        # Confiança média
        confidences = [p['confidence'] for p in recent_predictions] if recent_predictions else [0]
        
        # Taxa de balanço
        buy_pct = self.prediction_counts['BUY'] / total * 100
        sell_pct = self.prediction_counts['SELL'] / total * 100
        hold_pct = self.prediction_counts['HOLD'] / total * 100
        
        balance_score = 1.0 - abs(buy_pct - sell_pct) / 100
        
        return {
            'total_predictions': self.total_predictions,
            'accuracy': self.correct_predictions / total if total > 0 else 0,
            'accuracy_pct': (self.correct_predictions / total * 100) if total > 0 else 0,
            'avg_confidence': np.mean(confidences),
            'action_distribution': dict(self.prediction_counts),
            'buy_pct': buy_pct,
            'sell_pct': sell_pct,
            'hold_pct': hold_pct,
            'balance_score': balance_score,
            'feature_importance': self.get_feature_importance(),
            'adaptation_enabled': self.adaptation_enabled,
            'current_thresholds': {
                'buy': self.threshold_buy,
                'sell': self.threshold_sell
            }
        }
    
    def reset_adaptation(self):
        """Reseta pesos e estatísticas"""
        self.feature_weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15], dtype=np.float32)
        self.threshold_buy = 0.3
        self.threshold_sell = -0.3
        self.prediction_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        self.last_predictions.clear()
        logger.info("🔄 ML resetado para valores padrão")
    
    def enable_adaptation(self, enabled: bool = True):
        """Habilita/desabilita adaptação automática"""
        self.adaptation_enabled = enabled
        logger.info(f"🤖 Adaptação ML {'habilitada' if enabled else 'desabilitada'}")


# Importações necessárias
import time