"""
Machine Learning simplificado para predições rápidas
"""
import numpy as np
from collections import deque
from typing import Dict, Tuple, List
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class SimplifiedMLPredictor:
    """ML simplificado para predições rápidas"""
    
    def __init__(self):
        # Pesos do modelo
        self.feature_weights = np.array([
            0.3,   # RSI
            0.25,  # Momentum
            0.2,   # Volume ratio
            0.15,  # Spread
            0.1    # Volatility
        ], dtype=np.float32)
        
        # Thresholds adaptativos
        self.threshold_buy = 0.6
        self.threshold_sell = -0.6
        
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
        
        logger.info("🤖 ML Predictor inicializado")
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        """
        Predição ultra-rápida
        
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
        
        # Score ponderado
        score = np.dot(feature_vector, self.feature_weights)
        
        # Adicionar fatores não-lineares
        score = self._apply_nonlinear_factors(score, features)
        
        # Decisão com thresholds adaptativos
        action, confidence = self._make_decision(score)
        
        # Registrar
        self.total_predictions += 1
        self._record_prediction(features, score, action, confidence)
        
        # Adaptar se habilitado
        if self.adaptation_enabled and len(self.performance_history) > 20:
            self._adapt_weights()
        
        return action, confidence
    
    def _normalize_features(self, features: Dict) -> np.ndarray:
        """Normaliza features para o modelo"""
        # RSI: converter para -1 a 1
        rsi_norm = (features.get('rsi', 50) - 50) / 50
        
        # Momentum: já está normalizado
        momentum = features.get('momentum', 0)
        momentum = np.clip(momentum, -0.1, 0.1)  # Limitar extremos
        
        # Volume ratio: normalizar em torno de 1
        volume_ratio = features.get('volume_ratio', 1) - 1
        volume_ratio = np.clip(volume_ratio, -2, 2) / 2
        
        # Spread: negativo é bom
        spread_norm = -features.get('spread_bps', 0) / 100
        spread_norm = np.clip(spread_norm, -1, 0)
        
        # Volatilidade: negativa é boa para estabilidade
        volatility_norm = -features.get('volatility', 0.01) * 10
        volatility_norm = np.clip(volatility_norm, -1, 0)
        
        return np.array([
            rsi_norm,
            momentum * 10,  # Amplificar momentum
            volume_ratio,
            spread_norm,
            volatility_norm
        ], dtype=np.float32)
    
    def _apply_nonlinear_factors(self, score: float, features: Dict) -> float:
        """Aplica fatores não-lineares ao score"""
        # Boost para RSI extremo
        rsi = features.get('rsi', 50)
        if rsi < 20:
            score += 0.3
        elif rsi > 80:
            score -= 0.3
        
        # Penalizar alta volatilidade
        if features.get('volatility', 0) > 0.03:
            score *= 0.7
        
        # Boost para volume alto
        if features.get('volume_ratio', 1) > 2:
            score *= 1.2
        
        # Considerar tendência
        if 'price_trend' in features:
            trend = features['price_trend']
            score += trend * 0.2
        
        return np.clip(score, -2, 2)
    
    def _make_decision(self, score: float) -> Tuple[str, float]:
        """Toma decisão baseada no score"""
        # Aplicar sigmoid para confiança
        confidence = 1 / (1 + np.exp(-abs(score)))
        
        if score > self.threshold_buy:
            action = 'BUY'
            # Boost de confiança para scores muito altos
            if score > self.threshold_buy * 1.5:
                confidence = min(0.95, confidence * 1.1)
        elif score < self.threshold_sell:
            action = 'SELL'
            if score < self.threshold_sell * 1.5:
                confidence = min(0.95, confidence * 1.1)
        else:
            action = 'HOLD'
            confidence *= 0.6  # Reduzir confiança em HOLD
        
        return action, float(confidence)
    
    def _record_prediction(self, features: Dict, score: float, action: str, confidence: float):
        """Registra predição no histórico"""
        record = {
            'timestamp': time.time(),
            'features': features.copy(),
            'score': score,
            'action': action,
            'confidence': confidence,
            'weights': self.feature_weights.copy()
        }
        
        self.prediction_history.append(record)
        self.feature_history.append(features)
    
    def update_performance(self, prediction_id: int, was_correct: bool, profit: float = 0):
        """
        Atualiza performance de uma predição
        
        Args:
            prediction_id: ID da predição
            was_correct: Se a predição foi correta
            profit: Lucro/prejuízo resultante
        """
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
            # Adicionar ruído aos pesos
            noise = np.random.randn(len(self.feature_weights)) * self.learning_rate
            self.feature_weights += noise
            
            # Normalizar pesos
            self.feature_weights = np.abs(self.feature_weights)
            self.feature_weights /= np.sum(self.feature_weights)
            
            logger.info(f"🔧 Pesos adaptados. Nova distribuição: {self.feature_weights}")
    
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
        if self.total_predictions == 0:
            return {
                'total_predictions': 0,
                'accuracy': 0,
                'avg_confidence': 0
            }
        
        recent_predictions = list(self.prediction_history)[-50:]
        
        # Contar ações
        action_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidences = []
        
        for pred in recent_predictions:
            action_counts[pred['action']] += 1
            confidences.append(pred['confidence'])
        
        accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        
        return {
            'total_predictions': self.total_predictions,
            'accuracy': accuracy,
            'accuracy_pct': accuracy * 100,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'action_distribution': action_counts,
            'feature_importance': self.get_feature_importance(),
            'adaptation_enabled': self.adaptation_enabled
        }
    
    def reset_adaptation(self):
        """Reseta pesos para valores padrão"""
        self.feature_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=np.float32)
        logger.info("🔄 Pesos do ML resetados para padrão")
    
    def enable_adaptation(self, enabled: bool = True):
        """Habilita/desabilita adaptação automática"""
        self.adaptation_enabled = enabled
        logger.info(f"🤖 Adaptação {'habilitada' if enabled else 'desabilitada'}")


# Importações necessárias
import time
