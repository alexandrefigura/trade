# trade_system/analysis/ml.py

"""
Machine Learning simplificado para predições rápidas
"""

import time
from collections import deque
from typing import Dict, Tuple

import numpy as np
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class SimplifiedMLPredictor:
    """ML simplificado para predições rápidas"""

    def __init__(self):
        # Pesos iniciais do modelo (RSI, Momentum, Volume Ratio, Spread, Volatilidade)
        self.feature_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=np.float32)

        # Thresholds adaptativos para decisão
        self.threshold_buy = 0.6
        self.threshold_sell = -0.6

        # Histórico de predições e performance
        self.prediction_history = deque(maxlen=100)
        self.feature_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)

        # Parâmetros de adaptação
        self.learning_rate = 0.01
        self.adaptation_enabled = True

        # Estatísticas
        self.total_predictions = 0
        self.correct_predictions = 0

        logger.info("🤖 ML Predictor inicializado")

    def predict(self, features: Dict) -> Tuple[str, float]:
        """
        Realiza uma predição rápida.

        Args:
            features: {
                'rsi': float,
                'momentum': float,
                'volume_ratio': float,
                'spread_bps': float,
                'volatility': float,
                'price_trend': optional float
            }

        Returns:
            (action, confidence)
        """
        # Confere se todas as features obrigatórias estão presentes
        required = ['rsi', 'momentum', 'volume_ratio', 'spread_bps', 'volatility']
        for feat in required:
            if feat not in features:
                logger.warning(f"Feature ausente: {feat}")
                return 'HOLD', 0.5

        # Normaliza e monta vetor de features
        vec = self._normalize_features(features)

        # Score linear
        score = float(np.dot(vec, self.feature_weights))

        # Ajustes não-lineares
        score = self._apply_nonlinear_factors(score, features)

        # Decide ação e confiança
        action, confidence = self._make_decision(score)

        # Registra e possivelmente adapta
        self.total_predictions += 1
        self._record_prediction(features, score, action, confidence)
        if self.adaptation_enabled and len(self.performance_history) > 20:
            self._adapt_weights()

        return action, confidence

    def _normalize_features(self, features: Dict) -> np.ndarray:
        """Transforma cada feature para a faixa adequada."""
        # RSI: [0,100] → [-1,1]
        rsi = (features['rsi'] - 50.0) / 50.0

        # Momentum: já em torno de 0, limitamos e amplificamos
        mom = np.clip(features['momentum'], -0.1, 0.1) * 10

        # Volume ratio: normaliza em torno de 0
        vr = np.clip(features['volume_ratio'] - 1.0, -2.0, 2.0) / 2.0

        # Spread bps: spread baixo (negativo) é bom → [–1,0]
        sp = np.clip(-features['spread_bps'] / 100.0, -1.0, 0.0)

        # Volatilidade: alta volatilidade penaliza → [–1,0]
        vol = np.clip(-features['volatility'] * 10.0, -1.0, 0.0)

        return np.array([rsi, mom, vr, sp, vol], dtype=np.float32)

    def _apply_nonlinear_factors(self, score: float, features: Dict) -> float:
        """Aplica ajustes adicionais não-lineares baseados em limites de mercado."""
        rsi = features['rsi']
        vol = features['volatility']
        vr = features['volume_ratio']

        # Boost se RSI extremo
        if rsi < 20:
            score += 0.3
        elif rsi > 80:
            score -= 0.3

        # Penaliza alta volatilidade
        if vol > 0.03:
            score *= 0.7

        # Boost para volume muito alto
        if vr > 2:
            score *= 1.2

        # Tendência de preço, se fornecida
        trend = features.get('price_trend')
        if trend is not None:
            score += trend * 0.2

        return float(np.clip(score, -2.0, 2.0))

    def _make_decision(self, score: float) -> Tuple[str, float]:
        """Converte score num rótulo BUY/SELL/HOLD e calcula confiança."""
        # Confiança via sigmoid em |score|
        base_conf = 1.0 / (1.0 + np.exp(-abs(score)))

        if score > self.threshold_buy:
            action = 'BUY'
            # Boost extra se muito acima do threshold
            if score > self.threshold_buy * 1.5:
                base_conf = min(0.95, base_conf * 1.1)
        elif score < self.threshold_sell:
            action = 'SELL'
            if score < self.threshold_sell * 1.5:
                base_conf = min(0.95, base_conf * 1.1)
        else:
            action = 'HOLD'
            base_conf *= 0.6

        return action, float(base_conf)

    def _record_prediction(self, features: Dict, score: float, action: str, confidence: float):
        """Salva histórico da predição para análise/adaptação futura."""
        self.prediction_history.append({
            'timestamp': time.time(),
            'features': features.copy(),
            'score': score,
            'action': action,
            'confidence': confidence,
            'weights': self.feature_weights.copy()
        })
        self.feature_history.append(features.copy())

    def update_performance(self, prediction_id: int, was_correct: bool, profit: float = 0.0):
        """
        Atualiza histórico de performance de uma predição.

        Args:
            prediction_id: índice na history
            was_correct: se acertou o trade
            profit: P&L resultante
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
        """Ajusta pesos se a performance recente estiver abaixo do esperado."""
        recent = list(self.performance_history)[-20:]
        accuracy = sum(p['correct'] for p in recent) / len(recent)
        if accuracy < 0.45:
            noise = np.random.randn(len(self.feature_weights)) * self.learning_rate
            self.feature_weights += noise
            # Garante pesos positivos somando a 1
            self.feature_weights = np.abs(self.feature_weights)
            self.feature_weights /= np.sum(self.feature_weights)
            logger.info(f"🔧 Pesos adaptados: {self.feature_weights}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna a importância relativa de cada feature."""
        total = float(np.sum(self.feature_weights))
        keys = ['rsi', 'momentum', 'volume_ratio', 'spread', 'volatility']
        return {k: float(w / total) for k, w in zip(keys, self.feature_weights)}

    def get_prediction_stats(self) -> Dict:
        """Devolve estatísticas agregadas de predições recentes."""
        if self.total_predictions == 0:
            return {
                'total': 0, 'accuracy': 0.0, 'avg_confidence': 0.0,
                'distribution': {}, 'feature_importance': {}, 'adaptation': self.adaptation_enabled
            }

        recent = list(self.prediction_history)[-50:]
        dist = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confs = []
        for rec in recent:
            dist[rec['action']] += 1
            confs.append(rec['confidence'])

        accuracy = self.correct_predictions / self.total_predictions
        return {
            'total': self.total_predictions,
            'accuracy': accuracy,
            'accuracy_pct': accuracy * 100,
            'avg_confidence': float(np.mean(confs)),
            'distribution': dist,
            'feature_importance': self.get_feature_importance(),
            'adaptation': self.adaptation_enabled
        }

    def reset_adaptation(self):
        """Reseta os pesos aos valores iniciais."""
        self.feature_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=np.float32)
        logger.info("🔄 Pesos do ML resetados para padrão")

    def enable_adaptation(self, enabled: bool = True):
        """Habilita ou desabilita adaptação automática."""
        self.adaptation_enabled = enabled
        logger.info(f"🤖 Adaptação {'habilitada' if enabled else 'desabilitada'}")
