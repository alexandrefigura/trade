"""Sistema de Machine Learning para predição"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import logging
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path

class MLPredictor:
    """Preditor ML avançado para trading"""
    
    def __init__(self, config: Any):
        self.config = config.ml if hasattr(config, 'ml') else config.get('ml', {})
        self.logger = logging.getLogger(__name__)
        
        # Modelos
        self.model = None
        self.scaler = StandardScaler()
        
        # Features
        self.feature_names = self.config.get('features', [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
            'bb_middle', 'volume_ratio', 'price_change', 'volatility',
            'momentum', 'support', 'resistance', 'trend_strength',
            'volume_profile', 'ema_diff', 'atr'
        ])
        
        # Estado
        self.is_trained = False
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        # Métricas
        self.training_metrics = {}
        
        self.logger.info("🤖 ML Predictor inicializado")
    
    def prepare_features(self, candles: pd.DataFrame, 
                        ta_indicators: Dict[str, float]) -> np.ndarray:
        """
        Prepara features para o modelo
        
        Args:
            candles: DataFrame com dados OHLCV
            ta_indicators: Indicadores técnicos calculados
            
        Returns:
            Array de features normalizado
        """
        features = []
        
        for feature_name in self.feature_names:
            if feature_name in ta_indicators:
                value = ta_indicators[feature_name]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    features.append(value)
                else:
                    self.logger.warning(f"Feature ausente: {feature_name}")
                    features.append(0.0)
            else:
                features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def train(self, candles: pd.DataFrame, ta_analyzer: Any):
        """
        Treina o modelo com dados históricos
        
        Args:
            candles: DataFrame com dados históricos
            ta_analyzer: Analisador técnico para extrair features
        """
        try:
            self.logger.info("🔄 Iniciando treinamento do modelo ML...")
            
            # Preparar dataset
            X, y = self._prepare_training_data(candles, ta_analyzer)
            
            if len(X) < 100:
                self.logger.warning("Dados insuficientes para treinar ML")
                return
            
            # Normalizar features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split treino/teste
            test_size = self.config.get('test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Treinar modelo ensemble
            self.model = self._create_ensemble_model()
            self.model.fit(X_train, y_train)
            
            # Avaliar modelo
            self._evaluate_model(X_test, y_test)
            
            # Salvar modelo
            self._save_model()
            
            self.is_trained = True
            self.logger.info("✅ Modelo ML treinado com sucesso!")
            
        except Exception as e:
            self.logger.error(f"Erro ao treinar ML: {e}")
            self.is_trained = False
    
    def _prepare_training_data(self, candles: pd.DataFrame, 
                              ta_analyzer: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para treinamento"""
        lookback = self.config.get('lookback', 100)
        X = []
        y = []
        
        for i in range(lookback, len(candles) - 1):
            # Analisar período
            period_candles = candles.iloc[i-lookback:i].copy()
            analysis = ta_analyzer.analyze(period_candles)
            
            if analysis['indicators']:
                # Extrair features
                features = self.prepare_features(period_candles, analysis['indicators'])
                X.append(features[0])
                
                # Criar label (1 = preço sobe, 0 = preço desce)
                current_price = candles.iloc[i]['close']
                future_price = candles.iloc[i+1]['close']
                
                # Considerar movimento significativo (> 0.1%)
                price_change = (future_price - current_price) / current_price
                y.append(1 if price_change > 0.001 else 0)
        
        return np.array(X), np.array(y)
    
    def _create_ensemble_model(self):
        """Cria modelo ensemble"""
        model_type = self.config.get('model_type', 'gradient_boosting')
        n_estimators = self.config.get('n_estimators', 100)
        
        if model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                subsample=0.8,
                min_samples_split=5
            )
        else:  # random_forest
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                random_state=42,
                min_samples_split=5,
                class_weight='balanced'
            )
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Avalia performance do modelo"""
        y_pred = self.model.predict(X_test)
        
        self.training_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'samples': len(y_test)
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            top_features = sorted(feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            
            self.logger.info("Top 5 features:")
            for feature, importance in top_features:
                self.logger.info(f"  - {feature}: {importance:.3f}")
        
        self.logger.info(f"Métricas do modelo:")
        self.logger.info(f"  - Acurácia: {self.training_metrics['accuracy']:.2%}")
        self.logger.info(f"  - Precisão: {self.training_metrics['precision']:.2%}")
        self.logger.info(f"  - Recall: {self.training_metrics['recall']:.2%}")
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Faz predição com o modelo treinado
        
        Args:
            features: Features normalizadas
            
        Returns:
            Tuple (sinal, confiança)
        """
        if not self.is_trained or self.model is None:
            return 'HOLD', 0.0
        
        try:
            # Normalizar features
            features_scaled = self.scaler.transform(features)
            
            # Predição com probabilidade
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Confiança é a probabilidade da classe predita
            confidence = max(probabilities)
            
            # Ajustar threshold baseado nas métricas de treinamento
            threshold = 0.6
            if self.training_metrics.get('accuracy', 0) < 0.55:
                threshold = 0.7  # Mais conservador se acurácia baixa
            
            if prediction == 1 and confidence > threshold:
                return 'BUY', confidence
            elif prediction == 0 and confidence > threshold:
                return 'SELL', confidence
            else:
                return 'HOLD', confidence
            
        except Exception as e:
            self.logger.error(f"Erro na predição ML: {e}")
            return 'HOLD', 0.0
    
    def _save_model(self):
        """Salva modelo treinado"""
        try:
            model_file = self.model_path / "trading_model.pkl"
            scaler_file = self.model_path / "scaler.pkl"
            
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            # Salvar métricas
            metrics_file = self.model_path / "metrics.json"
            import json
            with open(metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            
            self.logger.info(f"Modelo salvo em: {model_file}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {e}")
    
    def load_model(self) -> bool:
        """Carrega modelo salvo"""
        try:
            model_file = self.model_path / "trading_model.pkl"
            scaler_file = self.model_path / "scaler.pkl"
            
            if model_file.exists() and scaler_file.exists():
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                self.is_trained = True
                
                self.logger.info("✅ Modelo carregado com sucesso")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def should_retrain(self, trades_count: int) -> bool:
        """Verifica se deve retreinar o modelo"""
        retrain_interval = self.config.get('retrain_interval', 1000)
        return trades_count % retrain_interval == 0
