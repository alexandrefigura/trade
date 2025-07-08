"""
Machine Learning avançado com modelos de ensemble
Suporta LightGBM, XGBoost e modelo linear como fallback
"""
import numpy as np
import pandas as pd
import random
import time
import pickle
import json
from collections import deque
from typing import Dict, Tuple, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

# Modelos avançados (instalação condicional)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Feature engineering e validação
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier

from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class AdvancedMLPredictor:
    """
    Sistema ML avançado com múltiplos modelos e ensemble
    
    Features principais:
    - Modelos: LightGBM, XGBoost, Random Forest, SGD, Passive Aggressive
    - Feature Engineering avançado com VWAP, CMF, momentum composto, volatilidade implícita
    - Aprendizado online com partial_fit após N candles
    - Retreinamento automático adaptativo
    - Cache de predições para performance
    - Adaptação a mudanças de regime de mercado
    - Métricas detalhadas e importância de features
    
    Uso:
        predictor = AdvancedMLPredictor(model_type='auto')
        action, confidence = predictor.predict(features)
        predictor.add_training_sample(features, action, reward, was_correct)
    """
    
    def __init__(self, model_type: str = 'auto', config: Dict = None):
        """
        Args:
            model_type: 'lightgbm', 'xgboost', 'ensemble', 'simple', 'auto'
            config: Configurações customizadas
        """
        self.model_type = self._select_model_type(model_type)
        self.config = config or {}
        
        # Modelos
        self.models = {}
        self.active_model = None
        self.simple_predictor = SimplifiedMLPredictor()  # Fallback
        
        # Feature engineering
        self.scaler = RobustScaler()  # Mais robusto a outliers
        self.feature_names = []
        self.engineered_features = []
        
        # Dados de treinamento
        self.training_data = deque(maxlen=10000)
        self.feature_buffer = deque(maxlen=1000)
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = deque(maxlen=1000)
        self.last_training_time = None
        self.min_training_samples = 500
        
        # Configurações
        self.retrain_interval = timedelta(hours=self.config.get('retrain_hours', 24))
        self.min_accuracy = self.config.get('min_accuracy', 0.55)
        self.use_ensemble = self.config.get('use_ensemble', True)
        
        # Cache de predições
        self.prediction_cache = {}
        self.cache_ttl = 60  # segundos
        
        # Regime de mercado
        self._last_regime = None
        
        # Inicializar modelos
        self._initialize_models()
        
        logger.info(f"🤖 ML Avançado inicializado: {self.model_type}")
    
    def _select_model_type(self, requested_type: str) -> str:
        """Seleciona o melhor modelo disponível"""
        if requested_type == 'auto':
            if LIGHTGBM_AVAILABLE:
                return 'lightgbm'
            elif XGBOOST_AVAILABLE:
                return 'xgboost'
            else:
                logger.warning("LightGBM e XGBoost não disponíveis, usando modelo simples")
                return 'simple'
        
        if requested_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM não disponível, usando alternativa")
            return 'xgboost' if XGBOOST_AVAILABLE else 'simple'
            
        if requested_type == 'xgboost' and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost não disponível, usando alternativa")
            return 'lightgbm' if LIGHTGBM_AVAILABLE else 'simple'
            
        return requested_type
    
    def _initialize_models(self):
        """Inicializa os modelos ML com suporte para aprendizado incremental"""
        # LightGBM com configurações otimizadas
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                importance_type='gain'
            )
        
        # XGBoost com early stopping
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                eval_metric='logloss',
                early_stopping_rounds=10
            )
        
        # Random Forest como modelo adicional
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            warm_start=True  # Permite adicionar árvores incrementalmente
        )
        
        # SGD Classifier para aprendizado online verdadeiro
        from sklearn.linear_model import SGDClassifier
        self.models['sgd_online'] = SGDClassifier(
            loss='log_loss',  # Para probabilidades
            penalty='elasticnet',
            alpha=0.001,
            l1_ratio=0.5,
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            warm_start=True  # Crucial para partial_fit
        )
        
        # Passive Aggressive Classifier - bom para dados não-estacionários
        from sklearn.linear_model import PassiveAggressiveClassifier
        self.models['passive_aggressive'] = PassiveAggressiveClassifier(
            C=0.1,
            random_state=42,
            warm_start=True
        )
        
        # Ensemble Voting Classifier
        if self.use_ensemble and len(self.models) > 1:
            # Para ensemble online, usar apenas modelos que suportam partial_fit
            online_models = ['sgd_online', 'passive_aggressive']
            estimators = [(name, self.models[name]) for name in online_models if name in self.models]
            if estimators:
                self.models['ensemble_online'] = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    n_jobs=-1
                )
        
        # Selecionar modelo ativo
        if self.model_type in self.models:
            self.active_model = self.models[self.model_type]
        else:
            self.active_model = None
        
        # Tracking para retreinamento online
        self.online_batch_size = self.config.get('online_batch_size', 50)
        self.online_buffer = deque(maxlen=self.online_batch_size)
        self.candles_since_retrain = 0
        self.retrain_every_n_candles = self.config.get('retrain_candles', 100)
        self.partial_fit_enabled = True
        
        logger.info(f"🤖 Modelos inicializados com suporte para aprendizado online")
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        """
        Predição usando modelo avançado
        
        Args:
            features: Features do mercado
            
        Returns:
            Tupla (action, confidence)
        """
        # Verificar cache
        cache_key = self._get_cache_key(features)
        if cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['prediction']
        
        # Se não há modelo treinado, usar predictor simples
        if self.active_model is None or not hasattr(self.active_model, 'predict_proba'):
            return self.simple_predictor.predict(features)
        
        try:
            # Engenharia de features
            feature_vector = self._engineer_features(features)
            
            # Detectar e adaptar ao regime de mercado
            current_regime = self.detect_market_regime(features)
            if hasattr(self, '_last_regime') and self._last_regime != current_regime:
                logger.info(f"📊 Mudança de regime detectada: {self._last_regime} → {current_regime}")
                self.adapt_to_regime_change(current_regime)
            self._last_regime = current_regime
            
            # Fazer predição
            X = feature_vector.reshape(1, -1)
            
            # Normalizar se o scaler foi ajustado
            if hasattr(self.scaler, 'scale_'):
                X = self.scaler.transform(X)
            
            # Predição probabilística
            proba = self.active_model.predict_proba(X)[0]
            prediction = self.active_model.predict(X)[0]
            
            # Converter para ação
            action = self._convert_prediction_to_action(prediction)
            
            # Calcular confiança
            confidence = self._calculate_confidence(proba, features)
            
            # Aplicar filtros de segurança
            action, confidence = self._apply_safety_filters(action, confidence, features)
            
            # Cachear resultado
            self.prediction_cache[cache_key] = {
                'prediction': (action, confidence),
                'timestamp': time.time()
            }
            
            # Registrar
            self._record_prediction(features, action, confidence, proba)
            
            # Verificar se precisa retreinar
            if self._should_retrain():
                self._retrain_model()
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"Erro na predição ML: {e}")
            # Fallback para modelo simples
            return self.simple_predictor.predict(features)
    
    def _engineer_features(self, features: Dict) -> np.ndarray:
        """Engenharia de features avançada com indicadores técnicos complexos"""
        base_features = []
        
        # Features básicas
        rsi = features.get('rsi', 50)
        momentum = features.get('momentum', 0)
        volume_ratio = features.get('volume_ratio', 1)
        spread_bps = features.get('spread_bps', 10)
        volatility = features.get('volatility', 0.01)
        
        # Features normalizadas
        base_features.extend([
            (rsi - 50) / 50,  # RSI normalizado
            np.clip(momentum, -0.1, 0.1) * 10,  # Momentum
            np.log(volume_ratio) if volume_ratio > 0 else 0,  # Log volume
            -np.log(spread_bps + 1) / 10,  # Spread invertido
            -np.log(volatility * 100 + 1) / 5  # Volatilidade
        ])
        
        # VWAP (Volume Weighted Average Price)
        if 'vwap' in features:
            vwap = features['vwap']
            price = features.get('price', vwap)
            vwap_ratio = (price - vwap) / vwap if vwap > 0 else 0
            base_features.extend([
                np.clip(vwap_ratio, -0.1, 0.1),  # VWAP deviation
                1 if price > vwap else 0,  # Above VWAP
                abs(vwap_ratio) * 100  # VWAP distance
            ])
        
        # Chaikin Money Flow (CMF)
        if 'cmf' in features:
            cmf = features['cmf']
            base_features.extend([
                cmf,  # Raw CMF (-1 to 1)
                1 if cmf > 0.2 else 0,  # Strong buying
                1 if cmf < -0.2 else 0,  # Strong selling
                cmf ** 2  # Non-linear CMF
            ])
        
        # Momentum Composto
        if 'momentum_5' in features and 'momentum_10' in features:
            mom5 = features['momentum_5']
            mom10 = features['momentum_10']
            mom20 = features.get('momentum_20', mom10)
            
            # Momentum composto multi-período
            compound_momentum = (mom5 * 0.5 + mom10 * 0.3 + mom20 * 0.2)
            momentum_acceleration = mom5 - mom10  # Aceleração
            
            base_features.extend([
                np.clip(compound_momentum, -0.1, 0.1) * 10,
                momentum_acceleration * 50,
                1 if mom5 > 0 and mom10 > 0 and mom20 > 0 else 0,  # Trend alignment
                np.sign(mom5) * np.log(abs(mom5) + 1)  # Log momentum
            ])
        
        # Volatilidade Implícita e Realized
        if 'implied_volatility' in features:
            iv = features['implied_volatility']
            rv = features.get('realized_volatility', volatility)
            vol_premium = iv - rv  # Prêmio de volatilidade
            
            base_features.extend([
                np.clip(iv, 0, 1),  # IV normalizada
                np.clip(vol_premium, -0.1, 0.1),  # Vol premium
                1 if vol_premium > 0.02 else 0,  # High IV
                iv / (rv + 0.001)  # IV/RV ratio
            ])
        
        # Features derivadas avançadas
        # RSI zones com Stochastic RSI
        if 'stoch_rsi' in features:
            stoch_rsi = features['stoch_rsi']
            base_features.extend([
                stoch_rsi,
                1 if stoch_rsi < 20 else 0,  # Extreme oversold
                1 if stoch_rsi > 80 else 0,  # Extreme overbought
                abs(rsi - stoch_rsi) / 100  # RSI divergence
            ])
        else:
            # RSI zones tradicionais
            base_features.extend([
                1 if rsi < 30 else 0,  # Oversold
                1 if rsi > 70 else 0,  # Overbought
                1 if 45 < rsi < 55 else 0  # Neutral
            ])
        
        # Volume patterns avançados
        if 'obv' in features:  # On-Balance Volume
            obv_norm = features['obv']
            base_features.extend([
                np.clip(obv_norm, -1, 1),
                np.sign(obv_norm) * np.log(abs(obv_norm) + 1)
            ])
        else:
            base_features.extend([
                1 if volume_ratio > 2 else 0,  # Volume spike
                1 if volume_ratio < 0.5 else 0,  # Low volume
                np.log(volume_ratio) ** 2  # Non-linear volume
            ])
        
        # Market Microstructure
        if 'bid_ask_spread' in features:
            micro_spread = features['bid_ask_spread']
            base_features.extend([
                np.log(micro_spread + 1),
                1 if micro_spread < features.get('avg_spread', 10) else 0
            ])
        
        # Order Flow
        if 'order_flow_imbalance' in features:
            ofi = features['order_flow_imbalance']
            base_features.extend([
                np.clip(ofi, -1, 1),
                ofi ** 2,
                1 if abs(ofi) > 0.7 else 0
            ])
        
        # Volatility regimes com GARCH
        if 'garch_forecast' in features:
            garch_vol = features['garch_forecast']
            base_features.extend([
                np.clip(garch_vol, 0, 0.1),
                1 if garch_vol < volatility else 0,  # Vol compression
                garch_vol / (volatility + 0.001)  # Vol ratio
            ])
        else:
            base_features.extend([
                1 if volatility < 0.005 else 0,  # Low vol
                1 if volatility > 0.03 else 0,  # High vol
                volatility ** 0.5  # Sqrt volatility
            ])
        
        # Interações complexas entre features
        base_features.extend([
            rsi * momentum,  # RSI-momentum interaction
            volume_ratio * volatility,  # Volume-volatility
            spread_bps * volatility,  # Spread-volatility
        ])
        
        # Adicionar interações com VWAP e CMF se disponíveis
        if 'vwap' in features and 'cmf' in features:
            vwap_ratio = (features.get('price', 0) - features['vwap']) / features['vwap'] if features['vwap'] > 0 else 0
            base_features.extend([
                vwap_ratio * features['cmf'],  # VWAP-CMF interaction
                vwap_ratio * momentum,  # VWAP-momentum
            ])
        
        # Features de orderbook se disponível
        if 'orderbook_imbalance' in features:
            imbalance = features['orderbook_imbalance']
            base_features.extend([
                imbalance,
                imbalance ** 2,
                1 if imbalance > 0.8 else 0,
                1 if imbalance < -0.8 else 0
            ])
        
        # Features de tempo se disponível
        if 'hour' in features:
            hour = features['hour']
            base_features.extend([
                np.sin(2 * np.pi * hour / 24),  # Cyclical hour
                np.cos(2 * np.pi * hour / 24),
                1 if 9 <= hour <= 17 else 0  # Trading hours
            ])
        
        # Features de tendência se disponível
        if 'sma_cross' in features:
            base_features.append(features['sma_cross'])
        
        if 'ema_cross' in features:
            base_features.append(features['ema_cross'])
        
        # Adicionar features históricas do buffer com mais complexidade
        if len(self.feature_buffer) >= 10:
            recent = list(self.feature_buffer)[-10:]
            
            # Média móvel de RSI
            rsi_values = [f.get('rsi', 50) for f in recent]
            avg_rsi = np.mean(rsi_values)
            rsi_std = np.std(rsi_values)
            base_features.extend([
                (avg_rsi - 50) / 50,
                rsi_std / 50  # RSI volatility
            ])
            
            # Tendência de volume
            volumes = [f.get('volume_ratio', 1) for f in recent]
            vol_trend = (volumes[-1] - volumes[0]) / (volumes[0] + 0.001)
            vol_acceleration = np.gradient(volumes).mean()
            base_features.extend([
                np.clip(vol_trend, -1, 1),
                np.clip(vol_acceleration, -1, 1)
            ])
            
            # Momentum histórico
            if 'momentum' in recent[0]:
                momentums = [f.get('momentum', 0) for f in recent]
                mom_mean = np.mean(momentums)
                mom_std = np.std(momentums)
                base_features.extend([
                    np.clip(mom_mean * 10, -1, 1),
                    mom_std * 100  # Momentum volatility
                ])
        
        return np.array(base_features, dtype=np.float32)
    
    def _convert_prediction_to_action(self, prediction: int) -> str:
        """Converte predição numérica para ação"""
        # Assumindo 3 classes: 0=SELL, 1=HOLD, 2=BUY
        if prediction == 0:
            return 'SELL'
        elif prediction == 2:
            return 'BUY'
        else:
            return 'HOLD'
    
    def _calculate_confidence(self, proba: np.ndarray, features: Dict) -> float:
        """Calcula confiança ajustada da predição"""
        # Confiança base é a probabilidade máxima
        base_confidence = float(np.max(proba))
        
        # Ajustar por spread de probabilidades
        prob_spread = np.max(proba) - np.min(proba)
        confidence = base_confidence * (0.5 + 0.5 * prob_spread)
        
        # Penalizar por alta volatilidade
        volatility = features.get('volatility', 0.01)
        if volatility > 0.03:
            confidence *= 0.8
        
        # Penalizar por spread alto
        spread_bps = features.get('spread_bps', 10)
        if spread_bps > 20:
            confidence *= 0.9
        
        # Boost por volume alto
        volume_ratio = features.get('volume_ratio', 1)
        if volume_ratio > 2:
            confidence = min(0.95, confidence * 1.1)
        
        return float(np.clip(confidence, 0.1, 0.95))
    
    def _apply_safety_filters(self, action: str, confidence: float, 
                            features: Dict) -> Tuple[str, float]:
        """Aplica filtros de segurança na predição"""
        # Evitar trades em condições extremas
        volatility = features.get('volatility', 0.01)
        spread_bps = features.get('spread_bps', 10)
        
        if volatility > 0.05 or spread_bps > 30:
            # Forçar HOLD em condições muito adversas
            return 'HOLD', confidence * 0.5
        
        # Anti-overtrading: verificar últimas predições
        recent_actions = [p['action'] for p in list(self.prediction_history)[-10:]]
        if len(recent_actions) >= 10:
            action_counts = {
                'BUY': recent_actions.count('BUY'),
                'SELL': recent_actions.count('SELL'),
                'HOLD': recent_actions.count('HOLD')
            }
            
            # Se muito enviesado, reduzir confiança
            if action != 'HOLD':
                if action_counts[action] > 7:
                    confidence *= 0.7
        
        return action, confidence
    
    def _should_retrain(self) -> bool:
        """Verifica se deve retreinar o modelo"""
        if len(self.training_data) < self.min_training_samples:
            return False
        
        if self.last_training_time is None:
            return True
        
        # Retreinar periodicamente
        if datetime.now() - self.last_training_time > self.retrain_interval:
            return True
        
        # Retreinar se performance caiu
        if self._check_performance_degradation():
            return True
        
        return False
    
    def _check_performance_degradation(self) -> bool:
        """Verifica se a performance degradou"""
        if len(self.prediction_history) < 50:
            return False
        
        recent = list(self.prediction_history)[-50:]
        recent_accuracy = sum(p.get('correct', False) for p in recent) / len(recent)
        
        return recent_accuracy < self.min_accuracy
    
    def _retrain_model(self):
        """Retreina o modelo com dados recentes"""
        try:
            logger.info("🔄 Iniciando retreinamento do modelo ML")
            
            # Preparar dados
            X, y = self._prepare_training_data()
            
            if len(X) < self.min_training_samples:
                logger.warning(f"Dados insuficientes para treinar: {len(X)}")
                return
            
            # Dividir dados (time series split)
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Treinar e avaliar cada modelo
            best_score = 0
            best_model = None
            
            for model_name, model in self.models.items():
                if model_name == 'ensemble' and not self.use_ensemble:
                    continue
                
                try:
                    # Cross-validation
                    scores = cross_val_score(model, X, y, cv=tscv, 
                                           scoring='accuracy', n_jobs=-1)
                    avg_score = np.mean(scores)
                    
                    logger.info(f"  {model_name}: {avg_score:.3f} (+/- {np.std(scores):.3f})")
                    
                    # Treinar no conjunto completo
                    model.fit(X, y)
                    
                    # Atualizar melhor modelo
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model_name
                    
                    # Salvar métricas
                    self.model_performance[model_name] = {
                        'accuracy': avg_score,
                        'std': np.std(scores),
                        'last_trained': datetime.now()
                    }
                    
                except Exception as e:
                    logger.error(f"Erro treinando {model_name}: {e}")
            
            # Selecionar melhor modelo
            if best_model and best_score > self.min_accuracy:
                self.active_model = self.models[best_model]
                self.model_type = best_model
                logger.info(f"✅ Modelo ativo: {best_model} (accuracy: {best_score:.3f})")
            else:
                logger.warning("Nenhum modelo atingiu accuracy mínima")
            
            self.last_training_time = datetime.now()
            
            # Salvar modelo
            self._save_model()
            
        except Exception as e:
            logger.error(f"Erro no retreinamento: {e}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para treinamento"""
        X = []
        y = []
        
        for sample in self.training_data:
            if 'features' in sample and 'label' in sample:
                feature_vec = self._engineer_features(sample['features'])
                X.append(feature_vec)
                
                # Converter label para numérico
                label = sample['label']
                if label == 'SELL':
                    y.append(0)
                elif label == 'HOLD':
                    y.append(1)
                else:  # BUY
                    y.append(2)
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalizar features
        if len(X) > 0:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        
        return X, y
    
    def add_training_sample(self, features: Dict, action: str, 
                          reward: float = None, was_correct: bool = None):
        """Adiciona amostra aos dados de treinamento com suporte para aprendizado online"""
        # Calcular features técnicas avançadas se não presentes
        features = self._calculate_technical_features(features)
        
        sample = {
            'features': features.copy(),
            'label': action,
            'reward': reward,
            'correct': was_correct,
            'timestamp': time.time()
        }
        
        self.training_data.append(sample)
        self.feature_buffer.append(features)
        self.online_buffer.append(sample)
        
        # Incrementar contador de candles
        self.candles_since_retrain += 1
        
        # Atualizar histórico de predições se foi uma predição real
        if was_correct is not None:
            if len(self.prediction_history) > 0:
                self.prediction_history[-1]['correct'] = was_correct
        
        # Verificar se deve fazer partial fit
        if self.partial_fit_enabled:
            # Opção 1: A cada N candles
            if self.candles_since_retrain >= self.retrain_every_n_candles:
                self._online_retrain()
                self.candles_since_retrain = 0
            
            # Opção 2: Quando o buffer online está cheio
            elif len(self.online_buffer) >= self.online_batch_size:
                self._partial_fit_batch()
    
    def _calculate_technical_features(self, features: Dict) -> Dict:
        """Calcula features técnicas avançadas se não estiverem presentes"""
        # Se já tem as features avançadas, retornar
        if all(key in features for key in ['vwap', 'cmf', 'momentum_5']):
            return features
        
        # Simular cálculo de features técnicas (em produção, viriam do TA)
        enhanced_features = features.copy()
        
        # VWAP simulado
        if 'vwap' not in enhanced_features and 'price' in enhanced_features:
            price = enhanced_features['price']
            volume = enhanced_features.get('volume', 1)
            # Simulação simplificada
            enhanced_features['vwap'] = price * (1 + np.random.uniform(-0.001, 0.001))
        
        # CMF simulado
        if 'cmf' not in enhanced_features:
            # Baseado em volume e momentum
            volume_ratio = enhanced_features.get('volume_ratio', 1)
            momentum = enhanced_features.get('momentum', 0)
            enhanced_features['cmf'] = np.clip(momentum * volume_ratio * 2, -1, 1)
        
        # Momentum multi-período
        if 'momentum_5' not in enhanced_features:
            base_momentum = enhanced_features.get('momentum', 0)
            enhanced_features['momentum_5'] = base_momentum * 1.1
            enhanced_features['momentum_10'] = base_momentum * 0.9
            enhanced_features['momentum_20'] = base_momentum * 0.8
        
        # Volatilidade implícita simulada
        if 'implied_volatility' not in enhanced_features:
            realized_vol = enhanced_features.get('volatility', 0.01)
            # IV geralmente maior que RV
            enhanced_features['implied_volatility'] = realized_vol * np.random.uniform(1.1, 1.3)
            enhanced_features['realized_volatility'] = realized_vol
        
        return enhanced_features
    
    def _online_retrain(self):
        """Retreinamento online completo a cada N candles"""
        try:
            logger.info(f"🔄 Retreinamento online após {self.retrain_every_n_candles} candles")
            
            # Usar dados recentes para retreinamento
            recent_samples = list(self.training_data)[-2000:]  # Últimas 2000 amostras
            
            if len(recent_samples) < self.min_training_samples:
                logger.warning("Dados insuficientes para retreinamento online")
                return
            
            # Preparar dados
            X = []
            y = []
            for sample in recent_samples:
                if 'features' in sample and 'label' in sample:
                    feature_vec = self._engineer_features(sample['features'])
                    X.append(feature_vec)
                    
                    # Converter label
                    label = sample['label']
                    if label == 'SELL':
                        y.append(0)
                    elif label == 'HOLD':
                        y.append(1)
                    else:  # BUY
                        y.append(2)
            
            X = np.array(X)
            y = np.array(y)
            
            # Normalizar
            if len(X) > 0:
                self.scaler.fit(X)
                X = self.scaler.transform(X)
            
            # Retreinar modelo ativo
            if self.active_model is not None:
                # Para modelos que suportam warm_start
                if hasattr(self.active_model, 'warm_start') and self.active_model.warm_start:
                    self.active_model.fit(X, y)
                else:
                    # Retreinamento completo
                    self.active_model.fit(X, y)
                
                logger.info("✅ Retreinamento online concluído")
            
            # Limpar buffer online
            self.online_buffer.clear()
            
        except Exception as e:
            logger.error(f"Erro no retreinamento online: {e}")
    
    def _partial_fit_batch(self):
        """Faz partial fit com o batch atual (aprendizado incremental)"""
        try:
            if not self.online_buffer:
                return
            
            logger.debug(f"📊 Partial fit com {len(self.online_buffer)} amostras")
            
            # Preparar batch
            X_batch = []
            y_batch = []
            
            for sample in self.online_buffer:
                feature_vec = self._engineer_features(sample['features'])
                X_batch.append(feature_vec)
                
                label = sample['label']
                if label == 'SELL':
                    y_batch.append(0)
                elif label == 'HOLD':
                    y_batch.append(1)
                else:  # BUY
                    y_batch.append(2)
            
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            
            # Normalizar com o scaler existente
            if hasattr(self.scaler, 'scale_'):
                X_batch = self.scaler.transform(X_batch)
            else:
                # Primeira vez, fazer fit
                self.scaler.fit(X_batch)
                X_batch = self.scaler.transform(X_batch)
            
            # Partial fit nos modelos que suportam
            for model_name in ['sgd_online', 'passive_aggressive']:
                if model_name in self.models:
                    model = self.models[model_name]
                    
                    # Verificar se o modelo já foi inicializado
                    if not hasattr(model, 'classes_'):
                        # Primeira vez, precisa definir as classes
                        model.partial_fit(X_batch, y_batch, classes=[0, 1, 2])
                    else:
                        # Partial fit normal
                        model.partial_fit(X_batch, y_batch)
            
            # Para Random Forest com warm_start, adicionar mais árvores
            if 'random_forest' in self.models and self.models['random_forest'].warm_start:
                rf = self.models['random_forest']
                if hasattr(rf, 'estimators_'):
                    # Aumentar número de estimadores gradualmente
                    rf.n_estimators = min(rf.n_estimators + 10, 200)
                    rf.fit(X_batch, y_batch)
            
            # Limpar buffer após processamento
            self.online_buffer.clear()
            
        except Exception as e:
            logger.error(f"Erro no partial fit: {e}")
    
    def update_with_market_feedback(self, prediction_id: int, actual_price_change: float,
                                  time_elapsed: int = 300):
        """
        Atualiza modelo com feedback do mercado em tempo real
        
        Args:
            prediction_id: ID da predição
            actual_price_change: Mudança real de preço (%)
            time_elapsed: Tempo decorrido em segundos
        """
        if len(self.prediction_history) == 0:
            return
        
        # Encontrar predição
        prediction = None
        for p in self.prediction_history:
            if p.get('id') == prediction_id:
                prediction = p
                break
        
        if not prediction:
            return
        
        # Determinar se a predição foi correta
        action = prediction['action']
        was_correct = False
        
        if action == 'BUY' and actual_price_change > 0.001:  # 0.1%
            was_correct = True
        elif action == 'SELL' and actual_price_change < -0.001:
            was_correct = True
        elif action == 'HOLD' and abs(actual_price_change) < 0.001:
            was_correct = True
        
        # Calcular reward baseado na magnitude
        reward = abs(actual_price_change) if was_correct else -abs(actual_price_change)
        
        # Adicionar como amostra de treinamento
        self.add_training_sample(
            features=prediction['features'],
            action=action,
            reward=reward,
            was_correct=was_correct
        )
        
        # Log de feedback
        logger.debug(f"📈 Feedback: {action} -> {actual_price_change:.2%} "
                    f"({'✓' if was_correct else '✗'})")
    
    def detect_market_regime(self, features: Dict) -> str:
        """
        Detecta o regime de mercado atual baseado nas features
        
        Returns:
            'trending', 'ranging', 'volatile'
        """
        volatility = features.get('volatility', 0.01)
        momentum = features.get('momentum', 0)
        
        # Momentum multi-período se disponível
        if 'momentum_5' in features and 'momentum_10' in features:
            mom5 = features['momentum_5']
            mom10 = features['momentum_10']
            mom20 = features.get('momentum_20', mom10)
            
            # Trending: momentum consistente em múltiplos períodos
            if (abs(mom5) > 0.02 and abs(mom10) > 0.015 and abs(mom20) > 0.01 and
                np.sign(mom5) == np.sign(mom10) == np.sign(mom20)):
                return 'trending'
        
        # Alta volatilidade
        if volatility > 0.03:
            return 'volatile'
        
        # RSI em zona neutra e baixo momentum = ranging
        rsi = features.get('rsi', 50)
        if 40 < rsi < 60 and abs(momentum) < 0.01:
            return 'ranging'
        
        # CMF pode indicar tendência
        if 'cmf' in features:
            cmf = features['cmf']
            if abs(cmf) > 0.3:
                return 'trending'
        
        # Default baseado em volatilidade
        if volatility < 0.015:
            return 'ranging'
        else:
            return 'volatile'
    
    def adapt_to_regime_change(self, market_regime: str):
        """
        Adapta o modelo a mudanças de regime de mercado
        
        Args:
            market_regime: 'trending', 'ranging', 'volatile'
        """
        logger.info(f"🔄 Adaptando para regime: {market_regime}")
        
        # Ajustar hiperparâmetros baseado no regime
        if market_regime == 'trending':
            # Dar mais peso para momentum
            if hasattr(self.simple_predictor, 'feature_weights'):
                self.simple_predictor.feature_weights[1] *= 1.2  # Momentum
                self.simple_predictor.feature_weights /= self.simple_predictor.feature_weights.sum()
            
            # Ajustar thresholds para ser mais agressivo
            self.simple_predictor.threshold_buy *= 0.9
            self.simple_predictor.threshold_sell *= 0.9
            
        elif market_regime == 'ranging':
            # Dar mais peso para RSI e reversão
            if hasattr(self.simple_predictor, 'feature_weights'):
                self.simple_predictor.feature_weights[0] *= 1.2  # RSI
                self.simple_predictor.feature_weights /= self.simple_predictor.feature_weights.sum()
            
            # Ajustar thresholds para ser mais conservador
            self.simple_predictor.threshold_buy *= 1.1
            self.simple_predictor.threshold_sell *= 1.1
            
        elif market_regime == 'volatile':
            # Dar mais peso para volatilidade e spread
            if hasattr(self.simple_predictor, 'feature_weights'):
                self.simple_predictor.feature_weights[4] *= 1.3  # Volatility
                self.simple_predictor.feature_weights[3] *= 1.2  # Spread
                self.simple_predictor.feature_weights /= self.simple_predictor.feature_weights.sum()
            
            # Ser muito conservador
            self.simple_predictor.threshold_buy = 0.5
            self.simple_predictor.threshold_sell = -0.5
        
        # Forçar retreinamento com foco no regime atual
        if len(self.training_data) > self.min_training_samples:
            self._online_retrain()
    
    def _record_prediction(self, features: Dict, action: str, 
                         confidence: float, proba: np.ndarray):
        """Registra predição no histórico"""
        record = {
            'timestamp': time.time(),
            'features': features.copy(),
            'action': action,
            'confidence': confidence,
            'probabilities': proba.tolist() if isinstance(proba, np.ndarray) else proba,
            'model_type': self.model_type
        }
        
        self.prediction_history.append(record)
    
    def _get_cache_key(self, features: Dict) -> str:
        """Gera chave de cache para features"""
        # Usar apenas features principais para a chave
        key_features = ['rsi', 'momentum', 'volume_ratio', 'spread_bps', 'volatility']
        values = [round(features.get(f, 0), 4) for f in key_features]
        return f"{'-'.join(map(str, values))}"
    
    def get_feature_importance_advanced(self) -> Dict:
        """Retorna importância das features para modelos avançados"""
        if self.active_model is None:
            return self.simple_predictor.get_feature_importance()
        
        feature_names = [
            # Features básicas
            'rsi_norm', 'momentum', 'volume_log', 'spread_inv', 'volatility_norm',
            
            # VWAP features
            'vwap_deviation', 'above_vwap', 'vwap_distance',
            
            # CMF features  
            'cmf_raw', 'cmf_strong_buy', 'cmf_strong_sell', 'cmf_squared',
            
            # Momentum composto
            'compound_momentum', 'momentum_acceleration', 'trend_alignment', 'log_momentum',
            
            # Volatilidade implícita
            'implied_vol', 'vol_premium', 'high_iv', 'iv_rv_ratio',
            
            # RSI avançado
            'stoch_rsi_or_oversold', 'stoch_rsi_or_overbought', 'neutral_or_divergence',
            
            # Volume avançado
            'obv_or_spike', 'obv_log_or_low_vol',
            
            # Market microstructure
            'micro_spread_log', 'below_avg_spread',
            
            # Order flow
            'order_flow_imb', 'ofi_squared', 'ofi_extreme',
            
            # Volatility regimes
            'garch_or_low_vol', 'vol_compression_or_high', 'vol_ratio_or_sqrt',
            
            # Interações
            'rsi_momentum', 'volume_volatility', 'spread_volatility',
            'vwap_cmf_interaction', 'vwap_momentum',
            
            # Orderbook
            'orderbook_imb', 'orderbook_imb_sq', 'orderbook_buy_extreme', 'orderbook_sell_extreme',
            
            # Tempo
            'hour_sin', 'hour_cos', 'trading_hours',
            
            # Tendências
            'sma_cross', 'ema_cross',
            
            # Features históricas
            'rsi_ma', 'rsi_volatility', 'volume_trend', 'volume_acceleration',
            'momentum_ma', 'momentum_volatility'
        ]
        
        # Truncar se necessário
        n_features = len(self._engineer_features({'rsi': 50, 'momentum': 0, 
                                                 'volume_ratio': 1, 'spread_bps': 10, 
                                                 'volatility': 0.01}))
        feature_names = feature_names[:n_features]
        
        # Preencher nomes faltantes
        while len(feature_names) < n_features:
            feature_names.append(f'feature_{len(feature_names)}')
        
        importance_dict = {}
        
        try:
            # Para modelos baseados em árvore
            if hasattr(self.active_model, 'feature_importances_'):
                importances = self.active_model.feature_importances_
                
                # Normalizar
                importances = importances / importances.sum()
                
                # Criar dicionário
                for i, (name, imp) in enumerate(zip(feature_names, importances)):
                    if imp > 0.001:  # Apenas features relevantes
                        importance_dict[name] = float(imp)
                
                # Ordenar por importância
                importance_dict = dict(sorted(importance_dict.items(), 
                                           key=lambda x: x[1], reverse=True))
                
                # Top 10 features
                logger.info("🎯 Top 10 features mais importantes:")
                for i, (feat, imp) in enumerate(list(importance_dict.items())[:10]):
                    logger.info(f"  {i+1}. {feat}: {imp:.3f}")
                    
            # Para modelos lineares
            elif hasattr(self.active_model, 'coef_'):
                coef = self.active_model.coef_
                if len(coef.shape) > 1:
                    # Multi-class, usar média absoluta
                    coef = np.abs(coef).mean(axis=0)
                else:
                    coef = np.abs(coef)
                
                # Normalizar
                coef = coef / coef.sum()
                
                for name, imp in zip(feature_names, coef):
                    if imp > 0.001:
                        importance_dict[name] = float(imp)
                        
                importance_dict = dict(sorted(importance_dict.items(), 
                                           key=lambda x: x[1], reverse=True))
                
        except Exception as e:
            logger.error(f"Erro calculando importância de features: {e}")
            # Fallback para importância do modelo simples
            return self.simple_predictor.get_feature_importance()
        
        return importance_dict
    
    def get_model_stats(self) -> Dict:
        """Retorna estatísticas dos modelos"""
        stats = {
            'active_model': self.model_type,
            'models_available': list(self.models.keys()),
            'training_samples': len(self.training_data),
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'model_performance': self.model_performance,
            'cache_size': len(self.prediction_cache),
            'online_learning': {
                'enabled': self.partial_fit_enabled,
                'batch_size': self.online_batch_size,
                'buffer_size': len(self.online_buffer),
                'candles_since_retrain': self.candles_since_retrain,
                'retrain_interval': self.retrain_every_n_candles
            }
        }
        
        # Feature importance
        if self.active_model is not None:
            stats['feature_importance'] = self.get_feature_importance_advanced()
        
        # Estatísticas de predições
        if self.prediction_history:
            recent = list(self.prediction_history)[-100:]
            actions = [p['action'] for p in recent]
            confidences = [p['confidence'] for p in recent]
            
            stats['prediction_stats'] = {
                'total': len(self.prediction_history),
                'recent_distribution': {
                    'BUY': actions.count('BUY'),
                    'SELL': actions.count('SELL'),
                    'HOLD': actions.count('HOLD')
                },
                'avg_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences)
            }
        
        return stats
    
    def _save_model(self):
        """Salva modelo treinado em disco"""
        try:
            model_dir = Path('models')
            model_dir.mkdir(exist_ok=True)
            
            # Salvar modelo
            model_path = model_dir / f'{self.model_type}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.active_model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'performance': self.model_performance,
                    'timestamp': datetime.now()
                }, f)
            
            logger.info(f"💾 Modelo salvo: {model_path}")
            
        except Exception as e:
            logger.error(f"Erro salvando modelo: {e}")
    
    def load_model(self, model_path: str = None):
        """Carrega modelo do disco"""
        try:
            if model_path is None:
                model_path = Path('models') / f'{self.model_type}_model.pkl'
            
            if not Path(model_path).exists():
                logger.warning(f"Modelo não encontrado: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.active_model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data.get('feature_names', [])
            self.model_performance = data.get('performance', {})
            
            logger.info(f"✅ Modelo carregado: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro carregando modelo: {e}")
            return False


class SimplifiedMLPredictor:
    """Predictor simples como fallback (mantido do código original)"""
    
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
        
        logger.info("🤖 ML Predictor simples inicializado")
    
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


# Exemplo de uso do sistema ML avançado
if __name__ == "__main__":
    # Configuração para ambiente de produção
    config = {
        'retrain_hours': 6,  # Retreinar a cada 6 horas
        'min_accuracy': 0.55,
        'use_ensemble': True,
        'online_batch_size': 50,
        'retrain_candles': 100
    }
    
    # Inicializar predictor
    ml_predictor = AdvancedMLPredictor(model_type='auto', config=config)
    
    # Exemplo de features com indicadores avançados
    features = {
        # Básicas
        'rsi': 45,
        'momentum': 0.02,
        'volume_ratio': 1.5,
        'spread_bps': 15,
        'volatility': 0.015,
        'price': 50000,
        'volume': 1000000,
        
        # Avançadas
        'vwap': 49800,
        'cmf': 0.15,
        'momentum_5': 0.025,
        'momentum_10': 0.018,
        'momentum_20': 0.012,
        'implied_volatility': 0.018,
        'realized_volatility': 0.015,
        'stoch_rsi': 35,
        'obv': 0.25,
        'orderbook_imbalance': 0.65,
        'hour': 14
    }
    
    # Fazer predição
    action, confidence = ml_predictor.predict(features)
    print(f"Predição: {action} (confiança: {confidence:.2%})")
    
    # Simular feedback do mercado após 5 minutos
    actual_price_change = 0.002  # +0.2%
    ml_predictor.update_with_market_feedback(
        prediction_id=0, 
        actual_price_change=actual_price_change,
        time_elapsed=300
    )
    
    # Adicionar amostra de treinamento
    ml_predictor.add_training_sample(features, action='BUY', reward=0.002, was_correct=True)
    
    # Verificar estatísticas
    stats = ml_predictor.get_model_stats()
    print(f"\nEstatísticas do modelo:")
    print(f"- Modelo ativo: {stats['active_model']}")
    print(f"- Amostras de treinamento: {stats['training_samples']}")
    print(f"- Aprendizado online: {stats['online_learning']}")
    
    # Adaptar a mudança de regime
    ml_predictor.adapt_to_regime_change('volatile')
    
    # Ver importância das features
    importance = ml_predictor.get_feature_importance_advanced()
    print(f"\nTop 5 features mais importantes:")
    for i, (feat, imp) in enumerate(list(importance.items())[:5]):
        print(f"{i+1}. {feat}: {imp:.3f}")