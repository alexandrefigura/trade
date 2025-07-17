#!/usr/bin/env python3
"""
Script para criar os módulos faltantes do Sistema de Trading v5.2
Execute após extrair o ZIP principal para adicionar os módulos restantes
"""

import os
import zipfile
from datetime import datetime

# Módulos faltantes
MISSING_MODULES = {
    "trade_system/websocket_manager.py": '''"""
WebSocket Manager ultra-rápido para dados em tempo real
"""
import time
import queue
import threading
import logging
import asyncio
import numpy as np
from typing import Dict, Optional
from binance import ThreadedWebsocketManager
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class UltraFastWebSocketManager:
    """WebSocket otimizado para máxima throughput com reconexão automática"""
    
    def __init__(self, config, cache):
        self.config = config
        self.cache = cache
        self.twm = None
        
        # Buffers NumPy pré-alocados
        self.price_buffer = np.zeros(config.price_buffer_size, dtype=np.float32)
        self.volume_buffer = np.zeros(config.price_buffer_size, dtype=np.float32)
        self.time_buffer = np.zeros(config.price_buffer_size, dtype=np.int64)
        self.buffer_index = 0
        self.buffer_filled = False
        
        # Orderbook otimizado
        self.orderbook_bids = np.zeros((20, 2), dtype=np.float32)
        self.orderbook_asks = np.zeros((20, 2), dtype=np.float32)
        
        # Métricas
        self.messages_processed = 0
        self.last_update_time = time.perf_counter()
        self.last_message_time = time.time()
        
        # Queues thread-safe
        self.high_priority_queue = queue.Queue(maxsize=100)
        self.normal_queue = queue.Queue(maxsize=1000)
        
        # Controle
        self.is_connected = False
        self.reconnect_count = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1
        
        # Threads
        self.monitor_thread = None
        self.processor_thread = None
        
        # Flags
        self._initialized = False
        self._start_lock = threading.Lock()
        self._running = True
        
        # Modo simulado para testes
        self.simulate_mode = False
        
        logger.info("📡 WebSocket Manager inicializado")
    
    def start_delayed(self):
        """Inicia o WebSocket de forma atrasada"""
        with self._start_lock:
            if not self._initialized and not self.simulate_mode:
                self._initialized = True
                threading.Thread(target=self._start_in_thread, daemon=True).start()
                logger.info("⏳ WebSocket agendado para iniciar...")
    
    def _start_in_thread(self):
        """Inicia em thread separada com delay"""
        time.sleep(2)
        self._start()
    
    def _start(self):
        """Inicia WebSocket com streams otimizados"""
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                logger.info("🔧 Novo event loop criado")
            
            self.twm = ThreadedWebsocketManager()
            self.twm.start()
            
            time.sleep(0.5)
            
            logger.info(f"🔌 Conectando aos streams do {self.config.symbol}...")
            
            # Streams essenciais
            self.twm.start_aggtrade_socket(
                callback=self._process_trade,
                symbol=self.config.symbol
            )
            
            self.twm.start_depth_socket(
                callback=self._process_orderbook,
                symbol=self.config.symbol,
                depth=20,
                interval=100
            )
            
            # Thread de processamento
            if not self.processor_thread or not self.processor_thread.is_alive():
                self.processor_thread = threading.Thread(
                    target=self._process_queue_loop,
                    daemon=True
                )
                self.processor_thread.start()
            
            # Thread de monitoramento
            if not self.monitor_thread or not self.monitor_thread.is_alive():
                self.monitor_thread = threading.Thread(
                    target=self._monitor_connection,
                    daemon=True
                )
                self.monitor_thread.start()
            
            self.is_connected = True
            self.reconnect_count = 0
            logger.info("🚀 WebSocket conectado com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar WebSocket: {e}")
            self._schedule_reconnect()
    
    def _monitor_connection(self):
        """Monitora conexão e reconecta se necessário"""
        while self._running:
            try:
                time.sleep(5)
                
                if self.is_connected and time.time() - self.last_message_time > 30:
                    logger.warning("⚠️ WebSocket sem mensagens há 30s")
                    self._reconnect()
                    
            except Exception as e:
                logger.error(f"Erro no monitor: {e}")
    
    def _reconnect(self):
        """Reconecta com backoff exponencial"""
        self.is_connected = False
        
        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.error("❌ Máximo de reconexões atingido")
            return
        
        self.reconnect_count += 1
        delay = self.reconnect_delay * (2 ** (self.reconnect_count - 1))
        delay = min(delay, 60)
        
        logger.info(f"🔄 Reconectando em {delay}s (tentativa {self.reconnect_count})")
        
        if self.twm:
            try:
                self.twm.stop()
            except:
                pass
        
        time.sleep(delay)
        self._start()
    
    def _schedule_reconnect(self):
        """Agenda reconexão assíncrona"""
        threading.Thread(
            target=self._reconnect,
            daemon=True
        ).start()
    
    def _process_trade(self, msg):
        """Processa trades com latência mínima"""
        try:
            self.last_message_time = time.time()
            
            price = float(msg['p'])
            volume = float(msg['q'])
            timestamp = msg['T']
            is_buyer_maker = msg['m']
            
            # Detectar trades grandes
            if volume * price > 50000:
                self.high_priority_queue.put((
                    'large_trade',
                    price,
                    volume,
                    is_buyer_maker,
                    timestamp
                ))
            
            # Atualizar buffer
            idx = self.buffer_index % self.config.price_buffer_size
            self.price_buffer[idx] = price
            self.volume_buffer[idx] = volume
            self.time_buffer[idx] = timestamp
            
            self.buffer_index += 1
            if self.buffer_index >= self.config.price_buffer_size:
                self.buffer_filled = True
            
            self.messages_processed += 1
            
            if self.messages_processed == 1:
                logger.info(f"✅ Primeiro trade: ${price:,.2f}")
            
        except Exception as e:
            logger.error(f"Erro processando trade: {e}")
    
    def _process_orderbook(self, msg):
        """Processa orderbook otimizado"""
        try:
            self.last_message_time = time.time()
            
            bids = msg['bids']
            asks = msg['asks']
            
            # Copiar para arrays
            for i in range(min(20, len(bids))):
                self.orderbook_bids[i, 0] = float(bids[i][0])
                self.orderbook_bids[i, 1] = float(bids[i][1])
            
            for i in range(min(20, len(asks))):
                self.orderbook_asks[i, 0] = float(asks[i][0])
                self.orderbook_asks[i, 1] = float(asks[i][1])
            
            # Calcular imbalance
            bid_volume = np.sum(self.orderbook_bids[:5, 1])
            ask_volume = np.sum(self.orderbook_asks[:5, 1])
            
            if bid_volume + ask_volume > 0:
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                
                self.cache.set('orderbook_imbalance', imbalance, ttl=1)
                
                if abs(imbalance) > 0.7:
                    self.high_priority_queue.put(('imbalance', imbalance))
                    
        except Exception as e:
            logger.error(f"Erro processando orderbook: {e}")
    
    def _process_queue_loop(self):
        """Loop de processamento dedicado"""
        while self._running:
            try:
                while not self.high_priority_queue.empty():
                    data = self.high_priority_queue.get_nowait()
                    self._handle_priority_event(data)
                
                time.sleep(0.001)
                
            except:
                pass
    
    def _handle_priority_event(self, data):
        """Processa eventos de alta prioridade"""
        event_type = data[0]
        
        if event_type == 'large_trade':
            _, price, volume, is_buyer, timestamp = data
            side = "COMPRA" if is_buyer else "VENDA"
            logger.info(f"🐋 TRADE GRANDE [{side}]: ${volume * price:,.0f} @ ${price:,.2f}")
            
        elif event_type == 'imbalance':
            _, imbalance = data
            direction = "COMPRA" if imbalance > 0 else "VENDA"
            logger.info(f"📊 IMBALANCE [{direction}]: {abs(imbalance)*100:.1f}%")
    
    def get_latest_data(self) -> Dict:
        """Retorna dados mais recentes"""
        if self.simulate_mode and self.buffer_filled:
            return self._get_simulated_data()
        
        if not self.buffer_filled and self.buffer_index < 100:
            return None
        
        end_idx = self.buffer_index if not self.buffer_filled else self.config.price_buffer_size
        start_idx = max(0, end_idx - 1000)
        
        return {
            'prices': self.price_buffer[start_idx:end_idx],
            'volumes': self.volume_buffer[start_idx:end_idx],
            'timestamps': self.time_buffer[start_idx:end_idx],
            'orderbook_bids': self.orderbook_bids.copy(),
            'orderbook_asks': self.orderbook_asks.copy(),
            'messages_per_second': self._calculate_mps()
        }
    
    def _get_simulated_data(self) -> Dict:
        """Retorna dados simulados para testes"""
        return {
            'prices': self.price_buffer[:1000],
            'volumes': self.volume_buffer[:1000],
            'timestamps': self.time_buffer[:1000],
            'orderbook_bids': self.orderbook_bids.copy(),
            'orderbook_asks': self.orderbook_asks.copy(),
            'messages_per_second': 100.0
        }
    
    def _calculate_mps(self) -> float:
        """Calcula mensagens por segundo"""
        now = time.perf_counter()
        elapsed = now - self.last_update_time
        
        if elapsed > 1.0:
            mps = self.messages_processed / elapsed
            self.messages_processed = 0
            self.last_update_time = now
            return mps
        
        return 0
    
    def stop(self):
        """Para o WebSocket de forma segura"""
        try:
            logger.info("🛑 Parando WebSocket...")
            self._running = False
            self.is_connected = False
            
            if self.twm:
                self.twm.stop()
                
            logger.info("✅ WebSocket parado")
        except Exception as e:
            logger.error(f"Erro ao parar: {e}")
    
    def get_connection_status(self) -> Dict:
        """Retorna status da conexão"""
        return {
            'connected': self.is_connected,
            'reconnect_count': self.reconnect_count,
            'buffer_filled': self.buffer_filled,
            'buffer_size': self.buffer_index,
            'last_message': datetime.fromtimestamp(self.last_message_time).strftime('%H:%M:%S') if self.last_message_time else 'Never',
            'messages_per_second': self._calculate_mps()
        }
    
    def enable_simulation_mode(self):
        """Ativa modo simulado para testes"""
        self.simulate_mode = True
        self._fill_with_simulated_data()
        self.is_connected = True
        logger.info("🎮 Modo simulado ativado")
    
    def _fill_with_simulated_data(self):
        """Preenche buffers com dados simulados"""
        base_price = 40000.0
        for i in range(1000):
            self.price_buffer[i] = base_price + np.random.randn() * 100
            self.volume_buffer[i] = np.random.rand() * 10
            self.time_buffer[i] = int(time.time() * 1000) + i * 1000
        
        self.buffer_index = 1000
        self.buffer_filled = True
        
        # Orderbook simulado
        for i in range(20):
            self.orderbook_bids[i, 0] = base_price - (i * 10)
            self.orderbook_bids[i, 1] = np.random.rand() * 5
            self.orderbook_asks[i, 0] = base_price + (i * 10)
            self.orderbook_asks[i, 1] = np.random.rand() * 5
''',

    "trade_system/analysis/orderbook.py": '''"""
Análise paralela de orderbook para detecção de imbalances
"""
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
        cached_imbalance = cache.get('orderbook_imbalance')
        if cached_imbalance is not None:
            return self._quick_decision(cached_imbalance)
        
        # Validar dados
        if len(bids) == 0 or len(asks) == 0:
            return 'HOLD', 0.5, {'reason': 'Orderbook vazio'}
        
        # Cálculos paralelos
        future_bid = self.executor.submit(self._analyze_bid_side, bids)
        future_ask = self.executor.submit(self._analyze_ask_side, asks)
        
        bid_strength = future_bid.result()
        ask_strength = future_ask.result()
        
        # Calcular imbalance
        total = bid_strength + ask_strength
        if total > 0:
            imbalance = (bid_strength - ask_strength) / total
        else:
            imbalance = 0
        
        # Registrar histórico
        self.imbalance_history.append(imbalance)
        
        # Analisar microestrutura
        spread = asks[0, 0] - bids[0, 0] if asks[0, 0] > 0 and bids[0, 0] > 0 else 0
        spread_bps = (spread / bids[0, 0]) * 10000 if bids[0, 0] > 0 else 0
        
        # Profundidade do book
        bid_depth = np.sum(bids[:5, 1])
        ask_depth = np.sum(asks[:5, 1])
        depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 1
        
        # Detectar ordens grandes
        self._detect_large_orders(bids, asks)
        
        # Análise de pressão
        pressure = self._calculate_pressure(bids, asks)
        self.pressure_history.append(pressure)
        
        # Decisão baseada em múltiplos fatores
        action, confidence = self._make_decision(
            imbalance, spread_bps, depth_ratio, pressure
        )
        
        # Detalhes para log
        details = {
            'imbalance': float(imbalance),
            'spread_bps': float(spread_bps),
            'bid_strength': float(bid_strength),
            'ask_strength': float(ask_strength),
            'bid_depth': float(bid_depth),
            'ask_depth': float(ask_depth),
            'depth_ratio': float(depth_ratio),
            'best_bid': float(bids[0, 0]) if bids[0, 0] > 0 else 0,
            'best_ask': float(asks[0, 0]) if asks[0, 0] > 0 else 0,
            'pressure': float(pressure),
            'large_orders': len(self.large_orders)
        }
        
        # Cache resultado
        cache.set('orderbook_analysis', {
            'action': action,
            'confidence': confidence,
            'details': details
        }, ttl=2)
        
        return action, confidence, details
    
    def _analyze_bid_side(self, bids: np.ndarray) -> float:
        """Analisa força do lado comprador"""
        if len(bids) == 0:
            return 0.0
        
        # Peso por proximidade do preço
        weights = np.exp(-np.arange(len(bids)) * 0.1)
        weighted_volume = np.sum(bids[:, 1] * weights[:len(bids)])
        
        # Considerar concentração de ordens
        top_5_volume = np.sum(bids[:5, 1]) if len(bids) >= 5 else np.sum(bids[:, 1])
        total_volume = np.sum(bids[:, 1])
        concentration = top_5_volume / total_volume if total_volume > 0 else 0
        
        return weighted_volume * (1 + concentration * 0.2)
    
    def _analyze_ask_side(self, asks: np.ndarray) -> float:
        """Analisa força do lado vendedor"""
        if len(asks) == 0:
            return 0.0
        
        weights = np.exp(-np.arange(len(asks)) * 0.1)
        weighted_volume = np.sum(asks[:, 1] * weights[:len(asks)])
        
        top_5_volume = np.sum(asks[:5, 1]) if len(asks) >= 5 else np.sum(asks[:, 1])
        total_volume = np.sum(asks[:, 1])
        concentration = top_5_volume / total_volume if total_volume > 0 else 0
        
        return weighted_volume * (1 + concentration * 0.2)
    
    def _calculate_pressure(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """Calcula pressão de compra/venda"""
        if len(bids) < 10 or len(asks) < 10:
            return 0.0
        
        # Pressão nos primeiros 10 níveis
        bid_pressure = np.sum(bids[:10, 1] * np.exp(-np.arange(10) * 0.05))
        ask_pressure = np.sum(asks[:10, 1] * np.exp(-np.arange(10) * 0.05))
        
        total_pressure = bid_pressure + ask_pressure
        if total_pressure > 0:
            return (bid_pressure - ask_pressure) / total_pressure
        
        return 0.0
    
    def _detect_large_orders(self, bids: np.ndarray, asks: np.ndarray):
        """Detecta ordens grandes no book"""
        # Calcular tamanho médio
        all_volumes = np.concatenate([bids[:, 1], asks[:, 1]])
        if len(all_volumes) == 0:
            return
        
        avg_volume = np.mean(all_volumes)
        std_volume = np.std(all_volumes)
        threshold = avg_volume + 2 * std_volume
        
        # Detectar ordens grandes
        for i, (price, volume) in enumerate(bids):
            if volume > threshold:
                self.large_orders.append({
                    'side': 'BID',
                    'price': price,
                    'volume': volume,
                    'level': i,
                    'timestamp': time.time()
                })
        
        for i, (price, volume) in enumerate(asks):
            if volume > threshold:
                self.large_orders.append({
                    'side': 'ASK',
                    'price': price,
                    'volume': volume,
                    'level': i,
                    'timestamp': time.time()
                })
    
    def _make_decision(
        self,
        imbalance: float,
        spread_bps: float,
        depth_ratio: float,
        pressure: float
    ) -> Tuple[str, float]:
        """Toma decisão baseada em múltiplos fatores"""
        # Pontuação composta
        score = 0.0
        confidence_factors = []
        
        # Fator 1: Imbalance
        if abs(imbalance) > 0.6:
            score += imbalance * 0.4
            confidence_factors.append(min(abs(imbalance), 0.9))
        
        # Fator 2: Spread (menor é melhor)
        if spread_bps < 10:
            score += np.sign(imbalance) * 0.2
            confidence_factors.append(0.8)
        elif spread_bps > 20:
            score *= 0.5  # Reduz confiança com spread alto
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.7)
        
        # Fator 3: Depth ratio
        if depth_ratio > 1.5:
            score += 0.2
            confidence_factors.append(0.7)
        elif depth_ratio < 0.67:
            score -= 0.2
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Fator 4: Pressão
        score += pressure * 0.3
        if abs(pressure) > 0.3:
            confidence_factors.append(min(abs(pressure) + 0.5, 0.9))
        else:
            confidence_factors.append(0.5)
        
        # Decisão final
        if score > 0.4:
            action = 'BUY'
        elif score < -0.4:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Confiança média ponderada
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Ajustar confiança para HOLD
        if action == 'HOLD':
            confidence *= 0.7
        
        return action, confidence
    
    def _quick_decision(self, imbalance: float) -> Tuple[str, float, Dict]:
        """Decisão rápida baseada em cache"""
        if imbalance > 0.6:
            return 'BUY', 0.7 + imbalance * 0.3, {'cached': True, 'imbalance': imbalance}
        elif imbalance < -0.6:
            return 'SELL', 0.7 + abs(imbalance) * 0.3, {'cached': True, 'imbalance': imbalance}
        else:
            return 'HOLD', 0.5, {'cached': True, 'imbalance': imbalance}
    
    def get_market_pressure(self) -> str:
        """Retorna pressão geral do mercado"""
        if len(self.imbalance_history) < 10:
            return "Neutro"
        
        avg_imbalance = np.mean(list(self.imbalance_history)[-20:])
        
        if avg_imbalance > 0.3:
            return "Pressão Compradora"
        elif avg_imbalance < -0.3:
            return "Pressão Vendedora"
        else:
            return "Neutro"
    
    def get_orderbook_stats(self) -> Dict:
        """Retorna estatísticas do orderbook"""
        if not self.imbalance_history:
            return {}
        
        recent_imbalances = list(self.imbalance_history)[-50:]
        recent_pressures = list(self.pressure_history)[-50:] if self.pressure_history else []
        
        stats = {
            'avg_imbalance': np.mean(recent_imbalances),
            'std_imbalance': np.std(recent_imbalances),
            'trend': 'UP' if recent_imbalances[-1] > recent_imbalances[0] else 'DOWN',
            'large_orders_count': len(self.large_orders),
            'market_pressure': self.get_market_pressure()
        }
        
        if recent_pressures:
            stats['avg_pressure'] = np.mean(recent_pressures)
            stats['pressure_trend'] = 'INCREASING' if recent_pressures[-1] > recent_pressures[0] else 'DECREASING'
        
        return stats


# Importações necessárias
import time
''',

    "trade_system/analysis/ml.py": '''"""
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
''',

    "trade_system/risk.py": '''"""
Gestão de risco ultra-rápida e validação de condições de mercado
"""
import time
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class UltraFastRiskManager:
    """Gestão de risco com cálculos otimizados"""
    
    def __init__(self, config):
        self.config = config
        self.current_balance = 10000.0  # Balance inicial padrão
        self.initial_balance = 10000.0
        self.daily_pnl = 0.0
        self.max_daily_loss = config.max_daily_loss
        self.position_info = None
        
        # Histórico em array NumPy
        self.pnl_history = np.zeros(1000, dtype=np.float32)
        self.pnl_index = 0
        
        # Métricas
        self.total_fees_paid = 0.0
        self.peak_balance = 10000.0
        self.drawdown_start = None
        self.max_drawdown = 0.0
        self.daily_trades = 0
        self.last_reset_day = datetime.now().date()
        
        # Limites de risco
        self.max_position_value = None
        self.max_positions = 1
        self.current_positions = 0
        
        logger.info(f"💰 Risk Manager inicializado - Balance: ${self.current_balance:,.2f}")
    
    def calculate_position_size(
        self,
        confidence: float,
        volatility: float,
        current_price: Optional[float] = None
    ) -> float:
        """
        Cálculo rápido do tamanho da posição
        
        Args:
            confidence: Confiança do sinal (0-1)
            volatility: Volatilidade atual do mercado
            current_price: Preço atual (opcional)
            
        Returns:
            Valor da posição em USD
        """
        # Reset diário se necessário
        self._check_daily_reset()
        
        # Verificar limites de risco
        if not self._check_risk_limits():
            return 0.0
        
        # Kelly Criterion simplificado
        win_rate = 0.55  # Assumir 55% de win rate
        avg_win_loss_ratio = 1.5  # Assumir 1.5:1
        
        kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Limitar a 25%
        
        # Ajustar por confiança
        position_pct = kelly_fraction * confidence * self.config.max_position_pct
        
        # Ajustar por volatilidade
        volatility_factor = self._calculate_volatility_factor(volatility)
        position_pct *= volatility_factor
        
        # Calcular valor da posição
        position_value = self.current_balance * position_pct
        
        # Aplicar limites
        position_value = self._apply_position_limits(position_value)
        
        # Log detalhado
        if current_price and position_value > 0:
            quantity = position_value / current_price
            logger.debug(
                f"📊 Posição calculada: ${position_value:.2f} "
                f"({position_pct*100:.1f}%) = {quantity:.6f} unidades @ ${current_price:.2f}"
            )
        
        return position_value
    
    def _check_risk_limits(self) -> bool:
        """Verifica se pode abrir nova posição"""
        # Stop loss diário
        if self.daily_pnl < -self.max_daily_loss * self.current_balance:
            logger.warning(f"🛑 Stop loss diário atingido: ${self.daily_pnl:.2f}")
            return False
        
        # Número máximo de posições
        if self.current_positions >= self.max_positions:
            logger.debug("Máximo de posições atingido")
            return False
        
        # Margem mínima
        min_balance = 100  # USD
        if self.current_balance < min_balance:
            logger.warning(f"Balance insuficiente: ${self.current_balance:.2f}")
            return False
        
        return True
    
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """Calcula fator de ajuste baseado na volatilidade"""
        if volatility < 0.01:  # Baixa volatilidade
            return 1.2
        elif volatility < 0.02:  # Normal
            return 1.0
        elif volatility < 0.03:  # Alta
            return 0.5
        else:  # Muito alta
            return 0.3
    
    def _apply_position_limits(self, position_value: float) -> float:
        """Aplica limites ao tamanho da posição"""
        # Limite mínimo
        min_position = 50.0  # USD
        if position_value < min_position:
            return 0.0
        
        # Limite máximo absoluto
        if self.max_position_value:
            position_value = min(position_value, self.max_position_value)
        
        # Limite por percentual do balance
        max_allowed = self.current_balance * 0.1  # Máximo 10% por posição
        position_value = min(position_value, max_allowed)
        
        return position_value
    
    def should_close_position(
        self,
        current_price: float,
        entry_price: float,
        side: str = 'BUY'
    ) -> Tuple[bool, str]:
        """
        Verificação rápida se deve fechar posição
        
        Returns:
            Tupla (should_close, reason)
        """
        if self.position_info is None:
            return False, ""
        
        # Calcular P&L
        if side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SELL
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Atualizar highest PnL
        if pnl_pct > self.position_info.get('highest_pnl', 0):
            self.position_info['highest_pnl'] = pnl_pct
        
        # Stop loss
        stop_loss_pct = self.position_info.get('stop_loss_pct', 0.01)
        if pnl_pct < -stop_loss_pct:
            return True, "Stop Loss"
        
        # Take profit
        take_profit_pct = self.position_info.get('take_profit_pct', 0.015)
        if pnl_pct > take_profit_pct:
            return True, "Take Profit"
        
        # Trailing stop
        if self.position_info.get('highest_pnl', 0) > 0.005:
            trailing_pct = 0.7  # Manter 70% do lucro máximo
            if pnl_pct < self.position_info['highest_pnl'] * trailing_pct:
                return True, "Trailing Stop"
        
        # Time-based stop
        position_duration = time.time() - self.position_info.get('entry_time', time.time())
        max_duration = self.position_info.get('max_duration', 3600)  # 1 hora padrão
        
        if position_duration > max_duration and abs(pnl_pct) < 0.002:
            return True, "Time Stop"
        
        return False, ""
    
    def update_pnl(self, pnl: float, fees: float = 0):
        """Atualiza P&L com array circular"""
        self.daily_pnl += pnl
        self.current_balance += pnl
        self.total_fees_paid += fees
        
        # Atualizar histórico
        idx = self.pnl_index % 1000
        self.pnl_history[idx] = pnl
        self.pnl_index += 1
        
        # Atualizar peak e drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.drawdown_start = None
        else:
            if self.drawdown_start is None:
                self.drawdown_start = datetime.now()
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _check_daily_reset(self):
        """Verifica e reseta métricas diárias"""
        current_day = datetime.now().date()
        if current_day > self.last_reset_day:
            logger.info(f"📅 Novo dia - Reset de métricas diárias")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_day = current_day
    
    def get_risk_metrics(self) -> Dict:
        """Retorna métricas de risco atuais"""
        current_drawdown = 0
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        return {
            'current_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': current_drawdown,
            'total_fees': self.total_fees_paid,
            'daily_trades': self.daily_trades,
            'can_trade': self._check_risk_limits(),
            'risk_level': self._calculate_risk_level()
        }
    
    def _calculate_risk_level(self) -> str:
        """Calcula nível de risco atual"""
        metrics = {
            'daily_loss_pct': abs(self.daily_pnl / self.current_balance) if self.current_balance > 0 else 0,
            'drawdown': (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0,
            'balance_pct': self.current_balance / self.initial_balance
        }
        
        # Classificar risco
        if metrics['daily_loss_pct'] > 0.015 or metrics['drawdown'] > 0.1:
            return "ALTO"
        elif metrics['daily_loss_pct'] > 0.01 or metrics['drawdown'] > 0.05:
            return "MÉDIO"
        else:
            return "BAIXO"
    
    def set_position_info(self, position: Dict):
        """Define informações da posição atual"""
        self.position_info = position
        self.current_positions = 1 if position else 0
        self.daily_trades += 1
    
    def clear_position(self):
        """Limpa informações da posição"""
        self.position_info = None
        self.current_positions = 0


class MarketConditionValidator:
    """Valida se as condições de mercado são seguras para operar"""
    
    def __init__(self, config):
        self.config = config
        self.last_validation = time.time()
        self.validation_interval = 60  # Validar a cada minuto
        self.is_market_safe = True
        self.unsafe_reasons = []
        self.market_score = 100  # Score de 0-100
        
        # Histórico de condições
        self.volatility_history = []
        self.spread_history = []
        self.volume_history = []
        
        logger.info("🛡️ Market Validator inicializado")
    
    async def validate_market_conditions(
        self,
        market_data: Dict,
        client = None
    ) -> Tuple[bool, List[str]]:
        """
        Valida condições de mercado
        
        Args:
            market_data: Dados atuais do mercado
            client: Cliente da exchange (opcional)
            
        Returns:
            Tupla (is_safe, reasons)
        """
        reasons = []
        score = 100
        
        # Em modo debug, sempre retorna mercado seguro
        if self.config.debug_mode:
            return True, []
        
        # 1. Verificar volatilidade
        volatility = self._check_volatility(market_data)
        if volatility:
            vol_pct = volatility * 100
            self.volatility_history.append(volatility)
            
            if volatility > self.config.max_volatility:
                reasons.append(f"Volatilidade muito alta: {vol_pct:.2f}%")
                score -= 30
            elif volatility > self.config.max_volatility * 0.8:
                reasons.append(f"Volatilidade elevada: {vol_pct:.2f}%")
                score -= 15
        
        # 2. Verificar spread
        spread_bps = self._check_spread(market_data)
        if spread_bps:
            self.spread_history.append(spread_bps)
            
            if spread_bps > self.config.max_spread_bps:
                reasons.append(f"Spread muito alto: {spread_bps:.1f} bps")
                score -= 25
            elif spread_bps > self.config.max_spread_bps * 0.8:
                reasons.append(f"Spread elevado: {spread_bps:.1f} bps")
                score -= 10
        
        # 3. Verificar volume (async)
        if client and time.time() - self.last_validation > self.validation_interval:
            volume_ok, volume_reason = await self._check_volume_async(client)
            if not volume_ok:
                reasons.append(volume_reason)
                score -= 20
            self.last_validation = time.time()
        
        # 4. Verificar liquidez do orderbook
        liquidity_ok, liquidity_reason = self._check_liquidity(market_data)
        if not liquidity_ok:
            reasons.append(liquidity_reason)
            score -= 15
        
        # 5. Verificar horário
        time_ok, time_reason = self._check_trading_time()
        if not time_ok:
            reasons.append(time_reason)
            score -= 10
        
        # 6. Verificar condições extremas
        if self._detect_flash_crash(market_data):
            reasons.append("⚠️ FLASH CRASH DETECTADO!")
            score = 0
        
        # Atualizar estado
        self.market_score = max(0, score)
        self.is_market_safe = score >= 50  # Mínimo 50 pontos
        self.unsafe_reasons = reasons
        
        return self.is_market_safe, reasons
    
    def _check_volatility(self, market_data: Dict) -> Optional[float]:
        """Calcula volatilidade atual"""
        if 'prices' not in market_data or len(market_data['prices']) < 100:
            return None
        
        prices = market_data['prices'][-100:]
        return np.std(prices) / np.mean(prices)
    
    def _check_spread(self, market_data: Dict) -> Optional[float]:
        """Calcula spread em basis points"""
        if 'orderbook_asks' not in market_data or 'orderbook_bids' not in market_data:
            return None
        
        asks = market_data['orderbook_asks']
        bids = market_data['orderbook_bids']
        
        if len(asks) > 0 and len(bids) > 0 and asks[0, 0] > 0 and bids[0, 0] > 0:
            spread = asks[0, 0] - bids[0, 0]
            return (spread / bids[0, 0]) * 10000
        
        return None
    
    async def _check_volume_async(self, client) -> Tuple[bool, str]:
        """Verifica volume 24h (assíncrono)"""
        try:
            ticker = await client.get_ticker(symbol=self.config.symbol)
            volume_24h = float(ticker['quoteVolume'])
            
            self.volume_history.append(volume_24h)
            
            if volume_24h < self.config.min_volume_24h:
                return False, f"Volume 24h baixo: ${volume_24h:,.0f}"
            
            return True, ""
            
        except Exception as e:
            logger.debug(f"Erro ao verificar volume: {e}")
            return True, ""  # Assumir OK se erro
    
    def _check_liquidity(self, market_data: Dict) -> Tuple[bool, str]:
        """Verifica liquidez do orderbook"""
        if 'orderbook_bids' not in market_data or 'orderbook_asks' not in market_data:
            return True, ""
        
        bids = market_data['orderbook_bids']
        asks = market_data['orderbook_asks']
        
        # Verificar profundidade
        if len(bids) < 5 or len(asks) < 5:
            return False, "Orderbook raso"
        
        # Verificar volume nos primeiros níveis
        bid_volume = np.sum(bids[:5, 1])
        ask_volume = np.sum(asks[:5, 1])
        
        min_volume = 10  # Mínimo em unidades base
        if bid_volume < min_volume or ask_volume < min_volume:
            return False, "Baixa liquidez no orderbook"
        
        return True, ""
    
    def _check_trading_time(self) -> Tuple[bool, str]:
        """Verifica horário de trading"""
        current_hour = datetime.now().hour
        
        # Evitar horários de baixa liquidez (UTC)
        if 2 <= current_hour <= 6:
            return False, "Horário de baixa liquidez (2-6 UTC)"
        
        # Domingo tem liquidez reduzida
        if datetime.now().weekday() == 6:
            return False, "Domingo - liquidez reduzida"
        
        return True, ""
    
    def _detect_flash_crash(self, market_data: Dict) -> bool:
        """Detecta movimentos extremos de preço"""
        if 'prices' not in market_data or len(market_data['prices']) < 50:
            return False
        
        prices = market_data['prices']
        
        # Verificar queda/subida súbita
        recent_prices = prices[-10:]
        older_prices = prices[-50:-40]
        
        if len(recent_prices) > 0 and len(older_prices) > 0:
            recent_avg = np.mean(recent_prices)
            older_avg = np.mean(older_prices)
            
            if older_avg > 0:
                change = abs(recent_avg - older_avg) / older_avg
                
                # Movimento > 5% em poucos minutos
                if change > 0.05:
                    return True
        
        return False
    
    def get_market_health(self) -> Dict:
        """Retorna saúde geral do mercado"""
        return {
            'score': self.market_score,
            'is_safe': self.is_market_safe,
            'status': self._get_market_status(),
            'reasons': self.unsafe_reasons,
            'metrics': {
                'avg_volatility': np.mean(self.volatility_history[-10:]) if self.volatility_history else 0,
                'avg_spread': np.mean(self.spread_history[-10:]) if self.spread_history else 0,
                'last_volume': self.volume_history[-1] if self.volume_history else 0
            }
        }
    
    def _get_market_status(self) -> str:
        """Retorna status textual do mercado"""
        if self.market_score >= 80:
            return "EXCELENTE"
        elif self.market_score >= 60:
            return "BOM"
        elif self.market_score >= 50:
            return "REGULAR"
        elif self.market_score >= 30:
            return "RUIM"
        else:
            return "PERIGOSO"
''',

    "trade_system/backtester.py": '''"""
Sistema de backtesting integrado para validação de estratégias
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from trade_system.logging_config import get_logger
from trade_system.analysis.technical import UltraFastTechnicalAnalysis
from trade_system.analysis.ml import SimplifiedMLPredictor
from trade_system.signals import OptimizedSignalConsolidator
from trade_system.utils import calculate_atr

logger = get_logger(__name__)


class IntegratedBacktester:
    """Sistema de backtesting para validação de estratégias"""
    
    def __init__(self, config):
        self.config = config
        self.results = []
        self.trades = []
        self.equity_curve = []
        
    async def backtest_strategy(
        self,
        historical_data: pd.DataFrame,
        initial_balance: float = 10000
    ) -> Dict:
        """
        Executa backtest com dados históricos
        
        Args:
            historical_data: DataFrame com OHLCV
            initial_balance: Capital inicial
            
        Returns:
            Dicionário com métricas do backtest
        """
        logger.info("🔄 Iniciando backtest...")
        
        # Validar dados
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in historical_data.columns:
                logger.error(f"Coluna {col} não encontrada nos dados")
                return {}
        
        # Preparar dados
        balance = initial_balance
        position = None
        trades = []
        equity_curve = []
        
        # Converter para arrays NumPy
        prices = historical_data['close'].values.astype(np.float32)
        volumes = historical_data['volume'].values.astype(np.float32)
        
        # Arrays adicionais se disponíveis
        highs = historical_data.get('high', prices).values.astype(np.float32)
        lows = historical_data.get('low', prices).values.astype(np.float32)
        
        # Calcular ATR
        atr_series = calculate_atr(highs, lows, prices, period=self.config.atr_period)
        
        # Componentes de análise
        technical_analyzer = UltraFastTechnicalAnalysis(self.config)
        ml_predictor = SimplifiedMLPredictor()
        consolidator = OptimizedSignalConsolidator()
        
        # Loop principal
        start_idx = max(200, self.config.atr_period)  # Mínimo para indicadores
        
        # Contadores
        total_signals = 0
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for i in range(start_idx, len(prices)):
            # Slice de dados
            price_slice = prices[:i+1]
            volume_slice = volumes[:i+1]
            
            # Análise técnica
            tech_action, tech_conf, tech_details = technical_analyzer.analyze(
                price_slice[-1000:],  # Últimos 1000 pontos
                volume_slice[-1000:]
            )
            
            # Features para ML
            features = self._extract_features(
                prices, volumes, i, tech_details
            )
            
            # Predição ML
            ml_action, ml_conf = ml_predictor.predict(features)
            
            # Consolidar sinais
            signals = [
                ('technical', tech_action, tech_conf),
                ('ml', ml_action, ml_conf)
            ]
            
            action, confidence = consolidator.consolidate(signals)
            
            # Contar sinais
            signal_counts[action] += 1
            if action != 'HOLD':
                total_signals += 1
            
            # Log periódico em modo debug
            if self.config.debug_mode and i % 500 == 0:
                logger.debug(f"Backtest progresso: {i}/{len(prices)} ({i/len(prices)*100:.1f}%)")
            
            # Gerenciar posições
            current_price = prices[i]
            
            # Entrada
            if position is None and action != 'HOLD' and confidence > self.config.min_confidence:
                # Calcular tamanho da posição
                position_size = self._calculate_position_size(
                    balance, confidence, features.get('volatility', 0.01)
                )
                
                if position_size > 0:
                    # ATR para stops
                    atr = atr_series[i] if i < len(atr_series) and not np.isnan(atr_series[i]) else None
                    
                    if atr and atr > 0:
                        # Calcular stops baseados em ATR
                        if action == 'BUY':
                            tp_price = current_price + (atr * self.config.tp_multiplier)
                            sl_price = current_price - (atr * self.config.sl_multiplier)
                        else:  # SELL
                            tp_price = current_price - (atr * self.config.tp_multiplier)
                            sl_price = current_price + (atr * self.config.sl_multiplier)
                    else:
                        # Fallback para percentuais fixos
                        if action == 'BUY':
                            tp_price = current_price * 1.015
                            sl_price = current_price * 0.99
                        else:
                            tp_price = current_price * 0.985
                            sl_price = current_price * 1.01
                    
                    # Abrir posição
                    position = {
                        'side': action,
                        'entry_price': current_price,
                        'entry_idx': i,
                        'entry_time': historical_data.index[i] if hasattr(historical_data, 'index') else i,
                        'size': position_size / current_price,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'confidence': confidence
                    }
            
            # Saída
            elif position is not None:
                should_close, reason = self._check_exit_conditions(
                    position, current_price, action, confidence
                )
                
                if should_close:
                    # Fechar posição
                    trade_result = self._close_position(
                        position, current_price, i, reason, balance
                    )
                    
                    trades.append(trade_result)
                    balance += trade_result['pnl_net']
                    position = None
            
            # Registrar equity
            equity_curve.append({
                'index': i,
                'timestamp': historical_data.index[i] if hasattr(historical_data, 'index') else i,
                'balance': balance,
                'in_position': position is not None,
                'price': current_price
            })
        
        # Fechar posição aberta no final
        if position is not None:
            trade_result = self._close_position(
                position, prices[-1], len(prices)-1, "Fim do backtest", balance
            )
            trades.append(trade_result)
            balance += trade_result['pnl_net']
        
        # Log final
        logger.info(f"""
📊 Backtest concluído:
- Período: {len(prices)} candles
- Sinais gerados: {total_signals}
- Distribuição: BUY={signal_counts['BUY']}, SELL={signal_counts['SELL']}, HOLD={signal_counts['HOLD']}
- Trades executados: {len(trades)}
- Balance final: ${balance:,.2f}
        """)
        
        # Calcular métricas
        return self._calculate_metrics(
            trades, initial_balance, balance, equity_curve
        )
    
    def _extract_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        idx: int,
        tech_details: Dict
    ) -> Dict:
        """Extrai features para ML"""
        # Momentum
        momentum = 0
        if idx >= 20:
            momentum = (prices[idx] - prices[idx-20]) / prices[idx-20]
        
        # Volume ratio
        volume_ratio = 1
        if idx >= 20:
            avg_volume = np.mean(volumes[max(0, idx-20):idx])
            if avg_volume > 0:
                volume_ratio = volumes[idx] / avg_volume
        
        # Volatilidade
        volatility = 0.01
        if idx >= 50:
            volatility = np.std(prices[max(0, idx-50):idx]) / np.mean(prices[max(0, idx-50):idx])
        
        # Trend
        price_trend = 0
        if idx >= 50:
            # Regressão linear simples
            x = np.arange(50)
            y = prices[idx-50:idx]
            if len(y) == 50:
                slope = np.polyfit(x, y, 1)[0]
                price_trend = slope / np.mean(y)
        
        return {
            'rsi': tech_details.get('rsi', 50),
            'momentum': momentum,
            'volume_ratio': volume_ratio,
            'spread_bps': 5,  # Assumir spread fixo no backtest
            'volatility': volatility,
            'price_trend': price_trend
        }
    
    def _calculate_position_size(
        self,
        balance: float,
        confidence: float,
        volatility: float
    ) -> float:
        """Calcula tamanho da posição para backtest"""
        # Kelly simplificado
        base_size = balance * self.config.max_position_pct * confidence
        
        # Ajustar por volatilidade
        if volatility > 0.03:
            base_size *= 0.5
        elif volatility > 0.02:
            base_size *= 0.7
        
        # Limites
        min_size = 50  # USD
        max_size = balance * 0.1  # 10% máximo
        
        return max(min_size, min(base_size, max_size))
    
    def _check_exit_conditions(
        self,
        position: Dict,
        current_price: float,
        signal_action: str,
        signal_confidence: float
    ) -> Tuple[bool, str]:
        """Verifica condições de saída"""
        side = position['side']
        entry_price = position['entry_price']
        tp_price = position['tp_price']
        sl_price = position['sl_price']
        
        # Stop Loss / Take Profit
        if side == 'BUY':
            if current_price >= tp_price:
                return True, "Take Profit"
            elif current_price <= sl_price:
                return True, "Stop Loss"
        else:  # SELL
            if current_price <= tp_price:
                return True, "Take Profit"
            elif current_price >= sl_price:
                return True, "Stop Loss"
        
        # Sinal contrário forte
        if side == 'BUY' and signal_action == 'SELL' and signal_confidence > 0.8:
            return True, "Sinal Contrário"
        elif side == 'SELL' and signal_action == 'BUY' and signal_confidence > 0.8:
            return True, "Sinal Contrário"
        
        return False, ""
    
    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_idx: int,
        reason: str,
        current_balance: float
    ) -> Dict:
        """Fecha posição e calcula resultado"""
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        # Calcular P&L
        if side == 'BUY':
            pnl_gross = (exit_price - entry_price) * size
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SELL
            pnl_gross = (entry_price - exit_price) * size
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # Taxas (0.1% entrada + saída)
        entry_fee = entry_price * size * 0.001
        exit_fee = exit_price * size * 0.001
        total_fees = entry_fee + exit_fee
        
        pnl_net = pnl_gross - total_fees
        
        return {
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_idx': position['entry_idx'],
            'exit_idx': exit_idx,
            'entry_time': position['entry_time'],
            'size': size,
            'pnl_gross': pnl_gross,
            'pnl_net': pnl_net,
            'pnl_pct': pnl_pct,
            'fees': total_fees,
            'reason': reason,
            'duration': exit_idx - position['entry_idx'],
            'confidence': position['confidence']
        }
    
    def _calculate_metrics(
        self,
        trades: List[Dict],
        initial_balance: float,
        final_balance: float,
        equity_curve: List[Dict]
    ) -> Dict:
        """Calcula métricas detalhadas do backtest"""
        if not trades:
            logger.warning("Nenhum trade executado no backtest!")
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_fees': 0,
                'net_profit': 0
            }
        
        # Converter para DataFrame para análise
        df_trades = pd.DataFrame(trades)
        
        # Separar ganhos e perdas
        wins = df_trades[df_trades['pnl_net'] > 0]
        losses = df_trades[df_trades['pnl_net'] < 0]
        
        # Métricas básicas
        num_trades = len(trades)
        win_rate = len(wins) / num_trades
        
        # Profit factor
        total_wins = wins['pnl_net'].sum() if not wins.empty else 0
        total_losses = abs(losses['pnl_net'].sum()) if not losses.empty else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        # Médias
        avg_win = wins['pnl_net'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl_net'].mean()) if not losses.empty else 0
        
        # Retorno total
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Taxas
        total_fees = df_trades['fees'].sum()
        gross_profit = df_trades['pnl_gross'].sum()
        net_profit = df_trades['pnl_net'].sum()
        
        # Sharpe ratio
        returns = df_trades['pnl_pct'].values
        sharpe = self._calculate_sharpe(returns)
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown(equity_curve)
        
        # Estatísticas adicionais
        avg_trade_duration = df_trades['duration'].mean()
        max_consecutive_wins = self._count_consecutive(df_trades['pnl_net'] > 0)
        max_consecutive_losses = self._count_consecutive(df_trades['pnl_net'] < 0)
        
        # Win/loss por tipo
        buy_trades = df_trades[df_trades['side'] == 'BUY']
        sell_trades = df_trades[df_trades['side'] == 'SELL']
        
        buy_win_rate = (buy_trades['pnl_net'] > 0).mean() if not buy_trades.empty else 0
        sell_win_rate = (sell_trades['pnl_net'] > 0).mean() if not sell_trades.empty else 0
        
        # Log detalhado
        logger.info(f"""
📊 MÉTRICAS DETALHADAS DO BACKTEST
═══════════════════════════════════════════════
💰 FINANCEIRO:
• Capital inicial: ${initial_balance:,.2f}
• Capital final: ${final_balance:,.2f}
• Retorno Total: {total_return*100:+.2f}%
• Lucro Bruto: ${gross_profit:,.2f}
• Taxas Totais: ${total_fees:,.2f}
• Lucro Líquido: ${net_profit:,.2f}

📈 PERFORMANCE:
• Número de Trades: {num_trades}
• Taxa de Acerto: {win_rate*100:.1f}%
• Profit Factor: {profit_factor:.2f}
• Sharpe Ratio: {sharpe:.2f}
• Max Drawdown: {max_dd*100:.2f}%

💵 MÉDIAS:
• Ganho Médio: ${avg_win:.2f}
• Perda Média: ${avg_loss:.2f}
• Duração Média: {avg_trade_duration:.0f} candles

📊 ANÁLISE POR TIPO:
• BUY Win Rate: {buy_win_rate*100:.1f}% ({len(buy_trades)} trades)
• SELL Win Rate: {sell_win_rate*100:.1f}% ({len(sell_trades)} trades)

🔢 SEQUÊNCIAS:
• Máx. vitórias consecutivas: {max_consecutive_wins}
• Máx. derrotas consecutivas: {max_consecutive_losses}
═══════════════════════════════════════════════
        """)
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_fees': total_fees,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'avg_trade_duration': avg_trade_duration,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calcula Sharpe Ratio"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Assumir 252 períodos por ano (ajustar conforme timeframe)
        return np.sqrt(252) * mean_return / std_return
    
    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        """Calcula drawdown máximo"""
        if not equity_curve:
            return 0
        
        balances = [e['balance'] for e in equity_curve]
        
        peak = balances[0]
        max_dd = 0
        
        for balance in balances[1:]:
            if balance > peak:
                peak = balance
            else:
                dd = (peak - balance) / peak
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _count_consecutive(self, series: pd.Series) -> int:
        """Conta máximo de valores True consecutivos"""
        if series.empty:
            return 0
        
        max_count = 0
        current_count = 0
        
        for value in series:
            if value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count


async def run_backtest_validation(
    config = None,
    days: int = 7,
    debug_mode: bool = False
) -> Optional[Dict]:
    """
    Executa backtest para validação de estratégia
    
    Args:
        config: Configuração (criará uma se None)
        days: Dias de dados históricos
        debug_mode: Modo debug ativado
        
    Returns:
        Resultados do backtest ou None se falhar
    """
    from trade_system.config import get_config
    from binance.client import Client
    
    # Configuração
    if config is None:
        config = get_config(debug_mode=debug_mode)
    
    logger.info(f"🔬 Executando backtest de validação ({days} dias)...")
    
    # Verificar credenciais
    if not config.api_key or not config.api_secret:
        logger.error("❌ Credenciais da Binance não configuradas")
        return None
    
    try:
        # Cliente Binance
        client = Client(config.api_key, config.api_secret)
        
        # Determinar intervalo
        if days <= 1:
            interval = Client.KLINE_INTERVAL_1MINUTE
            expected_candles = days * 24 * 60
        elif days <= 7:
            interval = Client.KLINE_INTERVAL_5MINUTE
            expected_candles = days * 24 * 12
        elif days <= 30:
            interval = Client.KLINE_INTERVAL_15MINUTE
            expected_candles = days * 24 * 4
        else:
            interval = Client.KLINE_INTERVAL_1HOUR
            expected_candles = days * 24
        
        # Limitar para não exceder limite da API
        limit = min(expected_candles, 1000)
        
        logger.info(f"📊 Baixando {limit} candles de {config.symbol}...")
        
        # Obter dados
        klines = client.get_klines(
            symbol=config.symbol,
            interval=interval,
            limit=limit
        )
        
        # Converter para DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Processar dados
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        logger.info(f"✅ Dados carregados: {len(df)} candles")
        logger.info(f"   Período: {df.index[0]} até {df.index[-1]}")
        logger.info(f"   Preço atual: ${df['close'].iloc[-1]:,.2f}")
        
        # Executar backtest
        backtester = IntegratedBacktester(config)
        results = await backtester.backtest_strategy(df)
        
        # Validar resultados
        if results and results['num_trades'] > 0:
            if results['win_rate'] < 0.40:
                logger.warning("⚠️ Taxa de acerto baixa no backtest")
            if results['profit_factor'] < 1.0:
                logger.warning("⚠️ Profit factor abaixo de 1.0")
            if results['max_drawdown'] > 0.20:
                logger.warning("⚠️ Drawdown máximo acima de 20%")
        else:
            logger.warning("⚠️ Nenhum trade gerado no backtest")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro no backtest: {e}")
        return None
''',

    "trade_system/main.py": '''"""
Módulo principal para orquestração do sistema de trading
"""
import os
import asyncio
import signal
from datetime import datetime
from typing import Optional
from trade_system.config import get_config
from trade_system.logging_config import setup_logging, get_logger
from trade_system.cache import UltraFastCache
from trade_system.rate_limiter import RateLimiter
from trade_system.alerts import AlertSystem
from trade_system.websocket_manager import UltraFastWebSocketManager
from trade_system.analysis.technical import UltraFastTechnicalAnalysis
from trade_system.analysis.orderbook import ParallelOrderbookAnalyzer
from trade_system.analysis.ml import SimplifiedMLPredictor
from trade_system.risk import UltraFastRiskManager, MarketConditionValidator
from trade_system.signals import OptimizedSignalConsolidator
from trade_system.checkpoint import CheckpointManager
from trade_system.backtester import run_backtest_validation

logger = get_logger(__name__)


class TradingSystem:
    """Sistema principal de trading"""
    
    def __init__(self, config=None, paper_trading=True):
        """
        Inicializa o sistema de trading
        
        Args:
            config: Configuração do sistema
            paper_trading: Se True, executa em modo simulado
        """
        self.config = config or get_config()
        self.paper_trading = paper_trading
        
        # Componentes principais
        self.cache = UltraFastCache(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.alert_system = AlertSystem(self.config)
        self.ws_manager = UltraFastWebSocketManager(self.config, self.cache)
        
        # Análise
        self.technical_analyzer = UltraFastTechnicalAnalysis(self.config)
        self.orderbook_analyzer = ParallelOrderbookAnalyzer(self.config)
        self.ml_predictor = SimplifiedMLPredictor()
        self.signal_consolidator = OptimizedSignalConsolidator()
        
        # Risk e validação
        self.risk_manager = UltraFastRiskManager(self.config)
        self.market_validator = MarketConditionValidator(self.config)
        
        # Checkpoint
        self.checkpoint_manager = CheckpointManager()
        
        # Estado
        self.position = None
        self.is_running = False
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'start_balance': self.risk_manager.current_balance,
            'session_start': datetime.now()
        }
        
        logger.info(f"🚀 Sistema inicializado - Modo: {'PAPER TRADING' if paper_trading else 'LIVE'}")
    
    async def initialize(self):
        """Inicializa componentes assíncronos"""
        # Carregar checkpoint se existir
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            self._restore_from_checkpoint(checkpoint)
        
        # Iniciar WebSocket
        self.ws_manager.start_delayed()
        
        # Alerta de início
        await self.alert_system.send_startup_alert(
            mode="PAPER" if self.paper_trading else "LIVE"
        )
        
        # Aguardar dados
        logger.info("⏳ Aguardando dados do WebSocket...")
        for i in range(50):
            await asyncio.sleep(0.1)
            if self.ws_manager.buffer_filled or self.ws_manager.buffer_index > 100:
                logger.info("✅ Dados recebidos!")
                break
    
    async def run(self):
        """Loop principal do sistema"""
        self.is_running = True
        cycle_count = 0
        
        try:
            while self.is_running:
                # Obter dados
                market_data = self.ws_manager.get_latest_data()
                if not market_data:
                    await asyncio.sleep(0.1)
                    continue
                
                # Validar mercado
                is_safe, reasons = await self.market_validator.validate_market_conditions(
                    market_data
                )
                
                if not is_safe and not self.config.debug_mode:
                    if cycle_count % 100 == 0:  # Log a cada 10s
                        logger.warning(f"⚠️ Mercado inseguro: {', '.join(reasons)}")
                    await asyncio.sleep(0.1)
                    continue
                
                # Analisar mercado
                signals = await self._analyze_market(market_data)
                
                # Consolidar sinais
                action, confidence = self.signal_consolidator.consolidate(signals)
                
                # Gerenciar posições
                if self.position:
                    await self._manage_position(market_data, action, confidence)
                else:
                    if action != 'HOLD' and confidence >= self.config.min_confidence:
                        await self._open_position(market_data, action, confidence)
                
                # Checkpoint periódico
                if self.checkpoint_manager.should_checkpoint():
                    await self._save_checkpoint()
                
                # Incrementar contador
                cycle_count += 1
                
                # Sleep
                await asyncio.sleep(self.config.main_loop_interval_ms / 1000)
                
        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}", exc_info=True)
            await self.alert_system.send_alert(
                "Erro no Sistema",
                str(e),
                "critical"
            )
        finally:
            await self.shutdown()
    
    async def _analyze_market(self, market_data):
        """Analisa mercado e retorna sinais"""
        # Análise técnica
        tech_action, tech_conf, tech_details = self.technical_analyzer.analyze(
            market_data['prices'],
            market_data['volumes']
        )
        
        # Análise orderbook
        ob_action, ob_conf, ob_details = self.orderbook_analyzer.analyze(
            market_data['orderbook_bids'],
            market_data['orderbook_asks'],
            self.cache
        )
        
        # ML
        features = {
            'rsi': tech_details.get('rsi', 50),
            'momentum': self._calculate_momentum(market_data['prices']),
            'volume_ratio': self._calculate_volume_ratio(market_data['volumes']),
            'spread_bps': ob_details.get('spread_bps', 10),
            'volatility': self._calculate_volatility(market_data['prices'])
        }
        
        ml_action, ml_conf = self.ml_predictor.predict(features)
        
        return [
            ('technical', tech_action, tech_conf),
            ('orderbook', ob_action, ob_conf),
            ('ml', ml_action, ml_conf)
        ]
    
    async def _open_position(self, market_data, action, confidence):
        """Abre nova posição"""
        if self.paper_trading:
            await self._open_paper_position(market_data, action, confidence)
        else:
            # TODO: Implementar trading real
            logger.error("Trading real ainda não implementado")
    
    async def _open_paper_position(self, market_data, action, confidence):
        """Abre posição simulada"""
        current_price = float(market_data['prices'][-1])
        volatility = self._calculate_volatility(market_data['prices'])
        
        # Calcular tamanho
        position_size = self.risk_manager.calculate_position_size(
            confidence, volatility, current_price
        )
        
        if position_size == 0:
            return
        
        # Calcular taxas
        entry_fee = position_size * 0.001
        
        # Criar posição
        self.position = {
            'side': action,
            'entry_price': current_price,
            'quantity': position_size / current_price,
            'entry_time': datetime.now(),
            'confidence': confidence,
            'entry_fee': entry_fee,
            'paper_trade': True
        }
        
        # Atualizar risk manager
        self.risk_manager.set_position_info(self.position)
        self.risk_manager.current_balance -= entry_fee
        
        logger.info(f"""
🟢 POSIÇÃO ABERTA [{action}]
- Preço: ${current_price:,.2f}
- Quantidade: {self.position['quantity']:.6f}
- Valor: ${position_size:,.2f}
- Taxa: ${entry_fee:.2f}
- Confiança: {confidence*100:.1f}%
        """)
        
        await self.alert_system.send_alert(
            f"Nova Posição {action}",
            f"Preço: ${current_price:,.2f}\\nValor: ${position_size:,.2f}",
            "info"
        )
    
    async def _manage_position(self, market_data, signal_action, signal_confidence):
        """Gerencia posição existente"""
        current_price = float(market_data['prices'][-1])
        
        should_close, reason = self.risk_manager.should_close_position(
            current_price,
            self.position['entry_price'],
            self.position['side']
        )
        
        if should_close:
            await self._close_position(current_price, reason)
    
    async def _close_position(self, exit_price, reason):
        """Fecha posição"""
        if not self.position:
            return
        
        # Calcular resultado
        entry_price = self.position['entry_price']
        quantity = self.position['quantity']
        side = self.position['side']
        
        if side == 'BUY':
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        # Taxas
        exit_fee = exit_price * quantity * 0.001
        total_fees = self.position['entry_fee'] + exit_fee
        pnl_net = pnl - exit_fee
        
        # Atualizar estatísticas
        self.performance_stats['total_trades'] += 1
        if pnl_net > 0:
            self.performance_stats['winning_trades'] += 1
        self.performance_stats['total_pnl'] += pnl_net
        self.performance_stats['total_fees'] += total_fees
        
        # Atualizar risk manager
        self.risk_manager.update_pnl(pnl_net, exit_fee)
        self.risk_manager.clear_position()
        
        logger.info(f"""
🔴 POSIÇÃO FECHADA
- Motivo: {reason}
- Saída: ${exit_price:,.2f}
- P&L: ${pnl_net:,.2f}
- Taxas: ${total_fees:.2f}
        """)
        
        self.position = None
    
    async def _save_checkpoint(self):
        """Salva checkpoint do sistema"""
        state = {
            'balance': self.risk_manager.current_balance,
            'position': self.position,
            'performance_stats': self.performance_stats,
            'paper_trading': self.paper_trading
        }
        
        self.checkpoint_manager.save_checkpoint(state)
        self.checkpoint_manager.update_checkpoint_time()
    
    def _restore_from_checkpoint(self, checkpoint):
        """Restaura estado do checkpoint"""
        self.risk_manager.current_balance = checkpoint.get('balance', 10000)
        self.position = checkpoint.get('position')
        self.performance_stats = checkpoint.get('performance_stats', self.performance_stats)
        
        logger.info("✅ Estado restaurado do checkpoint")
    
    async def shutdown(self):
        """Desliga o sistema"""
        logger.info("🛑 Desligando sistema...")
        self.is_running = False
        
        # Parar WebSocket
        self.ws_manager.stop()
        
        # Salvar checkpoint final
        await self._save_checkpoint()
        
        # Alerta de desligamento
        await self.alert_system.send_shutdown_alert()
        
        logger.info("✅ Sistema desligado")
    
    # Métodos auxiliares
    def _calculate_momentum(self, prices):
        if len(prices) < 20:
            return 0
        return (prices[-1] - prices[-20]) / prices[-20]
    
    def _calculate_volume_ratio(self, volumes):
        if len(volumes) < 20:
            return 1
        avg = np.mean(volumes[-20:])
        return volumes[-1] / avg if avg > 0 else 1
    
    def _calculate_volatility(self, prices):
        if len(prices) < 50:
            return 0.01
        return np.std(prices[-50:]) / np.mean(prices[-50:])


async def run_paper_trading(
    config=None,
    initial_balance=10000,
    debug_mode=False
):
    """
    Executa paper trading
    
    Args:
        config: Configuração do sistema
        initial_balance: Balance inicial
        debug_mode: Modo debug
    """
    # Setup
    setup_logging()
    
    if config is None:
        config = get_config(debug_mode=debug_mode)
    
    # Criar sistema
    system = TradingSystem(config, paper_trading=True)
    system.risk_manager.current_balance = initial_balance
    system.risk_manager.initial_balance = initial_balance
    
    # Inicializar
    await system.initialize()
    
    # Executar
    try:
        await system.run()
    except KeyboardInterrupt:
        logger.info("⏹️ Interrompido pelo usuário")
    finally:
        await system.shutdown()


def handle_signals():
    """Configura handlers de sinais"""
    def signal_handler(sig, frame):
        logger.info(f"Sinal {sig} recebido")
        # Será tratado no loop principal
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # Exemplo de uso direto
    handle_signals()
    asyncio.run(run_paper_trading(debug_mode=True))
''',
}


def create_missing_modules_zip():
    """Cria ZIP com os módulos faltantes"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"trading_system_missing_modules_{timestamp}.zip"
    
    print(f"📦 Criando arquivo ZIP com módulos faltantes: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filepath, content in MISSING_MODULES.items():
            print(f"   ✅ Adicionando: {filepath}")
            zipf.writestr(filepath, content)
    
    # Calcular tamanho
    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    
    print(f"\n✅ Arquivo criado com sucesso!")
    print(f"📊 Tamanho: {size_mb:.2f} MB")
    print(f"📁 Nome: {zip_filename}")
    print(f"\n📋 Instruções:")
    print(f"1. Extraia este arquivo na pasta do projeto")
    print(f"2. Os módulos serão adicionados aos já existentes")
    print(f"3. Reinstale o pacote: pip install -e .")
    print(f"4. Teste: trade-system paper")
    
    return zip_filename


if __name__ == "__main__":
    print("="*60)
    print("Módulos Faltantes - Sistema de Trading v5.2")
    print("="*60)
    
    try:
        zip_file = create_missing_modules_zip()
        print(f"\n🎉 Sucesso! O arquivo '{zip_file}' está pronto.")
        print("\n⚠️ IMPORTANTE: Extraia na mesma pasta do projeto principal!")
    except Exception as e:
        print(f"\n❌ Erro ao criar o arquivo ZIP: {e}")
        import traceback
        traceback.print_exc()