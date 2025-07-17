"""
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
