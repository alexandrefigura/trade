"""
WebSocket Manager ultra-rápido para dados em tempo real
"""
import time
import queue
import threading
import logging
import asyncio
import numpy as np
from datetime import datetime
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
        self.trades_processed = 0  # Contador separado para trades
        self.orderbook_updates = 0  # Contador para orderbook
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
        
        # Lock para proteger buffer
        self._buffer_lock = threading.Lock()
        
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
                
                # Log de status a cada 30 segundos
                if self.is_connected and self.buffer_index > 0 and self.buffer_index % 1000 == 0:
                    logger.info(f"📊 Status: Buffer={self.buffer_index}, Trades={self.trades_processed}, Orderbook={self.orderbook_updates}")
                    
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
            
            # Atualizar buffer com lock
            with self._buffer_lock:
                idx = self.buffer_index % self.config.price_buffer_size
                self.price_buffer[idx] = price
                self.volume_buffer[idx] = volume
                self.time_buffer[idx] = timestamp
                
                self.buffer_index += 1
                if self.buffer_index >= self.config.price_buffer_size:
                    self.buffer_filled = True
            
            self.messages_processed += 1
            self.trades_processed += 1
            
            # Log apenas no primeiro trade
            if self.trades_processed == 1:
                logger.info(f"✅ Primeiro trade: ${price:,.2f}")
            elif self.trades_processed % 1000 == 0:
                logger.debug(f"📊 {self.trades_processed} trades processados - Buffer: {self.buffer_index}")
            
        except Exception as e:
            logger.error(f"Erro processando trade: {e}")
    
    def _process_orderbook(self, msg):
        """Processa orderbook otimizado"""
        try:
            self.last_message_time = time.time()
            self.orderbook_updates += 1
            
            bids = msg['bids']
            asks = msg['asks']
            
            # Copiar para arrays com lock
            with self._buffer_lock:
                for i in range(min(20, len(bids))):
                    self.orderbook_bids[i, 0] = float(bids[i][0])
                    self.orderbook_bids[i, 1] = float(bids[i][1])
                
                for i in range(min(20, len(asks))):
                    self.orderbook_asks[i, 0] = float(asks[i][0])
                    self.orderbook_asks[i, 1] = float(asks[i][1])
                
                # IMPORTANTE: Adicionar mid price ao buffer de preços
                if len(bids) > 0 and len(asks) > 0:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    mid_price = (best_bid + best_ask) / 2
                    
                    # Adicionar ao buffer de preços
                    idx = self.buffer_index % self.config.price_buffer_size
                    self.price_buffer[idx] = mid_price
                    self.volume_buffer[idx] = (self.orderbook_bids[0, 1] + self.orderbook_asks[0, 1]) / 2
                    self.time_buffer[idx] = int(time.time() * 1000)
                    
                    self.buffer_index += 1
                    if self.buffer_index >= self.config.price_buffer_size:
                        self.buffer_filled = True
                    
                    self.messages_processed += 1
            
            # Calcular imbalance
            bid_volume = np.sum(self.orderbook_bids[:5, 1])
            ask_volume = np.sum(self.orderbook_asks[:5, 1])
            
            if bid_volume + ask_volume > 0:
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                
                self.cache.set('orderbook_imbalance', imbalance, ttl=1)
                
                # Log apenas imbalances significativos
                if abs(imbalance) > 0.7:
                    self.high_priority_queue.put(('imbalance', imbalance))
                    
            # Log primeiro orderbook
            if self.orderbook_updates == 1:
                logger.info("✅ Primeiro orderbook recebido")
                    
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
        # Log para debug
        if self.config.debug_mode and self.buffer_index > 0 and self.buffer_index % 100 == 0:
            logger.debug(f"get_latest_data: buffer_index={self.buffer_index}, filled={self.buffer_filled}")
        
        if self.simulate_mode:
            return self._get_simulated_data()
        
        # Verificar se temos dados mínimos (reduzido para 50 no início)
        with self._buffer_lock:
            min_required = 50 if self.buffer_index < 500 else 200
            
            if self.buffer_index < min_required:
                if self.config.debug_mode and self.buffer_index > 0 and self.buffer_index % 20 == 0:
                    logger.debug(f"Buffer insuficiente: {self.buffer_index}/{min_required}")
                return None
            
            # Determinar range de dados
            if self.buffer_filled:
                # Buffer circular cheio - pegar últimos 1000
                end_idx = self.buffer_index % self.config.price_buffer_size
                start_idx = (end_idx - 1000) % self.config.price_buffer_size
                
                if start_idx < end_idx:
                    prices = self.price_buffer[start_idx:end_idx].copy()
                    volumes = self.volume_buffer[start_idx:end_idx].copy()
                    timestamps = self.time_buffer[start_idx:end_idx].copy()
                else:
                    # Buffer circular - concatenar duas partes
                    prices = np.concatenate([
                        self.price_buffer[start_idx:],
                        self.price_buffer[:end_idx]
                    ])
                    volumes = np.concatenate([
                        self.volume_buffer[start_idx:],
                        self.volume_buffer[:end_idx]
                    ])
                    timestamps = np.concatenate([
                        self.time_buffer[start_idx:],
                        self.time_buffer[:end_idx]
                    ])
            else:
                # Buffer ainda não cheio - pegar tudo até agora
                prices = self.price_buffer[:self.buffer_index].copy()
                volumes = self.volume_buffer[:self.buffer_index].copy()
                timestamps = self.time_buffer[:self.buffer_index].copy()
            
            return {
                'prices': prices,
                'volumes': volumes,
                'timestamps': timestamps,
                'orderbook_bids': self.orderbook_bids.copy(),
                'orderbook_asks': self.orderbook_asks.copy(),
                'messages_per_second': self._calculate_mps()
            }
    
    def _get_simulated_data(self) -> Dict:
        """Retorna dados simulados para testes"""
        # Atualizar dados simulados periodicamente
        if self.buffer_index % 10 == 0:
            self._update_simulated_data()
        
        return {
            'prices': self.price_buffer[:1000].copy(),
            'volumes': self.volume_buffer[:1000].copy(),
            'timestamps': self.time_buffer[:1000].copy(),
            'orderbook_bids': self.orderbook_bids.copy(),
            'orderbook_asks': self.orderbook_asks.copy(),
            'messages_per_second': 100.0
        }
    
    def _update_simulated_data(self):
        """Atualiza dados simulados com variações"""
        # Adicionar tendência e ruído aos preços
        base_price = self.price_buffer[0] if self.price_buffer[0] > 0 else 108000.0
        trend = np.sin(self.buffer_index / 100) * 100  # Tendência senoidal
        
        for i in range(min(100, len(self.price_buffer))):
            idx = (self.buffer_index + i) % len(self.price_buffer)
            self.price_buffer[idx] = base_price + trend + np.random.randn() * 50
            self.volume_buffer[idx] = np.random.rand() * 10 + 1
        
        # Atualizar orderbook com variação
        current_price = self.price_buffer[self.buffer_index % len(self.price_buffer)]
        for i in range(20):
            self.orderbook_bids[i, 0] = current_price - (i * 10) - np.random.rand() * 5
            self.orderbook_bids[i, 1] = np.random.rand() * 5 + 1
            self.orderbook_asks[i, 0] = current_price + (i * 10) + np.random.rand() * 5
            self.orderbook_asks[i, 1] = np.random.rand() * 5 + 1
        
        self.buffer_index += 10
    
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
        with self._buffer_lock:
            return {
                'connected': self.is_connected,
                'reconnect_count': self.reconnect_count,
                'buffer_filled': self.buffer_filled,
                'buffer_size': self.buffer_index,
                'trades_processed': self.trades_processed,
                'orderbook_updates': self.orderbook_updates,
                'last_message': datetime.fromtimestamp(self.last_message_time).strftime('%H:%M:%S') if self.last_message_time else 'Never',
                'messages_per_second': self._calculate_mps(),
                'simulate_mode': self.simulate_mode
            }
    
    def enable_simulation_mode(self):
        """Ativa modo simulado para testes"""
        self.simulate_mode = True
        self._fill_with_simulated_data()
        self.is_connected = True
        logger.info("🎮 Modo simulado ativado")
    
    def _fill_with_simulated_data(self):
        """Preenche buffers com dados simulados realistas"""
        # Usar preço base do Bitcoin
        base_price = 108000.0  # Preço similar ao log
        
        # Gerar série de preços com tendência e volatilidade
        for i in range(1000):
            # Adicionar tendência leve e ruído
            trend = np.sin(i / 200) * 500  # Oscilação de ±500
            noise = np.random.randn() * 100  # Ruído de ±100
            self.price_buffer[i] = base_price + trend + noise
            
            # Volume com distribuição mais realista
            self.volume_buffer[i] = np.abs(np.random.randn() * 2 + 5)
            
            # Timestamps incrementais
            self.time_buffer[i] = int(time.time() * 1000) + i * 1000
        
        self.buffer_index = 1000
        self.buffer_filled = True
        
        # Orderbook simulado mais realista
        current_price = self.price_buffer[-1]
        for i in range(20):
            # Bids com volumes decrescentes
            self.orderbook_bids[i, 0] = current_price - (i * 10) - np.random.rand() * 2
            self.orderbook_bids[i, 1] = (20 - i) * np.random.rand() * 0.5 + 0.5
            
            # Asks com volumes decrescentes
            self.orderbook_asks[i, 0] = current_price + (i * 10) + np.random.rand() * 2
            self.orderbook_asks[i, 1] = (20 - i) * np.random.rand() * 0.5 + 0.5
        
        logger.info(f"📊 Dados simulados criados - Preço base: ${base_price:,.2f}")