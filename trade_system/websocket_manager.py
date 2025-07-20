"""WebSocket Manager para conexão em tempo real"""
import asyncio
import json
import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
import pandas as pd

class WebSocketManager:
    """Gerencia conexões WebSocket com a Binance"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.symbol = getattr(config, 'symbol', 'BTCUSDT').lower()
        
        # Buffers
        self.price_buffer = deque(maxlen=1000)
        self.trade_buffer = deque(maxlen=1000)
        self.orderbook_buffer = deque(maxlen=100)
        
        # Estado
        self.is_connected = False
        self.ws = None
        
        self.logger.info("📡 WebSocket Manager inicializado")
    
    async def connect(self):
        """Conecta ao WebSocket da Binance"""
        # Por enquanto, usar REST API com polling
        self.is_connected = True
        self.logger.info("✅ WebSocket simulado conectado")
        
        # Iniciar task de polling
        asyncio.create_task(self._polling_loop())
    
    async def disconnect(self):
        """Desconecta WebSocket"""
        self.is_connected = False
        self.logger.info("WebSocket desconectado")
    
    async def _polling_loop(self):
        """Loop de polling para simular WebSocket"""
        while self.is_connected:
            try:
                async with aiohttp.ClientSession() as session:
                    # Buscar preço atual
                    url = f"https://api.binance.com/api/v3/ticker/price?symbol={self.symbol.upper()}"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            price_data = {
                                'close': float(data['price']),
                                'time': datetime.now(),
                                'symbol': data['symbol']
                            }
                            self.price_buffer.append(price_data)
                
                await asyncio.sleep(5)  # Polling a cada 5 segundos
                
            except Exception as e:
                self.logger.error(f"Erro no polling: {e}")
                await asyncio.sleep(10)
    
    def get_latest_price(self) -> Optional[float]:
        """Retorna último preço"""
        if self.price_buffer:
            return self.price_buffer[-1]['close']
        return None
    
    def get_latest_orderbook(self) -> Optional[Dict]:
        """Retorna último orderbook"""
        # Simulado por enquanto
        if self.orderbook_buffer:
            return self.orderbook_buffer[-1]
        
        # Orderbook fake para testes
        return {
            'bids': [[str(self.get_latest_price() or 50000 - i*10), str(1.0/(i+1))] for i in range(10)],
            'asks': [[str(self.get_latest_price() or 50000 + i*10), str(1.0/(i+1))] for i in range(10)],
            'timestamp': datetime.now().timestamp()
        }
    
    def get_candles_df(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Retorna DataFrame com candles"""
        if len(self.price_buffer) < 2:
            return None
        
        # Criar DataFrame simples
        data = list(self.price_buffer)[-limit:]
        df = pd.DataFrame(data)
        
        # Adicionar colunas necessárias
        if 'close' in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df['close'] * 1.001
            df['low'] = df['close'] * 0.999
            df['volume'] = 1000.0
        
        return df
    
    def get_market_metrics(self) -> Dict[str, float]:
        """Calcula métricas de mercado"""
        return {
            'buy_volume_ratio': 0.5,
            'sell_volume_ratio': 0.5,
            'vwap': self.get_latest_price() or 0,
            'trade_count': len(self.trade_buffer)
        }

# Alias para compatibilidade
UltraFastWebSocketManager = WebSocketManager
