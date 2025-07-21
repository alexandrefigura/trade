#!/usr/bin/env python3
"""
Sistema de Trading Funcional - Paper Trading
"""
import asyncio
import os
import json
from datetime import datetime
from typing import Dict, Any
import aiohttp
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

class SimpleWebSocket:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price = 0
        self.running = False
        
    async def start(self):
        """Conecta ao websocket da Binance"""
        self.running = True
        session = aiohttp.ClientSession()
        
        try:
            # URL do websocket da Binance
            url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@ticker"
            
            async with session.ws_connect(url) as ws:
                print(f"✅ WebSocket conectado para {self.symbol}")
                
                async for msg in ws:
                    if not self.running:
                        break
                        
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        self.price = float(data['c'])  # Current price
                        
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f'❌ WebSocket error: {ws.exception()}')
                        break
                        
        except Exception as e:
            print(f"❌ Erro no WebSocket: {e}")
        finally:
            await session.close()
            
    def stop(self):
        self.running = False

class SimpleTechnicalAnalysis:
    def __init__(self):
        self.prices = []
        
    def update(self, price: float):
        self.prices.append(price)
        # Manter apenas últimos 100 preços
        if len(self.prices) > 100:
            self.prices.pop(0)
            
    def get_signal(self) -> str:
        """Retorna sinal simples baseado em média móvel"""
        if len(self.prices) < 20:
            return "HOLD"
            
        # Média móvel simples
        sma_short = np.mean(self.prices[-10:])
        sma_long = np.mean(self.prices[-20:])
        
        if sma_short > sma_long * 1.001:  # 0.1% acima
            return "BUY"
        elif sma_short < sma_long * 0.999:  # 0.1% abaixo
            return "SELL"
        else:
            return "HOLD"

class PaperTradingSystem:
    def __init__(self, symbol: str = "BTCUSDT", initial_balance: float = 10000):
        self.symbol = symbol
        self.balance = initial_balance
        self.position = 0
        self.trades = []
        self.websocket = SimpleWebSocket(symbol)
        self.ta = SimpleTechnicalAnalysis()
        
    async def run(self):
        """Executa o sistema de paper trading"""
        print(f"""
🤖 SISTEMA DE PAPER TRADING INICIADO
{'=' * 50}
📊 Par: {self.symbol}
💰 Balance Inicial: ${self.balance:,.2f}
⏰ Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}
        """)
        
        # Iniciar websocket em background
        ws_task = asyncio.create_task(self.websocket.start())
        
        # Aguardar conexão
        await asyncio.sleep(2)
        
        try:
            while True:
                if self.websocket.price > 0:
                    # Atualizar análise técnica
                    self.ta.update(self.websocket.price)
                    
                    # Obter sinal
                    signal = self.ta.get_signal()
                    
                    # Executar trade se necessário
                    if signal == "BUY" and self.position == 0:
                        # Comprar
                        amount = self.balance * 0.95  # Usar 95% do balance
                        self.position = amount / self.websocket.price
                        self.balance -= amount
                        
                        trade = {
                            'time': datetime.now(),
                            'action': 'BUY',
                            'price': self.websocket.price,
                            'amount': self.position,
                            'value': amount
                        }
                        self.trades.append(trade)
                        
                        print(f"""
📈 COMPRA EXECUTADA
   Preço: ${self.websocket.price:,.2f}
   Quantidade: {self.position:.6f}
   Valor: ${amount:,.2f}
   Balance: ${self.balance:,.2f}
                        """)
                        
                    elif signal == "SELL" and self.position > 0:
                        # Vender
                        amount = self.position * self.websocket.price
                        self.balance += amount
                        
                        trade = {
                            'time': datetime.now(),
                            'action': 'SELL',
                            'price': self.websocket.price,
                            'amount': self.position,
                            'value': amount
                        }
                        self.trades.append(trade)
                        
                        print(f"""
📉 VENDA EXECUTADA
   Preço: ${self.websocket.price:,.2f}
   Quantidade: {self.position:.6f}
   Valor: ${amount:,.2f}
   Balance: ${self.balance:,.2f}
                        """)
                        
                        self.position = 0
                    
                    # Status atual
                    total_value = self.balance + (self.position * self.websocket.price if self.position > 0 else 0)
                    profit_pct = ((total_value - 10000) / 10000) * 100
                    
                    print(f"\r💹 {self.symbol}: ${self.websocket.price:,.2f} | "
                          f"Balance: ${self.balance:,.2f} | "
                          f"Posição: {self.position:.6f} | "
                          f"Total: ${total_value:,.2f} ({profit_pct:+.2f}%) | "
                          f"Sinal: {signal}", end='', flush=True)
                
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Sistema interrompido pelo usuário")
        finally:
            # Parar websocket
            self.websocket.stop()
            
            # Mostrar resumo
            self.show_summary()
            
    def show_summary(self):
        """Mostra resumo das operações"""
        if not self.trades:
            print("\n📊 Nenhuma operação realizada")
            return
            
        print(f"""
\n{'=' * 50}
📊 RESUMO DAS OPERAÇÕES
{'=' * 50}
Total de trades: {len(self.trades)}
        """)
        
        for i, trade in enumerate(self.trades, 1):
            print(f"{i}. {trade['action']} - "
                  f"${trade['price']:,.2f} - "
                  f"{trade['amount']:.6f} - "
                  f"${trade['value']:,.2f}")

async def main():
    # Verificar API key
    api_key = os.getenv('BINANCE_API_KEY')
    if not api_key:
        print("❌ BINANCE_API_KEY não encontrada no .env")
        return
        
    print(f"✅ API Key: {api_key[:8]}...")
    
    # Criar e executar sistema
    system = PaperTradingSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())
