#!/usr/bin/env python3
"""
Sistema Integrado Funcional
Usa apenas módulos verificados
"""
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import json
import numpy as np

# Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports seguros
from trade_system.config import TradingConfig
from trade_system.logging_config import setup_logging

# Carregar env
load_dotenv()

class IntegratedTradingSystem:
    """Sistema integrado com módulos principais"""
    
    def __init__(self):
        # API keys
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key:
            raise ValueError("BINANCE_API_KEY não encontrada no .env")
        
        print(f"✅ API Key: {self.api_key[:8]}...")
        
        # Setup
        setup_logging()
        self.config = TradingConfig()
        
        # Componentes
        self.components = {}
        self._load_components()
        
        # Paper trading state
        self.balance = getattr(self.config, 'INITIAL_BALANCE', 10000)
        self.position = 0
        self.trades = []
        self.market_data = {}
        self.running = True
    
    def _load_components(self):
        """Carrega componentes disponíveis dinamicamente"""
        print("\n🔄 Carregando componentes...")
        
        # Tentar carregar cada componente
        components_to_try = [
            ('alerts', 'trade_system.alerts', 'AlertSystem'),
            ('websocket', 'trade_system.websocket_manager', 'WebSocketManager'),
            ('technical', 'trade_system.analysis.technical', 'TechnicalAnalyzer'),
            ('risk', 'trade_system.risk', 'RiskManager'),
        ]
        
        for name, module_path, class_name in components_to_try:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                self.components[name] = cls(self.config)
                print(f"   ✅ {name}: {class_name}")
            except Exception as e:
                print(f"   ⚠️ {name}: não carregado ({str(e)[:50]}...)")
    
    async def connect_websocket(self):
        """Conecta ao websocket se disponível"""
        if 'websocket' in self.components:
            try:
                async def on_data(data):
                    self.market_data = data
                
                await self.components['websocket'].connect(on_data)
                print("✅ WebSocket conectado")
            except Exception as e:
                print(f"⚠️ WebSocket: {e}")
                # Fallback - usar HTTP
                await self.use_http_fallback()
        else:
            await self.use_http_fallback()
    
    async def use_http_fallback(self):
        """Fallback para buscar preços via HTTP"""
        import aiohttp
        print("📡 Usando HTTP fallback para preços...")
        
        async def fetch_price():
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={self.config.SYMBOL}"
            async with aiohttp.ClientSession() as session:
                while self.running:
                    try:
                        async with session.get(url) as resp:
                            data = await resp.json()
                            self.market_data = {
                                'price': float(data['price']),
                                'symbol': data['symbol']
                            }
                    except Exception as e:
                        print(f"⚠️ Erro HTTP: {e}")
                    
                    await asyncio.sleep(1)
        
        asyncio.create_task(fetch_price())
    
    async def analyze_market(self):
        """Analisa mercado com componentes disponíveis"""
        while self.running:
            try:
                if not self.market_data or 'price' not in self.market_data:
                    await asyncio.sleep(1)
                    continue
                
                price = float(self.market_data['price'])
                signal = {'action': 'HOLD', 'confidence': 0}
                
                # Análise técnica se disponível
                if 'technical' in self.components:
                    try:
                        result = await self.components['technical'].analyze(self.market_data)
                        if result:
                            signal = result
                    except Exception as e:
                        print(f"⚠️ Erro análise técnica: {e}")
                
                # Validação de risco se disponível
                if signal['action'] != 'HOLD' and 'risk' in self.components:
                    try:
                        risk_data = {
                            'balance': self.balance,
                            'position': self.position,
                            'price': price
                        }
                        if not self.components['risk'].validate_trade(signal, self.market_data, risk_data):
                            signal['action'] = 'HOLD'
                            signal['reason'] = 'Bloqueado pelo Risk Manager'
                    except Exception as e:
                        print(f"⚠️ Erro risk manager: {e}")
                
                # Executar trade
                if signal['action'] != 'HOLD':
                    await self.execute_trade(signal, price)
                
                # Display
                self.display_status(price, signal)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"\n❌ Erro na análise: {e}")
                await asyncio.sleep(5)
    
    async def execute_trade(self, signal, price):
        """Executa paper trade"""
        timestamp = datetime.now()
        
        if signal['action'] == 'BUY' and self.position == 0:
            # Comprar
            amount = self.balance * 0.95
            self.position = amount / price
            self.balance -= amount
            
            self.trades.append({
                'time': timestamp,
                'action': 'BUY',
                'price': price,
                'amount': self.position,
                'value': amount
            })
            
            print(f"\n📈 COMPRA @ ${price:,.2f} - {self.position:.6f} unidades")
            
            # Alerta se disponível
            if 'alerts' in self.components:
                try:
                    self.components['alerts'].send_alert(
                        "TRADE", 
                        f"Compra executada @ ${price:,.2f}",
                        signal
                    )
                except:
                    pass
                    
        elif signal['action'] == 'SELL' and self.position > 0:
            # Vender
            amount = self.position * price
            self.balance += amount
            
            # Calcular lucro
            buy_value = next((t['value'] for t in reversed(self.trades) if t['action'] == 'BUY'), 0)
            profit = amount - buy_value if buy_value else 0
            profit_pct = (profit / buy_value * 100) if buy_value else 0
            
            self.trades.append({
                'time': timestamp,
                'action': 'SELL',
                'price': price,
                'amount': self.position,
                'value': amount,
                'profit': profit,
                'profit_pct': profit_pct
            })
            
            print(f"\n📉 VENDA @ ${price:,.2f} - Lucro: ${profit:,.2f} ({profit_pct:+.2f}%)")
            
            self.position = 0
    
    def display_status(self, price, signal):
        """Mostra status"""
        total = self.balance + (self.position * price if self.position > 0 else 0)
        pnl = total - getattr(self.config, 'INITIAL_BALANCE', 10000)
        pnl_pct = (pnl / getattr(self.config, 'INITIAL_BALANCE', 10000)) * 100
        
        status = f"💹 {self.config.SYMBOL}: ${price:,.2f} | "
        status += f"Balance: ${total:,.2f} ({pnl_pct:+.2f}%) | "
        status += f"Trades: {len(self.trades)} | "
        status += f"Sinal: {signal['action']}"
        
        print(f"\r{status}", end='', flush=True)
    
    async def run(self):
        """Executa o sistema"""
        print(f"""
{'='*60}
🤖 SISTEMA INTEGRADO DE TRADING
{'='*60}
📊 Par: {self.config.SYMBOL}
💰 Balance: ${getattr(self.config, 'INITIAL_BALANCE', 10000):,.2f}
🔧 Componentes: {', '.join(self.components.keys())}
{'='*60}
        """)
        
        # Conectar
        await self.connect_websocket()
        await asyncio.sleep(2)
        
        # Executar
        try:
            await self.analyze_market()
        except KeyboardInterrupt:
            print("\n\n⏹️ Interrompido")
        finally:
            self.running = False
            self.show_summary()
    
    def show_summary(self):
        """Resumo final"""
        if not self.trades:
            print("\nNenhum trade executado")
            return
            
        total = self.balance
        if self.position > 0 and self.market_data:
            total += self.position * float(self.market_data.get('price', 0))
        
        initial = getattr(self.config, 'INITIAL_BALANCE', 10000)
        profit = total - initial
        
        print(f"""
\n{'='*60}
📊 RESUMO
{'='*60}
Trades: {len(self.trades)}
Balance inicial: ${initial:,.2f}
Balance final: ${total:,.2f}
Lucro/Prejuízo: ${profit:,.2f} ({(profit/initial)*100:+.2f}%)
{'='*60}
        """)

async def main():
    try:
        system = IntegratedTradingSystem()
        await system.run()
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("\n💡 Use o sistema básico:")
        print("   python working_paper_trading.py")

if __name__ == "__main__":
    asyncio.run(main())
