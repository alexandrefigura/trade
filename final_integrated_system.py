#!/usr/bin/env python3
"""
Sistema Integrado Final - Com todas as correções
Usa os nomes corretos das classes encontradas
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

# Carregar env
load_dotenv()

class FinalIntegratedTrading:
    """Sistema integrado com nomes de classes corretos"""
    
    def __init__(self):
        # API keys
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key:
            raise ValueError("BINANCE_API_KEY não encontrada no .env")
        
        print(f"✅ API Key: {self.api_key[:8]}...")
        
        # Configuração padrão
        self.symbol = "BTCUSDT"
        self.initial_balance = 10000.0
        
        # Tentar carregar config se existir
        try:
            from trade_system.config import TradingConfig
            from trade_system.logging_config import setup_logging
            setup_logging()
            self.config = TradingConfig()
            # Usar atributos se existirem
            self.symbol = getattr(self.config, 'symbol', self.symbol)
            if not self.symbol:
                self.symbol = "BTCUSDT"
            self.initial_balance = getattr(self.config, 'initial_balance', self.initial_balance)
            if not self.initial_balance:
                self.initial_balance = 10000.0
        except Exception as e:
            print(f"⚠️ Usando configuração padrão: {e}")
            self.config = None
        
        # Componentes
        self.components = {}
        self._load_components()
        
        # Paper trading state
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.market_data = {}
        self.running = True
    
    def _load_components(self):
        """Carrega componentes com nomes corretos"""
        print("\n🔄 Carregando componentes...")
        
        # Componentes com nomes CORRETOS baseados no diagnóstico
        components_to_try = [
            ('cache', 'trade_system.cache', 'CacheManager'),  # Corrigido
            ('alerts', 'trade_system.alerts', 'AlertManager'),  # Corrigido
            ('websocket', 'trade_system.websocket_manager', 'WebSocketManager'),
            ('technical', 'trade_system.analysis.technical', 'TechnicalAnalyzer'),
            ('ml', 'trade_system.analysis.ml', 'MLPredictor'),  # Corrigido
            ('orderbook', 'trade_system.analysis.orderbook', 'OrderbookAnalyzer'),
            ('risk', 'trade_system.risk', 'RiskManager'),
            ('paper_trader', 'trade_system.paper_trader', 'PaperTrader'),
        ]
        
        for name, module_path, class_name in components_to_try:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                
                # Instanciar com ou sem config
                if self.config:
                    try:
                        self.components[name] = cls(self.config)
                    except:
                        self.components[name] = cls()
                else:
                    self.components[name] = cls()
                    
                print(f"   ✅ {name}: {class_name}")
            except Exception as e:
                print(f"   ⚠️ {name}: {str(e)[:50]}...")
    
    async def connect_data_source(self):
        """Conecta ao websocket ou usa HTTP"""
        if 'websocket' in self.components:
            try:
                async def on_data(data):
                    self.market_data = data
                    # Cache se disponível
                    if 'cache' in self.components:
                        try:
                            await self.components['cache'].set(
                                f"price:{self.symbol}", 
                                data.get('price', 0),
                                ttl=60
                            )
                        except:
                            pass
                
                await self.components['websocket'].connect(on_data)
                print("✅ WebSocket conectado")
                return
            except Exception as e:
                print(f"⚠️ WebSocket falhou: {e}")
        
        # Fallback HTTP
        await self.start_http_stream()
    
    async def start_http_stream(self):
        """Stream de preços via HTTP"""
        import aiohttp
        print("📡 Usando stream HTTP...")
        
        async def fetch_loop():
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={self.symbol}"
            async with aiohttp.ClientSession() as session:
                while self.running:
                    try:
                        async with session.get(url) as resp:
                            data = await resp.json()
                            self.market_data = {
                                'price': float(data['lastPrice']),
                                'volume': float(data['volume']),
                                'high': float(data['highPrice']),
                                'low': float(data['lowPrice']),
                                'symbol': data['symbol']
                            }
                    except Exception as e:
                        print(f"⚠️ Erro HTTP: {e}")
                    
                    await asyncio.sleep(1)
        
        asyncio.create_task(fetch_loop())
    
    async def trading_loop(self):
        """Loop principal de trading"""
        while self.running:
            try:
                if not self.market_data or 'price' not in self.market_data:
                    await asyncio.sleep(1)
                    continue
                
                price = float(self.market_data['price'])
                
                # Análise e sinal
                signal = await self.analyze_market()
                
                # Executar trade se necessário
                if signal['action'] == 'BUY' and self.position == 0:
                    await self.execute_buy(price, signal)
                elif signal['action'] == 'SELL' and self.position > 0:
                    await self.execute_sell(price, signal)
                
                # Display status
                self.display_status(price, signal)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"\n❌ Erro no loop: {e}")
                await asyncio.sleep(5)
    
    async def analyze_market(self):
        """Análise usando componentes disponíveis"""
        signal = {'action': 'HOLD', 'confidence': 0, 'reasons': []}
        
        # Análise técnica
        if 'technical' in self.components:
            try:
                ta_result = await self.components['technical'].analyze(self.market_data)
                if ta_result:
                    signal = ta_result
            except Exception as e:
                print(f"⚠️ Erro TA: {e}")
        
        # ML prediction
        if 'ml' in self.components and signal['action'] == 'HOLD':
            try:
                ml_result = self.components['ml'].predict(self.market_data)
                if ml_result and ml_result.get('confidence', 0) > 0.7:
                    signal = ml_result
            except Exception as e:
                print(f"⚠️ Erro ML: {e}")
        
        # Validação de risco
        if signal['action'] != 'HOLD' and 'risk' in self.components:
            try:
                risk_data = {
                    'balance': self.balance,
                    'position': self.position,
                    'price': float(self.market_data['price'])
                }
                if not self.components['risk'].validate_trade(signal, self.market_data, risk_data):
                    signal['action'] = 'HOLD'
                    signal['reasons'].append('Bloqueado pelo Risk Manager')
            except Exception as e:
                print(f"⚠️ Erro risk: {e}")
        
        return signal
    
    async def execute_buy(self, price, signal):
        """Executa compra"""
        amount = self.balance * 0.95
        self.position = amount / price
        self.balance -= amount
        self.entry_price = price
        
        trade = {
            'time': datetime.now(),
            'action': 'BUY',
            'price': price,
            'amount': self.position,
            'value': amount,
            'reasons': signal.get('reasons', [])
        }
        self.trades.append(trade)
        
        print(f"""
{'='*60}
📈 COMPRA EXECUTADA
   Preço: ${price:,.2f}
   Quantidade: {self.position:.6f}
   Valor: ${amount:,.2f}
   Balance: ${self.balance:,.2f}
   Motivos: {', '.join(signal.get('reasons', ['Sinal técnico']))}
{'='*60}
        """)
        
        # Alertar se disponível
        if 'alerts' in self.components:
            try:
                self.components['alerts'].send_alert(
                    "TRADE",
                    f"Compra @ ${price:,.2f}",
                    signal
                )
            except:
                pass
    
    async def execute_sell(self, price, signal):
        """Executa venda"""
        amount = self.position * price
        profit = amount - (self.position * self.entry_price)
        profit_pct = (profit / (self.position * self.entry_price)) * 100
        
        self.balance += amount
        
        trade = {
            'time': datetime.now(),
            'action': 'SELL',
            'price': price,
            'amount': self.position,
            'value': amount,
            'profit': profit,
            'profit_pct': profit_pct,
            'reasons': signal.get('reasons', [])
        }
        self.trades.append(trade)
        
        print(f"""
{'='*60}
📉 VENDA EXECUTADA
   Preço: ${price:,.2f}
   Quantidade: {self.position:.6f}
   Valor: ${amount:,.2f}
   Lucro: ${profit:,.2f} ({profit_pct:+.2f}%)
   Balance: ${self.balance:,.2f}
   Motivos: {', '.join(signal.get('reasons', ['Take profit']))}
{'='*60}
        """)
        
        self.position = 0
        self.entry_price = 0
    
    def display_status(self, price, signal):
        """Mostra status atual"""
        total = self.balance + (self.position * price if self.position > 0 else 0)
        pnl = total - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100
        
        status = f"💹 {self.symbol}: ${price:,.2f} | "
        
        if 'volume' in self.market_data:
            status += f"Vol: {self.market_data['volume']:,.0f} | "
        
        status += f"Balance: ${total:,.2f} ({pnl_pct:+.2f}%) | "
        
        if self.position > 0:
            pos_pnl = ((price - self.entry_price) / self.entry_price) * 100
            status += f"Pos: {pos_pnl:+.2f}% | "
        
        status += f"Trades: {len(self.trades)} | "
        status += f"Sinal: {signal['action']}"
        
        print(f"\r{status}", end='', flush=True)
    
    async def run(self):
        """Executa o sistema"""
        print(f"""
{'='*60}
🤖 SISTEMA INTEGRADO FINAL
{'='*60}
📊 Par: {self.symbol}
💰 Balance: ${self.initial_balance:,.2f}
⏰ Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔧 Componentes: {', '.join(self.components.keys())}
{'='*60}
        """)
        
        # Conectar dados
        await self.connect_data_source()
        await asyncio.sleep(2)
        
        # Trading loop
        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            print("\n\n⏹️ Sistema interrompido")
        finally:
            self.running = False
            self.show_summary()
    
    def show_summary(self):
        """Resumo final detalhado"""
        if not self.trades:
            print("\n📊 Nenhum trade executado")
            return
        
        # Calcular estatísticas
        total = self.balance
        if self.position > 0 and self.market_data:
            total += self.position * float(self.market_data.get('price', self.entry_price))
        
        profit = total - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        # Wins/Losses
        sells = [t for t in self.trades if t['action'] == 'SELL']
        wins = len([t for t in sells if t.get('profit', 0) > 0])
        losses = len([t for t in sells if t.get('profit', 0) <= 0])
        
        print(f"""
{'='*60}
📊 RESUMO FINAL
{'='*60}
Total de trades: {len(self.trades)}
Trades fechados: {len(sells)}
Taxa de acerto: {(wins/(wins+losses)*100) if sells else 0:.1f}% ({wins}W/{losses}L)

Balance inicial: ${self.initial_balance:,.2f}
Balance final: ${total:,.2f}
Lucro/Prejuízo: ${profit:,.2f} ({profit_pct:+.2f}%)

Componentes usados: {', '.join(self.components.keys())}
{'='*60}
        """)

async def main():
    try:
        system = FinalIntegratedTrading()
        await system.run()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Se houver erros, use:")
        print("   python working_paper_trading.py")

if __name__ == "__main__":
    asyncio.run(main())
