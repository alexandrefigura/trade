#!/usr/bin/env python3
"""
Sistema Final Funcionando - Com todas as correções
"""
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

class FinalWorkingSystem:
    """Sistema corrigido e funcionando"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        if not self.api_key:
            raise ValueError("BINANCE_API_KEY não encontrada")
        
        print(f"✅ API Key: {self.api_key[:8]}...")
        
        # Config
        self.symbol = "BTCUSDT"
        self.initial_balance = 10000.0
        
        # Trading state
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.market_data = {}
        self.running = True
        
        # Análise técnica simples
        self.prices = []
        self.last_signal_time = None
        self.signal_cooldown = 60  # segundos entre sinais
        
        # Setup logging se disponível
        try:
            from trade_system.logging_config import setup_logging
            setup_logging()
        except:
            pass
    
    async def start_price_stream(self):
        """Stream de preços via HTTP"""
        import aiohttp
        print("📡 Conectando ao stream de preços...")
        
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
                                'change': float(data['priceChangePercent']),
                                'symbol': data['symbol']
                            }
                            
                            # Manter histórico de preços
                            self.prices.append(self.market_data['price'])
                            if len(self.prices) > 100:
                                self.prices.pop(0)
                                
                    except Exception as e:
                        print(f"⚠️ Erro stream: {e}")
                    
                    await asyncio.sleep(1)
        
        asyncio.create_task(fetch_loop())
        print("✅ Stream de preços iniciado")
    
    def analyze_market(self):
        """Análise de mercado simples mas efetiva"""
        if len(self.prices) < 20:
            return {'action': 'HOLD', 'confidence': 0, 'reasons': ['Dados insuficientes']}
        
        current_price = self.prices[-1]
        
        # Médias móveis
        sma_5 = np.mean(self.prices[-5:])
        sma_20 = np.mean(self.prices[-20:])
        
        # RSI simplificado
        changes = np.diff(self.prices[-15:])
        gains = changes[changes > 0].mean() if len(changes[changes > 0]) > 0 else 0
        losses = -changes[changes < 0].mean() if len(changes[changes < 0]) > 0 else 0
        
        if losses > 0:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100 if gains > 0 else 50
        
        # Volatilidade
        volatility = np.std(self.prices[-20:]) / np.mean(self.prices[-20:])
        
        # Decisão
        signal = {'action': 'HOLD', 'confidence': 0, 'reasons': []}
        
        # Check cooldown
        if self.last_signal_time:
            time_since_signal = (datetime.now() - self.last_signal_time).seconds
            if time_since_signal < self.signal_cooldown:
                return signal
        
        # Condições de compra
        if self.position == 0:
            buy_signals = 0
            
            if sma_5 > sma_20 * 1.002:  # Tendência de alta
                buy_signals += 1
                signal['reasons'].append('Tendência de alta (MA5 > MA20)')
            
            if rsi < 35:  # Oversold
                buy_signals += 1
                signal['reasons'].append(f'RSI oversold ({rsi:.1f})')
            
            if current_price < np.mean(self.prices[-50:]) * 0.98:  # Desconto
                buy_signals += 1
                signal['reasons'].append('Preço com desconto')
            
            if buy_signals >= 2 and volatility < 0.03:  # Baixa volatilidade
                signal['action'] = 'BUY'
                signal['confidence'] = buy_signals / 3
                self.last_signal_time = datetime.now()
        
        # Condições de venda
        elif self.position > 0:
            sell_signals = 0
            
            # Stop loss
            if current_price < self.entry_price * 0.98:
                signal['action'] = 'SELL'
                signal['reasons'] = ['Stop Loss 2%']
                signal['confidence'] = 1.0
                self.last_signal_time = datetime.now()
                return signal
            
            # Take profit
            if current_price > self.entry_price * 1.03:
                sell_signals += 1
                signal['reasons'].append('Take Profit 3%')
            
            if sma_5 < sma_20 * 0.998:  # Reversão
                sell_signals += 1
                signal['reasons'].append('Reversão de tendência')
            
            if rsi > 70:  # Overbought
                sell_signals += 1
                signal['reasons'].append(f'RSI overbought ({rsi:.1f})')
            
            if sell_signals >= 2:
                signal['action'] = 'SELL'
                signal['confidence'] = sell_signals / 3
                self.last_signal_time = datetime.now()
        
        return signal
    
    async def trading_loop(self):
        """Loop principal de trading"""
        while self.running:
            try:
                if not self.market_data or 'price' not in self.market_data:
                    await asyncio.sleep(1)
                    continue
                
                price = self.market_data['price']
                
                # Análise
                signal = self.analyze_market()
                
                # Executar trades
                if signal['action'] == 'BUY' and self.position == 0:
                    await self.execute_buy(price, signal)
                elif signal['action'] == 'SELL' and self.position > 0:
                    await self.execute_sell(price, signal)
                
                # Display
                self.display_status(price, signal)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"\n❌ Erro trading: {e}")
                await asyncio.sleep(5)
    
    async def execute_buy(self, price, signal):
        """Executa compra"""
        amount = self.balance * 0.95
        self.position = amount / price
        self.balance -= amount
        self.entry_price = price
        
        self.trades.append({
            'time': datetime.now(),
            'action': 'BUY',
            'price': price,
            'amount': self.position,
            'value': amount,
            'reasons': signal['reasons']
        })
        
        print(f"""
{'='*70}
📈 COMPRA EXECUTADA
   Preço: ${price:,.2f}
   Quantidade: {self.position:.8f} BTC
   Valor: ${amount:,.2f}
   Motivos: {', '.join(signal['reasons'])}
   Stop Loss: ${price * 0.98:,.2f} (-2%)
   Take Profit: ${price * 1.03:,.2f} (+3%)
{'='*70}
        """)
    
    async def execute_sell(self, price, signal):
        """Executa venda"""
        amount = self.position * price
        profit = amount - (self.position * self.entry_price)
        profit_pct = (profit / (self.position * self.entry_price)) * 100
        
        self.balance += amount
        
        self.trades.append({
            'time': datetime.now(),
            'action': 'SELL',
            'price': price,
            'amount': self.position,
            'value': amount,
            'profit': profit,
            'profit_pct': profit_pct,
            'reasons': signal['reasons']
        })
        
        print(f"""
{'='*70}
📉 VENDA EXECUTADA
   Preço: ${price:,.2f}
   Quantidade: {self.position:.8f} BTC
   Valor: ${amount:,.2f}
   Lucro/Prejuízo: ${profit:,.2f} ({profit_pct:+.2f}%)
   Motivos: {', '.join(signal['reasons'])}
{'='*70}
        """)
        
        self.position = 0
        self.entry_price = 0
    
    def display_status(self, price, signal):
        """Status display"""
        total = self.balance + (self.position * price if self.position > 0 else 0)
        pnl = total - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100
        
        status = f"💹 {self.symbol}: ${price:,.2f}"
        
        if 'change' in self.market_data:
            status += f" ({self.market_data['change']:+.1f}%)"
        
        status += f" | Balance: ${total:,.2f} ({pnl_pct:+.2f}%)"
        
        if self.position > 0:
            pos_pnl = ((price - self.entry_price) / self.entry_price) * 100
            status += f" | Pos: {pos_pnl:+.2f}%"
        
        status += f" | Trades: {len(self.trades)}"
        status += f" | {signal['action']}"
        
        print(f"\r{status}", end='', flush=True)
    
    async def run(self):
        """Executa o sistema"""
        print(f"""
{'='*70}
🤖 SISTEMA FINAL DE TRADING
{'='*70}
📊 Par: {self.symbol}
💰 Balance: ${self.initial_balance:,.2f}
⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📈 Estratégia: MA Crossover + RSI + Risk Management
🛡️ Risk: Stop Loss 2%, Take Profit 3%
⏱️ Cooldown entre sinais: {self.signal_cooldown}s
{'='*70}
        """)
        
        # Iniciar streams
        await self.start_price_stream()
        await asyncio.sleep(2)
        
        # Trading
        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            print("\n\n⏹️ Sistema interrompido")
        finally:
            self.running = False
            self.show_summary()
    
    def show_summary(self):
        """Resumo final"""
        if not self.trades:
            print("\n📊 Nenhum trade executado")
            return
        
        total = self.balance
        if self.position > 0 and self.market_data:
            total += self.position * self.market_data.get('price', self.entry_price)
        
        profit = total - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        # Estatísticas
        buys = [t for t in self.trades if t['action'] == 'BUY']
        sells = [t for t in self.trades if t['action'] == 'SELL']
        
        wins = len([t for t in sells if t.get('profit', 0) > 0])
        losses = len([t for t in sells if t.get('profit', 0) <= 0])
        
        print(f"""
{'='*70}
📊 RESUMO FINAL
{'='*70}
Período: {self.trades[0]['time'].strftime('%H:%M')} - {datetime.now().strftime('%H:%M')}
Total de trades: {len(self.trades)} ({len(buys)} compras, {len(sells)} vendas)
Taxa de acerto: {(wins/(wins+losses)*100) if sells else 0:.1f}% ({wins}W/{losses}L)

Balance inicial: ${self.initial_balance:,.2f}
Balance final: ${total:,.2f}
Lucro/Prejuízo: ${profit:,.2f} ({profit_pct:+.2f}%)

Melhor trade: ${max((t.get('profit', 0) for t in sells), default=0):,.2f}
Pior trade: ${min((t.get('profit', 0) for t in sells), default=0):,.2f}
{'='*70}
        """)

async def main():
    system = FinalWorkingSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())
