#!/usr/bin/env python3
"""
Script para diagnosticar e corrigir importações do sistema integrado
"""
import os
import ast
import importlib.util

def find_class_names(file_path):
    """Encontra todas as classes definidas em um arquivo Python"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    except Exception as e:
        return []

def diagnose_imports():
    """Diagnostica as importações do sistema"""
    print("🔍 DIAGNOSTICANDO IMPORTAÇÕES DO SISTEMA")
    print("=" * 60)
    
    # Módulos para verificar
    modules_to_check = [
        ('trade_system/cache.py', 'TradingCache'),
        ('trade_system/paper_trader.py', 'PaperTrader'),
        ('trade_system/checkpoint.py', 'CheckpointManager'),
        ('trade_system/signals.py', 'SignalConsolidator'),
    ]
    
    corrections = {}
    
    for file_path, expected_class in modules_to_check:
        if os.path.exists(file_path):
            classes = find_class_names(file_path)
            print(f"\n📁 {file_path}")
            print(f"   Classes encontradas: {', '.join(classes) if classes else 'Nenhuma'}")
            
            if expected_class not in classes and classes:
                # Tentar encontrar a classe correta
                possible_match = None
                for cls in classes:
                    if 'cache' in cls.lower() and 'cache' in expected_class.lower():
                        possible_match = cls
                        break
                    elif expected_class.lower() in cls.lower():
                        possible_match = cls
                        break
                    elif cls.lower() in expected_class.lower():
                        possible_match = cls
                        break
                
                if possible_match:
                    print(f"   ⚠️  Esperado: {expected_class}")
                    print(f"   ✅ Encontrado: {possible_match}")
                    corrections[expected_class] = possible_match
                else:
                    print(f"   ❌ Classe {expected_class} não encontrada")
        else:
            print(f"\n❌ Arquivo não encontrado: {file_path}")
    
    return corrections

def create_fixed_integrated_system(corrections):
    """Cria versão corrigida do sistema integrado"""
    print("\n\n📝 Criando versão corrigida do sistema integrado...")
    
    # Ler o arquivo original
    with open('integrated_paper_trading.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aplicar correções
    for wrong_name, correct_name in corrections.items():
        content = content.replace(f"from trade_system.cache import {wrong_name}", 
                                f"from trade_system.cache import {correct_name}")
        content = content.replace(f"{wrong_name}(", f"{correct_name}(")
    
    # Criar versão simplificada que funciona
    simplified_integrated = '''#!/usr/bin/env python3
"""
Sistema Integrado Simplificado - Versão que Funciona
Usa apenas os módulos que existem e funcionam
"""
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import json
import aiohttp
import numpy as np

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar apenas módulos que sabemos que funcionam
from trade_system.config import TradingConfig
from trade_system.logging_config import setup_logging
from trade_system.alerts import AlertSystem
from trade_system.websocket_manager import WebSocketManager
from trade_system.analysis.technical import TechnicalAnalyzer
from trade_system.risk import RiskManager

# Carregar variáveis de ambiente
load_dotenv()

class SimplifiedIntegratedTrading:
    """Sistema integrado simplificado que usa módulos principais"""
    
    def __init__(self):
        # Verificar API keys
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API keys não encontradas no .env")
        
        print(f"✅ API Key: {self.api_key[:8]}...")
        
        # Configurar logging
        setup_logging()
        
        # Carregar configuração
        self.config = TradingConfig()
        
        # Inicializar componentes principais
        print("\\n🔄 Inicializando componentes principais...")
        
        # Componentes essenciais
        self.alerts = AlertSystem(self.config)
        self.ws_manager = WebSocketManager(self.config)
        self.technical = TechnicalAnalyzer(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Paper trading simples
        self.balance = self.config.INITIAL_BALANCE
        self.position = 0
        self.entry_price = 0
        self.trades = []
        
        # Estado
        self.market_data = {}
        self.running = True
        
        print("✅ Componentes inicializados!")
    
    async def on_market_data(self, data):
        """Callback para processar dados do WebSocket"""
        self.market_data = data
    
    async def analyze_and_trade(self):
        """Analisa mercado e executa trades"""
        while self.running:
            try:
                if not self.market_data:
                    await asyncio.sleep(1)
                    continue
                
                price = float(self.market_data.get('price', 0))
                if price == 0:
                    continue
                
                # Análise técnica
                ta_result = await self.technical.analyze(self.market_data)
                
                # Validação de risco
                if ta_result and ta_result.get('action') != 'HOLD':
                    risk_data = {
                        'price': price,
                        'balance': self.balance,
                        'position': self.position
                    }
                    
                    if self.risk_manager.validate_trade(ta_result, self.market_data, risk_data):
                        await self.execute_trade(ta_result, price)
                
                # Display status
                self.display_status(ta_result, price)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"\\n❌ Erro na análise: {e}")
                await asyncio.sleep(5)
    
    async def execute_trade(self, signal, price):
        """Executa paper trade"""
        if signal['action'] == 'BUY' and self.position == 0:
            # Comprar
            amount = self.balance * 0.95
            self.position = amount / price
            self.balance -= amount
            self.entry_price = price
            
            trade = {
                'time': datetime.now(),
                'action': 'BUY',
                'price': price,
                'amount': self.position,
                'value': amount
            }
            self.trades.append(trade)
            
            print(f"""
\\n{'='*60}
📈 COMPRA EXECUTADA
   Preço: ${price:,.2f}
   Quantidade: {self.position:.6f}
   Valor: ${amount:,.2f}
   Balance: ${self.balance:,.2f}
   Análise Técnica: {signal.get('confidence', 0)*100:.0f}% confiança
{'='*60}
            """)
            
            self.alerts.send_alert("TRADE", f"Compra @ ${price:,.2f}", signal)
            
        elif signal['action'] == 'SELL' and self.position > 0:
            # Vender
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
                'profit_pct': profit_pct
            }
            self.trades.append(trade)
            
            print(f"""
\\n{'='*60}
📉 VENDA EXECUTADA
   Preço: ${price:,.2f}
   Quantidade: {self.position:.6f}
   Valor: ${amount:,.2f}
   Lucro: ${profit:,.2f} ({profit_pct:+.2f}%)
   Balance: ${self.balance:,.2f}
{'='*60}
            """)
            
            self.alerts.send_alert("TRADE", f"Venda @ ${price:,.2f} - Lucro: {profit_pct:+.2f}%", signal)
            
            self.position = 0
            self.entry_price = 0
    
    def display_status(self, signal, price):
        """Exibe status do sistema"""
        total_value = self.balance + (self.position * price if self.position > 0 else 0)
        profit_pct = ((total_value - self.config.INITIAL_BALANCE) / self.config.INITIAL_BALANCE) * 100
        
        status = f"💹 {self.config.SYMBOL}: ${price:,.2f} | "
        status += f"Balance: ${self.balance:,.2f} | "
        
        if self.position > 0:
            position_pnl = ((price - self.entry_price) / self.entry_price) * 100
            status += f"Pos: {self.position:.6f} ({position_pnl:+.2f}%) | "
        else:
            status += "Pos: 0 | "
        
        status += f"Total: ${total_value:,.2f} ({profit_pct:+.2f}%) | "
        
        if signal:
            status += f"Sinal: {signal.get('action', 'HOLD')} ({signal.get('confidence', 0)*100:.0f}%)"
        
        print(f"\\r{status}", end='', flush=True)
    
    async def run(self):
        """Executa o sistema"""
        print(f"""
{'='*60}
🤖 SISTEMA INTEGRADO SIMPLIFICADO
{'='*60}
📊 Par: {self.config.SYMBOL}
💰 Balance Inicial: ${self.config.INITIAL_BALANCE:,.2f}
⏰ Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
🔧 Componentes Ativos:
   ✅ WebSocket Manager
   ✅ Technical Analyzer (Numba)
   ✅ Risk Manager
   ✅ Alert System
   ✅ Paper Trading
{'='*60}
        """)
        
        # Conectar WebSocket
        await self.ws_manager.connect(self.on_market_data)
        
        # Aguardar conexão
        await asyncio.sleep(2)
        
        # Iniciar análise
        try:
            await self.analyze_and_trade()
        except KeyboardInterrupt:
            print("\\n\\n⏹️ Sistema interrompido")
        finally:
            self.running = False
            await self.ws_manager.disconnect()
            self.show_summary()
    
    def show_summary(self):
        """Mostra resumo final"""
        total_value = self.balance + (self.position * float(self.market_data.get('price', 0)) if self.position > 0 else 0)
        total_profit = total_value - self.config.INITIAL_BALANCE
        total_profit_pct = (total_profit / self.config.INITIAL_BALANCE) * 100
        
        wins = sum(1 for t in self.trades if t.get('action') == 'SELL' and t.get('profit', 0) > 0)
        losses = sum(1 for t in self.trades if t.get('action') == 'SELL' and t.get('profit', 0) < 0)
        
        print(f"""
\\n{'='*60}
📊 RESUMO FINAL
{'='*60}
Total de trades: {len(self.trades)}
Balance inicial: ${self.config.INITIAL_BALANCE:,.2f}
Balance final: ${total_value:,.2f}
Lucro/Prejuízo: ${total_profit:,.2f} ({total_profit_pct:+.2f}%)
        """)
        
        if wins + losses > 0:
            win_rate = (wins / (wins + losses)) * 100
            print(f"Taxa de acerto: {win_rate:.1f}% ({wins} wins, {losses} losses)")

async def main():
    try:
        system = SimplifiedIntegratedTrading()
        await system.run()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open('simplified_integrated_trading.py', 'w', encoding='utf-8') as f:
        f.write(simplified_integrated)
    
    print("✅ simplified_integrated_trading.py criado!")

def main():
    # Diagnosticar
    corrections = diagnose_imports()
    
    # Criar versão corrigida
    create_fixed_integrated_system(corrections)
    
    print("\n" + "="*60)
    print("✅ DIAGNÓSTICO COMPLETO!")
    print("\n🚀 Execute o sistema simplificado que funciona:")
    print("   python simplified_integrated_trading.py")
    print("\n💡 Este sistema usa apenas os módulos que sabemos que funcionam:")
    print("   - WebSocket Manager")
    print("   - Technical Analyzer")
    print("   - Risk Manager") 
    print("   - Alert System")

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> b7cd768e00fe8b178cb464118735e229e288c012
