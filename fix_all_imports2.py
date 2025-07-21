#!/usr/bin/env python3
"""
Script para corrigir TODOS os problemas de importação e nomes de classes
"""
import os
import re

def fix_all_issues():
    print("🔧 CORRIGINDO TODOS OS PROBLEMAS DE IMPORTAÇÃO")
    print("=" * 60)
    
    # 1. Corrigir main.py
    main_path = 'trade_system/main.py'
    if os.path.exists(main_path):
        print(f"📝 Corrigindo {main_path}...")
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Backup
        with open(f"{main_path}.backup", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Corrigir o erro SimplifiedWebSocketManager -> SimplifiedMLPredictor
        content = re.sub(
            r'self\.ml_predictor\s*=\s*SimplifiedWebSocketManager\(\)',
            'self.ml_predictor = SimplifiedMLPredictor()',
            content
        )
        
        # Garantir que as importações estão corretas
        if 'from trade_system.analysis.ml import SimplifiedMLPredictor' not in content:
            # Adicionar import se não existir
            import_section = re.search(r'(from trade_system\..*\n)+', content)
            if import_section:
                end_pos = import_section.end()
                content = (content[:end_pos] + 
                          'from trade_system.analysis.ml import SimplifiedMLPredictor\n' + 
                          content[end_pos:])
        
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ main.py corrigido!")
    
    # 2. Criar um sistema de trading mínimo que funciona
    print("\n📝 Criando sistema de trading funcional...")
    
    working_system = '''#!/usr/bin/env python3
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
                    
                    print(f"\\r💹 {self.symbol}: ${self.websocket.price:,.2f} | "
                          f"Balance: ${self.balance:,.2f} | "
                          f"Posição: {self.position:.6f} | "
                          f"Total: ${total_value:,.2f} ({profit_pct:+.2f}%) | "
                          f"Sinal: {signal}", end='', flush=True)
                
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\\n\\n⏹️ Sistema interrompido pelo usuário")
        finally:
            # Parar websocket
            self.websocket.stop()
            
            # Mostrar resumo
            self.show_summary()
            
    def show_summary(self):
        """Mostra resumo das operações"""
        if not self.trades:
            print("\\n📊 Nenhuma operação realizada")
            return
            
        print(f"""
\\n{'=' * 50}
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
'''
    
    with open('working_paper_trading.py', 'w', encoding='utf-8') as f:
        f.write(working_system)
    print("✅ working_paper_trading.py criado!")
    
    # 3. Criar script de diagnóstico
    diagnostic = '''#!/usr/bin/env python3
"""
Diagnóstico do Sistema de Trading
"""
import os
import importlib.util
import sys

def check_module(module_path, class_name):
    """Verifica se um módulo e classe existem"""
    try:
        if not os.path.exists(module_path):
            return f"❌ Arquivo não encontrado: {module_path}"
            
        # Carregar módulo
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Verificar classe
        if hasattr(module, class_name):
            return f"✅ {class_name} encontrado em {module_path}"
        else:
            # Listar classes disponíveis
            classes = [name for name in dir(module) if name[0].isupper()]
            return f"❌ {class_name} não encontrado. Classes disponíveis: {', '.join(classes)}"
            
    except Exception as e:
        return f"❌ Erro ao carregar {module_path}: {str(e)}"

print("🔍 DIAGNÓSTICO DO SISTEMA")
print("=" * 60)

# Verificar estrutura de diretórios
print("\\n📁 Estrutura de diretórios:")
for root, dirs, files in os.walk('trade_system'):
    level = root.replace('trade_system', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        if file.endswith('.py'):
            print(f'{subindent}{file}')

# Verificar módulos críticos
print("\\n🔧 Verificando módulos críticos:")
checks = [
    ('trade_system/analysis/technical.py', 'TechnicalAnalyzer'),
    ('trade_system/analysis/ml.py', 'SimplifiedMLPredictor'),
    ('trade_system/analysis/orderbook.py', 'OrderbookAnalyzer'),
    ('trade_system/websocket_manager.py', 'WebSocketManager'),
    ('trade_system/risk.py', 'RiskManager'),
]

for module_path, class_name in checks:
    print(check_module(module_path, class_name))

print("\\n✅ Diagnóstico completo!")
print("\\n🚀 Para executar o sistema funcional:")
print("   python working_paper_trading.py")
'''
    
    with open('diagnose_system.py', 'w', encoding='utf-8') as f:
        f.write(diagnostic)
    print("✅ diagnose_system.py criado!")
    
    print("\n" + "=" * 60)
    print("✅ CORREÇÕES APLICADAS!")
    print("\n🚀 Execute na seguinte ordem:")
    print("   1. python diagnose_system.py  (para verificar o sistema)")
    print("   2. python working_paper_trading.py  (sistema funcional)")
    print("\n💡 O sistema funcional:")
    print("   - Conecta ao WebSocket da Binance em tempo real")
    print("   - Usa análise técnica simples (médias móveis)")
    print("   - Executa paper trading automático")
    print("   - Mostra resumo das operações")

if __name__ == "__main__":
    fix_all_issues()
