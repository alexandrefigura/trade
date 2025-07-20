import os
import re

print("🔧 CORRIGINDO ARGUMENTOS DO WebSocketManager")
print("=" * 60)

# 1. Verificar main.py
main_file = 'trade_system/main.py'

if os.path.exists(main_file):
    print("📝 Corrigindo main.py...")
    
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Encontrar a linha problemática
    if 'WebSocketManager(self.config, self.cache)' in content:
        # Remover o segundo argumento
        content = content.replace(
            'WebSocketManager(self.config, self.cache)',
            'WebSocketManager(self.config)'
        )
        print("✅ Removido argumento extra do WebSocketManager")
    
    # Salvar
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ main.py corrigido!")

# 2. Verificar websocket_manager.py
ws_file = 'trade_system/websocket_manager.py'

if os.path.exists(ws_file):
    print("\n📝 Verificando websocket_manager.py...")
    
    with open(ws_file, 'r', encoding='utf-8') as f:
        ws_content = f.read()
    
    # Verificar assinatura do __init__
    init_match = re.search(r'def __init__\(self[^)]*\):', ws_content)
    if init_match:
        print(f"   Assinatura atual: {init_match.group(0)}")

# 3. Criar script de execução melhorado
print("\n📝 Criando script de execução melhorado...")

run_script = '''"""Sistema de Paper Trading - Execução Principal"""
import asyncio
import os
import sys
import logging
from datetime import datetime

# Carregar variáveis de ambiente
from dotenv import load_dotenv
load_dotenv()

# Verificar API Key
api_key = os.getenv('BINANCE_API_KEY')
if not api_key:
    print("❌ Configure BINANCE_API_KEY no arquivo .env")
    sys.exit(1)

print(f"✅ Sistema configurado com API Key: {api_key[:8]}...")

async def main():
    try:
        # Imports
        from trade_system.config import TradingConfig
        from trade_system.logging_config import setup_logging
        from trade_system.main import TradingSystem
        
        # Configurar logging
        setup_logging("INFO")
        
        # Criar configuração
        config = TradingConfig.from_env()
        
        print(f"\\n{'='*60}")
        print(f"🤖 ULTRA TRADING BOT - PAPER TRADING")
        print(f"{'='*60}")
        print(f"📊 Par: {config.symbol}")
        print(f"💰 Balance Inicial: ${config.base_balance:,.2f}")
        print(f"🕐 Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\\n")
        
        # Criar e iniciar sistema
        system = TradingSystem(config, paper_trading=True)
        
        print("🚀 Iniciando sistema de trading...\\n")
        
        await system.start()
        
    except Exception as e:
        print(f"\\n❌ Erro: {e}")
        
        # Log detalhado do erro
        logging.error("Erro detalhado:", exc_info=True)
        
        print("\\n💡 Sugestões:")
        print("1. Verifique se todas as dependências estão instaladas:")
        print("   pip install -r requirements.txt")
        print("2. Execute o sistema mínimo para testar:")
        print("   python minimal_trading_system.py")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n\\n👋 Sistema encerrado pelo usuário")
        print("📊 Logs salvos em: logs/")
'''

with open('run_paper_trading.py', 'w', encoding='utf-8') as f:
    f.write(run_script)

print("✅ run_paper_trading.py criado!")

# 4. Verificar se há mais problemas potenciais
print("\n🔍 Verificando outros possíveis problemas...")

# Verificar se TradingSystem espera paper_trading como kwarg
if os.path.exists(main_file):
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar definição de TradingSystem
    class_match = re.search(r'class TradingSystem.*?def __init__\([^)]+\):', content, re.DOTALL)
    if class_match:
        init_line = class_match.group(0)
        if 'paper_trading' in init_line:
            print("✅ TradingSystem aceita paper_trading")
        else:
            print("⚠️ TradingSystem pode não aceitar paper_trading como parâmetro")

print("\n✅ Correções aplicadas!")
print("\n🚀 Execute:")
print("   python run_paper_trading.py")
print("\n   Se ainda houver erros, use:")
print("   python minimal_trading_system.py")
