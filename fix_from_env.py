import os
import re

print("🔧 CORRIGINDO MÉTODO from_env()")
print("=" * 60)

# 1. Corrigir config.py
config_file = 'trade_system/config.py'

if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📝 Atualizando método from_env()...")
    
    # Procurar o método from_env
    pattern = r'@classmethod\s+def from_env\(cls\):'
    
    if re.search(pattern, content):
        # Substituir para aceitar debug_mode
        content = re.sub(
            r'def from_env\(cls\):',
            'def from_env(cls, debug_mode=False):',
            content
        )
        print("✅ Parâmetro debug_mode adicionado")
    else:
        print("⚠️ Método from_env não encontrado no formato esperado")
    
    # Salvar
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ config.py atualizado!")

# 2. Corrigir main.py para não passar debug_mode se não for necessário
main_file = 'trade_system/main.py'

if os.path.exists(main_file):
    print("\n📝 Verificando main.py...")
    
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar a linha problemática
    if 'TradingConfig.from_env(debug_mode=debug_mode)' in content:
        # Verificar se o método aceita o parâmetro
        # Por hora, vamos remover o parâmetro
        content = content.replace(
            'TradingConfig.from_env(debug_mode=debug_mode)',
            'TradingConfig.from_env()'
        )
        
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ main.py corrigido!")

# 3. Criar novo wrapper
print("\n📝 Criando novo wrapper...")

wrapper = '''"""Execução do Paper Trading"""
import os
import sys
import asyncio
import logging

# Carregar .env
from dotenv import load_dotenv
load_dotenv()

# Verificar
api_key = os.getenv('BINANCE_API_KEY')
if not api_key:
    print("❌ BINANCE_API_KEY não encontrada no .env!")
    sys.exit(1)

print(f"✅ API Key: {api_key[:8]}...")
print("🚀 Iniciando Paper Trading...\\n")

# Configurar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    try:
        # Importar após configurar path
        from trade_system.config import TradingConfig
        from trade_system.logging_config import setup_logging
        from trade_system.main import TradingSystem
        
        # Setup
        setup_logging("INFO")
        
        # Criar config
        config = TradingConfig.from_env()
        
        # Verificar config
        if not config.api_key:
            print("❌ API Key não carregada!")
            return
        
        print(f"📊 Par: {config.symbol}")
        print(f"💰 Balance: ${config.base_balance:,.2f}")
        print()
        
        # Criar e executar sistema
        system = TradingSystem(config, paper_trading=True)
        await system.start()
        
    except ImportError as e:
        print(f"❌ Erro de import: {e}")
        print("\\nVerifique se todos os módulos estão instalados:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     PAPER TRADING SYSTEM                             ║
║                   Sistema de Trading Simulado                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n\\n👋 Sistema encerrado pelo usuário")
'''

with open('paper_trading.py', 'w', encoding='utf-8') as f:
    f.write(wrapper)

print("✅ paper_trading.py criado!")

# 4. Testar se as correções funcionaram
print("\n🧪 Testando correções...")

try:
    # Recarregar módulos
    if 'trade_system.config' in sys.modules:
        del sys.modules['trade_system.config']
    
    from trade_system.config import TradingConfig
    
    # Testar from_env
    config = TradingConfig.from_env()
    
    if hasattr(config, 'api_key') and config.api_key:
        print(f"✅ Teste bem sucedido! API Key: {config.api_key[:8]}...")
    else:
        print("⚠️ Config criada mas API Key não carregada")
        
except Exception as e:
    print(f"❌ Erro no teste: {e}")

print("\n🚀 Execute: python paper_trading.py")
print("   ou: python minimal_trading_system.py")
