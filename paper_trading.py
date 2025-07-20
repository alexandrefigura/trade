"""Execução do Paper Trading"""
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
print("🚀 Iniciando Paper Trading...\n")

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
        print("\nVerifique se todos os módulos estão instalados:")
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
        print("\n\n👋 Sistema encerrado pelo usuário")
