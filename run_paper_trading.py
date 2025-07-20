"""Sistema de Paper Trading - Execução Principal"""
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
        
        print(f"\n{'='*60}")
        print(f"🤖 ULTRA TRADING BOT - PAPER TRADING")
        print(f"{'='*60}")
        print(f"📊 Par: {config.symbol}")
        print(f"💰 Balance Inicial: ${config.base_balance:,.2f}")
        print(f"🕐 Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Criar e iniciar sistema
        system = TradingSystem(config, paper_trading=True)
        
        print("🚀 Iniciando sistema de trading...\n")
        
        await system.start()
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        
        # Log detalhado do erro
        logging.error("Erro detalhado:", exc_info=True)
        
        print("\n💡 Sugestões:")
        print("1. Verifique se todas as dependências estão instaladas:")
        print("   pip install -r requirements.txt")
        print("2. Execute o sistema mínimo para testar:")
        print("   python minimal_trading_system.py")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Sistema encerrado pelo usuário")
        print("📊 Logs salvos em: logs/")
