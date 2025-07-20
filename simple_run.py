"""Execução simplificada do Paper Trading"""
import asyncio
import os
import sys
import logging

# Configurar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

async def main():
    try:
        # Carregar config
        from trade_system.config import TradingConfig
        config = TradingConfig.from_env()
        
        if not config.api_key:
            print("❌ Configure BINANCE_API_KEY no .env")
            return
        
        print(f"✅ Configurado para {config.symbol}")
        print(f"💰 Balance inicial: ${config.base_balance:,.2f}")
        
        # Criar sistema
        from trade_system.main import TradingSystem
        system = TradingSystem(config, paper_trading=True)
        
        # Executar
        await system.start()
        
    except ImportError as e:
        print(f"❌ Erro de import: {e}")
        print("\nVerifique se todos os módulos estão corretos")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🤖 ULTRA TRADING BOT - PAPER TRADING")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Sistema encerrado")
