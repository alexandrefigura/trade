"""Paper Trading - Versão Final Corrigida"""
import asyncio
import os
import sys
import logging
from datetime import datetime

# Carregar variáveis de ambiente
from dotenv import load_dotenv
load_dotenv()

# Verificar credenciais
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key:
    print("❌ Configure BINANCE_API_KEY no arquivo .env")
    sys.exit(1)

print(f"✅ API Key: {api_key[:8]}...")

# Configurar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    try:
        print("\n🔄 Carregando sistema...")
        
        # Imports
        from trade_system.config import TradingConfig
        from trade_system.logging_config import setup_logging
        
        # Setup logging
        setup_logging("INFO")
        
        # Criar configuração
        config = TradingConfig.from_env()
        
        print(f"\n{'='*60}")
        print(f"🤖 ULTRA TRADING BOT - PAPER TRADING")
        print(f"{'='*60}")
        print(f"📊 Par: {config.symbol}")
        print(f"💰 Balance: ${config.base_balance:,.2f}")
        print(f"⏰ Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Tentar importar e criar sistema
        try:
            from trade_system.main import TradingSystem
            system = TradingSystem(config, paper_trading=True)
            
            print("🚀 Iniciando sistema de trading...\n")
            await system.start()
            
        except ImportError as e:
            print(f"\n❌ Erro de importação: {e}")
            print("\n🔧 Tentando importação alternativa...")
            
            # Tentar criar sistema mínimo
            from trade_system.websocket_manager import WebSocketManager
            from trade_system.analysis.technical import TechnicalAnalyzer
            from trade_system.risk import RiskManager
            
            print("✅ Módulos carregados com sucesso!")
            print("\n⚠️ Executando em modo limitado...")
            
            # Implementar loop básico aqui se necessário
            
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        logging.error("Erro detalhado:", exc_info=True)
        
        print("\n💡 Use o sistema simplificado:")
        print("   python working_paper_trading.py")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Sistema encerrado")
