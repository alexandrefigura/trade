#!/usr/bin/env python3
"""
Paper Trading Final - Versão Corrigida
"""
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trade_system.config import TradingConfig
from trade_system.main import TradingSystem
from trade_system.logging_config import setup_logging

# Carregar variáveis de ambiente
load_dotenv()

async def main():
    """Função principal com tratamento de erros aprimorado"""
    try:
        # Verificar API keys
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            print("❌ Erro: BINANCE_API_KEY ou BINANCE_API_SECRET não encontradas no arquivo .env")
            print("💡 Crie um arquivo .env com:")
            print('   BINANCE_API_KEY="sua_api_key"')
            print('   BINANCE_API_SECRET="sua_api_secret"')
            return
            
        print(f"✅ API Key: {api_key[:8]}...")
        
        # Configurar logging
        print("\n🔄 Carregando sistema...")
        setup_logging()
        
        # Carregar configuração
        config = TradingConfig()
        
        # Banner
        print(f"""
============================================================
🤖 ULTRA TRADING BOT - PAPER TRADING (CORRIGIDO)
============================================================
📊 Par: {config.SYMBOL}
💰 Balance: ${config.INITIAL_BALANCE:,.2f}
⏰ Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================
        """)
        
        # Criar sistema com paper_trading=True
        system = TradingSystem(config, paper_trading=True)
        
        # Executar
        print("\n🚀 Iniciando Paper Trading...\n")
        await system.run()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 Use o sistema simplificado:")
        print("   python working_paper_trading.py")

if __name__ == "__main__":
    asyncio.run(main())
