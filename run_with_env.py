"""Wrapper que garante carregamento do .env"""
import os
import sys
import asyncio

# Carregar .env PRIMEIRO
from dotenv import load_dotenv
load_dotenv()

# Verificar se carregou
api_key = os.getenv('BINANCE_API_KEY')
if not api_key:
    # Tentar carregar manualmente
    if os.path.exists('.env'):
        print("📄 Carregando .env manualmente...")
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip('"').strip("'")
        
        api_key = os.getenv('BINANCE_API_KEY')

if not api_key:
    print("❌ BINANCE_API_KEY não encontrada!")
    print("\nVerifique o arquivo .env")
    sys.exit(1)

print(f"✅ API Key carregada: {api_key[:8]}...")

# Agora importar e executar
from trade_system.config import TradingConfig
from trade_system.main import run_paper_trading

if __name__ == "__main__":
    asyncio.run(run_paper_trading())
