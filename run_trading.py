#!/usr/bin/env python3
"""Script direto para executar o trading bot"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Adicionar diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

async def main():
    try:
        # Importar após configurar path
        from trade_system.config import TradingConfig
        from trade_system.main import TradingSystem
        
        # Carregar configuração
        config = TradingConfig.from_env()
        
        # Verificar credenciais
        if not config.api_key or not config.api_secret:
            print("❌ ERRO: Configure as variáveis de ambiente:")
            print("   BINANCE_API_KEY")
            print("   BINANCE_API_SECRET")
            return
        
        print(f"✅ Configuração carregada")
        print(f"📊 Par: {config.symbol}")
        print(f"💰 Balance: ${config.base_balance:,.2f}")
        print()
        
        # Criar e executar sistema
        system = TradingSystem(config, paper_trading=True)
        await system.start()
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("\nInstale as dependências:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     PAPER TRADING MODE                               ║
║                Execução simulada com dados reais                     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Sistema interrompido pelo usuário")
