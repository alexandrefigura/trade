"""Wrapper para executar o sistema com imports corrigidos"""
import sys
import os

# Adicionar diretório ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Criar módulo fake com todos os aliases necessários
class ImportFixer:
    def __init__(self):
        # Mapear classes erradas para corretas
        self.mappings = {
            'SimplifiedMLPredictor': 'MLPredictor',
            'ParallelOrderbookAnalyzer': 'OrderbookAnalyzer',
            'UltraFastTechnicalAnalyzer': 'TechnicalAnalyzer',
            'UltraFastWebSocketManager': 'WebSocketManager'
        }
    
    def fix_imports(self):
        """Adiciona aliases nos módulos"""
        # ML
        try:
            from trade_system.analysis import ml
            if hasattr(ml, 'MLPredictor') and not hasattr(ml, 'SimplifiedMLPredictor'):
                ml.SimplifiedMLPredictor = ml.MLPredictor
                print("✅ Fixed: SimplifiedMLPredictor")
        except: pass
        
        # Orderbook
        try:
            from trade_system.analysis import orderbook
            if hasattr(orderbook, 'OrderbookAnalyzer') and not hasattr(orderbook, 'ParallelOrderbookAnalyzer'):
                orderbook.ParallelOrderbookAnalyzer = orderbook.OrderbookAnalyzer
                print("✅ Fixed: ParallelOrderbookAnalyzer")
        except: pass
        
        # Technical
        try:
            from trade_system.analysis import technical
            if hasattr(technical, 'TechnicalAnalyzer') and not hasattr(technical, 'UltraFastTechnicalAnalyzer'):
                technical.UltraFastTechnicalAnalyzer = technical.TechnicalAnalyzer
                print("✅ Fixed: UltraFastTechnicalAnalyzer")
        except: pass

# Aplicar fixes
fixer = ImportFixer()
fixer.fix_imports()

# Agora executar o sistema
print("\n🚀 Iniciando sistema...")
import asyncio
from trade_system.config import TradingConfig
from trade_system.main import TradingSystem

async def main():
    config = TradingConfig.from_env()
    system = TradingSystem(config, paper_trading=True)
    await system.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Sistema encerrado")
