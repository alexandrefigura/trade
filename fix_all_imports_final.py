import os
import re

print("🔧 CORREÇÃO DEFINITIVA DE TODOS OS IMPORTS")
print("=" * 60)

# 1. Mapear TODAS as classes que o sistema está tentando importar
print("\n🔍 Analisando todos os arquivos Python...")

all_imports = {}
all_classes = {}

# Encontrar todos os imports
for root, dirs, files in os.walk('trade_system'):
    if '__pycache__' in root:
        continue
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Encontrar imports
                import_pattern = r'from\s+(trade_system[\w.]*)\s+import\s+([^\n]+)'
                imports = re.findall(import_pattern, content)
                
                if imports:
                    all_imports[filepath] = imports
                
                # Encontrar classes definidas
                class_pattern = r'^class\s+(\w+)'
                classes = re.findall(class_pattern, content, re.MULTILINE)
                
                if classes:
                    all_classes[filepath] = classes
                    
            except Exception as e:
                print(f"❌ Erro ao ler {filepath}: {e}")

# 2. Mostrar imports problemáticos
print("\n⚠️ Imports que podem estar errados:")

missing_classes = set()
for filepath, imports in all_imports.items():
    for module, imported_items in imports:
        items = [item.strip() for item in imported_items.split(',')]
        
        for item in items:
            # Limpar item
            item = item.split(' as ')[0].strip()
            
            # Verificar se a classe existe
            found = False
            for class_file, classes in all_classes.items():
                if item in classes:
                    found = True
                    break
            
            if not found and item not in ['Any', 'Dict', 'List', 'Optional', 'Tuple']:
                missing_classes.add(item)
                print(f"  - {item} (importado em {os.path.basename(filepath)})")

# 3. Criar mapeamento de correções
print("\n📝 Criando correções...")

corrections = {
    # Nomes errados conhecidos
    'ParallelOrderbookAnalyzer': 'OrderbookAnalyzer',
    'UltraFastTechnicalAnalyzer': 'TechnicalAnalyzer',
    'UltraFastWebSocketManager': 'WebSocketManager',
    'AlertSystem': 'AlertManager',
    'UltraFastCache': 'CacheManager',
    'MLEngine': 'MLPredictor',
    'SignalManager': 'SignalAggregator',
    'ConfigManager': 'TradingConfig',
    'get_logger': 'logging.getLogger'
}

# 4. Aplicar correções em todos os arquivos
print("\n🔧 Aplicando correções...")

fixed_files = 0
for root, dirs, files in os.walk('trade_system'):
    if '__pycache__' in root:
        continue
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Aplicar todas as correções
                for wrong, correct in corrections.items():
                    if wrong in content:
                        # Corrigir imports
                        content = re.sub(
                            f'import\\s+{wrong}',
                            f'import {correct}',
                            content
                        )
                        
                        # Corrigir uso
                        content = re.sub(
                            f'\\b{wrong}\\b',
                            correct,
                            content
                        )
                
                # Salvar se mudou
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Corrigido: {filepath}")
                    fixed_files += 1
                    
            except Exception as e:
                print(f"❌ Erro ao corrigir {filepath}: {e}")

print(f"\n✅ Total de arquivos corrigidos: {fixed_files}")

# 5. Criar script de execução simplificado
print("\n📝 Criando script de execução simplificado...")

simple_run = '''"""Execução simplificada do Paper Trading"""
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
        print("\\nVerifique se todos os módulos estão corretos")
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
        print("\\n\\n👋 Sistema encerrado")
'''

with open('simple_run.py', 'w', encoding='utf-8') as f:
    f.write(simple_run)

print("✅ simple_run.py criado!")

print("\n✅ Correção completa!")
print("\n🚀 Tente executar:")
print("   python run_trading.py")
print("\n   ou se não funcionar:")
print("   python simple_run.py")
