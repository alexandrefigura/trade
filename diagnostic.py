import os
import re
import json

print("🔍 DIAGNÓSTICO COMPLETO DO SISTEMA")
print("=" * 60)

# 1. Mapear TODAS as classes existentes
print("\n📋 Classes existentes no projeto:\n")

all_classes = {}
for root, dirs, files in os.walk('trade_system'):
    if '__pycache__' in root:
        continue
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Encontrar todas as classes
                classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                
                if classes:
                    module_path = filepath.replace('\\', '.').replace('/', '.').replace('.py', '')
                    all_classes[module_path] = classes
                    print(f"📄 {os.path.basename(filepath)}:")
                    for cls in classes:
                        print(f"   - {cls}")
                    print()
                    
            except Exception as e:
                print(f"❌ Erro ao ler {filepath}: {e}")

# 2. Verificar o que main.py está tentando importar
print("\n🔍 Analisando imports em main.py...\n")

main_file = 'trade_system/main.py'
if os.path.exists(main_file):
    with open(main_file, 'r', encoding='utf-8') as f:
        main_content = f.read()
    
    # Extrair todos os imports
    imports = re.findall(r'from\s+(trade_system[\w.]*)\s+import\s+([^\n]+)', main_content)
    
    print("Imports encontrados em main.py:")
    for module, items in imports:
        print(f"\nfrom {module} import {items}")
        
        # Verificar cada item
        items_list = [item.strip() for item in items.split(',')]
        for item in items_list:
            item = item.split(' as ')[0].strip()
            
            # Verificar se existe
            found = False
            for mod_path, classes in all_classes.items():
                if item in classes:
                    found = True
                    if not mod_path.endswith(module.replace('trade_system.', '').replace('.', os.sep)):
                        print(f"   ⚠️ {item} existe em {mod_path}, não em {module}")
                    break
            
            if not found:
                print(f"   ❌ {item} NÃO ENCONTRADO!")

# 3. Criar solução
print("\n\n💡 SOLUÇÃO RÁPIDA:")
print("=" * 60)

solution = '''"""Solução rápida para os imports"""
import os

# Criar aliases para as classes com nomes errados

# 1. Para analysis/ml.py
ml_file = 'trade_system/analysis/ml.py'
if os.path.exists(ml_file):
    with open(ml_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Adicionar SimplifiedMLPredictor se não existir
    if 'SimplifiedMLPredictor' not in content and 'MLPredictor' in content:
        # Adicionar alias no final do arquivo
        content += '\\n\\n# Alias para compatibilidade\\nSimplifiedMLPredictor = MLPredictor\\n'
        
        with open(ml_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Alias SimplifiedMLPredictor criado!")

# 2. Para outros arquivos com problemas similares
files_to_check = {
    'trade_system/analysis/orderbook.py': {
        'ParallelOrderbookAnalyzer': 'OrderbookAnalyzer'
    },
    'trade_system/analysis/technical.py': {
        'UltraFastTechnicalAnalyzer': 'TechnicalAnalyzer'
    }
}

for filepath, aliases in files_to_check.items():
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Adicionar aliases
        added = False
        for alias, original in aliases.items():
            if alias not in content and original in content:
                content += f'\\n{alias} = {original}\\n'
                added = True
                print(f"✅ Alias {alias} criado em {filepath}")
        
        if added:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

print("\\n✅ Correções aplicadas!")
print("\\nTente executar novamente: python run_trading.py")
'''

with open('quick_solution.py', 'w', encoding='utf-8') as f:
    f.write(solution)

print("\n📝 Script de solução criado: quick_solution.py")
print("\nExecute: python quick_solution.py")

# 4. Alternativa - criar wrapper
print("\n\n🔧 ALTERNATIVA - Criar Wrapper:")

wrapper_content = '''"""Wrapper para executar o sistema com imports corrigidos"""
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
print("\\n🚀 Iniciando sistema...")
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
        print("\\n\\n👋 Sistema encerrado")
'''

with open('run_with_fixes.py', 'w', encoding='utf-8') as f:
    f.write(wrapper_content)

print("📝 Wrapper criado: run_with_fixes.py")
