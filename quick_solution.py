"""Solução rápida para os imports"""
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
        content += '\n\n# Alias para compatibilidade\nSimplifiedMLPredictor = MLPredictor\n'
        
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
                content += f'\n{alias} = {original}\n'
                added = True
                print(f"✅ Alias {alias} criado em {filepath}")
        
        if added:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

print("\n✅ Correções aplicadas!")
print("\nTente executar novamente: python run_trading.py")
