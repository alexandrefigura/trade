#!/usr/bin/env python3
"""
Script para corrigir o erro de importação do numpy no technical.py
"""
import re
from pathlib import Path

print("🔧 Corrigindo importação do numpy...")

technical_path = Path("trade_system/analysis/technical.py")

if technical_path.exists():
    with open(technical_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar se numpy já está importado
    if 'import numpy as np' not in content:
        # Encontrar a seção de imports
        lines = content.split('\n')
        
        # Procurar onde adicionar o import
        import_index = 0
        for i, line in enumerate(lines):
            if line.startswith('import') or line.startswith('from'):
                import_index = i + 1
            elif import_index > 0 and not line.strip():
                # Encontrou linha em branco após imports
                break
        
        # Se não encontrou imports, adicionar no início
        if import_index == 0:
            lines.insert(0, 'import numpy as np')
            lines.insert(1, '')
        else:
            # Adicionar após os outros imports
            lines.insert(import_index, 'import numpy as np')
        
        # Salvar o arquivo corrigido
        with open(technical_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print("✅ Import do numpy adicionado!")
    else:
        print("✅ Numpy já está importado")
        
    # Verificar se a função calculate_momentum está correta
    if 'calculate_momentum' in content and 'momentum' not in content.split('calculate_momentum')[1]:
        print("⚠️  A função calculate_momentum pode não estar sendo usada no get_signals")
        print("Verificando e corrigindo...")
        
        # Procurar a função get_signals e adicionar momentum
        pattern = r'(def get_signals[^:]+:[^}]+features = {[^}]+)'
        replacement = r'\1,\n            "momentum": calculate_momentum(prices)'
        
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        if new_content != content:
            with open(technical_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("✅ Momentum adicionado ao get_signals!")

print("\n✅ Correção concluída!")
print("Execute novamente: trade-system paper")
