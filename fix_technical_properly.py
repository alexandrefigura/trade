#!/usr/bin/env python3
"""
Script para corrigir completamente o arquivo technical.py
"""
from pathlib import Path

print("🔧 Corrigindo technical.py completamente...")

technical_path = Path("trade_system/analysis/technical.py")

if technical_path.exists():
    with open(technical_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remover a função calculate_momentum mal posicionada
    lines = content.split('\n')
    new_lines = []
    skip_momentum = False
    
    for line in lines:
        if line.strip().startswith('def calculate_momentum'):
            skip_momentum = True
            continue
        if skip_momentum and line.strip() and not line.startswith(' '):
            skip_momentum = False
        if not skip_momentum:
            new_lines.append(line)
    
    # Reconstruir o conteúdo
    content = '\n'.join(new_lines)
    
    # Agora adicionar a função no lugar correto
    # Primeiro, garantir que numpy está importado
    if 'import numpy as np' not in content:
        # Adicionar após outros imports
        import_section_end = 0
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import') or line.startswith('from'):
                import_section_end = i + 1
        
        lines.insert(import_section_end, 'import numpy as np')
        content = '\n'.join(lines)
    
    # Adicionar calculate_momentum após os imports e antes da classe
    momentum_function = '''
def calculate_momentum(prices: np.ndarray, period: int = 10) -> float:
    """Calcula o momentum dos preços"""
    if len(prices) < period + 1:
        return 0.0
    
    current = prices[-1]
    past = prices[-(period + 1)]
    
    if past == 0:
        return 0.0
        
    return ((current - past) / past) * 100
'''
    
    # Encontrar onde adicionar (antes da classe UltraFastTechnicalAnalysis)
    if 'def calculate_momentum' not in content:
        class_index = content.find('class UltraFastTechnicalAnalysis')
        if class_index > 0:
            # Adicionar antes da classe
            before_class = content[:class_index]
            after_class = content[class_index:]
            content = before_class + momentum_function + '\n' + after_class
    
    # Adicionar momentum ao get_signals se ainda não estiver
    if '"momentum"' not in content:
        # Procurar features no get_signals
        import re
        pattern = r'(features = {[^}]+)'
        
        def add_momentum(match):
            features = match.group(1)
            if 'momentum' not in features:
                return features.rstrip() + ',\n            "momentum": calculate_momentum(prices)'
            return features
        
        content = re.sub(pattern, add_momentum, content)
    
    # Salvar o arquivo corrigido
    with open(technical_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ technical.py corrigido completamente!")
    
    # Mostrar as primeiras linhas para verificação
    print("\nPrimeiras linhas do arquivo:")
    print("-" * 40)
    lines = content.split('\n')[:20]
    for i, line in enumerate(lines, 1):
        print(f"{i:2}: {line}")

else:
    print("❌ Arquivo technical.py não encontrado!")

print("\n✅ Execute novamente: trade-system paper")
