import os
import re

def add_typing_imports(filepath):
    """Adiciona imports de typing necessários"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar se precisa de typing imports
        needs_typing = any(word in content for word in ['Dict[', 'List[', 'Optional[', 'Tuple[', 'Any', 'Union['])
        
        if needs_typing and 'from typing import' not in content:
            # Encontrar onde adicionar o import
            lines = content.split('\n')
            import_line = None
            
            # Procurar após outros imports
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_line = i
            
            if import_line is not None:
                # Adicionar typing import
                typing_imports = []
                if 'Dict[' in content or 'Dict,' in content:
                    typing_imports.append('Dict')
                if 'List[' in content or 'List,' in content:
                    typing_imports.append('List')
                if 'Optional[' in content:
                    typing_imports.append('Optional')
                if 'Tuple[' in content:
                    typing_imports.append('Tuple')
                if 'Any' in content:
                    typing_imports.append('Any')
                if 'Union[' in content:
                    typing_imports.append('Union')
                
                if typing_imports:
                    import_statement = f"from typing import {', '.join(typing_imports)}"
                    lines.insert(import_line + 1, import_statement)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    
                    print(f"✅ Added typing imports to: {filepath}")
                    return True
    
    except Exception as e:
        print(f"❌ Error in {filepath}: {e}")
    
    return False

# Corrigir todos os arquivos Python
fixed_count = 0
for root, dirs, files in os.walk('trade_system'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            if add_typing_imports(filepath):
                fixed_count += 1

print(f"\n✅ Total de arquivos corrigidos: {fixed_count}")

# Corrigir especificamente o cache.py
print("\n🔧 Verificando cache.py especificamente...")
cache_file = 'trade_system/cache.py'

try:
    with open(cache_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Se não tem o import correto, adicionar
    if 'from typing import' not in content:
        lines = content.split('\n')
        # Adicionar após primeiros imports
        for i, line in enumerate(lines):
            if line.startswith('import'):
                lines.insert(i + 1, 'from typing import Dict, Any, Optional')
                break
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print("✅ cache.py corrigido!")
        
except Exception as e:
    print(f"❌ Erro ao corrigir cache.py: {e}")

print("\n✅ Todas as correções de typing foram aplicadas!")
