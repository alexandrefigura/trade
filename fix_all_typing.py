import os
import re

def fix_typing_in_file(filepath):
    """Corrige todos os imports de typing em um arquivo"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Encontrar todos os tipos usados
        types_used = set()
        
        # Padrões para encontrar tipos
        patterns = [
            (r'\bDict\[', 'Dict'),
            (r'\bList\[', 'List'),
            (r'\bTuple\[', 'Tuple'),
            (r'\bOptional\[', 'Optional'),
            (r'\bUnion\[', 'Union'),
            (r'\bAny\b', 'Any'),
            (r'\bSet\[', 'Set'),
            (r'\bCallable\[', 'Callable'),
            (r': Dict\b', 'Dict'),
            (r': List\b', 'List'),
            (r': Any\b', 'Any'),
            (r': Optional\b', 'Optional'),
            (r': Tuple\b', 'Tuple'),
            (r': Set\b', 'Set'),
        ]
        
        for pattern, type_name in patterns:
            if re.search(pattern, content):
                types_used.add(type_name)
        
        if not types_used:
            return False
        
        # Verificar se já tem import de typing
        has_typing_import = 'from typing import' in content
        
        if has_typing_import:
            # Atualizar import existente
            match = re.search(r'from typing import ([^\n]+)', content)
            if match:
                existing_imports = match.group(1).strip()
                existing_types = set(t.strip() for t in existing_imports.split(','))
                
                # Adicionar tipos faltantes
                all_types = existing_types.union(types_used)
                new_import = f"from typing import {', '.join(sorted(all_types))}"
                
                content = re.sub(r'from typing import [^\n]+', new_import, content)
        else:
            # Adicionar novo import
            lines = content.split('\n')
            
            # Encontrar onde inserir
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_pos = i + 1
                    else:
                        break
            
            # Inserir import de typing
            typing_import = f"from typing import {', '.join(sorted(types_used))}"
            lines.insert(insert_pos, typing_import)
            
            content = '\n'.join(lines)
        
        # Salvar se mudou
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corrigido: {filepath}")
            return True
            
    except Exception as e:
        print(f"❌ Erro em {filepath}: {e}")
    
    return False

# Corrigir todos os arquivos Python
print("🔧 Corrigindo imports de typing em todos os arquivos...\n")

fixed_count = 0
for root, dirs, files in os.walk('trade_system'):
    # Pular diretórios desnecessários
    if '__pycache__' in root:
        continue
        
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            if fix_typing_in_file(filepath):
                fixed_count += 1

print(f"\n✅ Total de arquivos corrigidos: {fixed_count}")

# Verificar especificamente os arquivos problemáticos
problem_files = [
    'trade_system/rate_limiter.py',
    'trade_system/cache.py',
    'trade_system/config.py',
    'trade_system/main.py'
]

print("\n🔍 Verificando arquivos problemáticos:")
for file in problem_files:
    if os.path.exists(file):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'from typing import' in content:
                match = re.search(r'from typing import ([^\n]+)', content)
                if match:
                    print(f"✅ {file}: {match.group(0)}")
            else:
                print(f"⚠️ {file}: Sem import de typing")
        except Exception as e:
            print(f"❌ {file}: {e}")

print("\n✅ Correção completa!")
