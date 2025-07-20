import os
import re

# Mapeamento de nomes errados para corretos
class_mappings = {
    'AlertSystem': 'AlertManager',
    'UltraFastCache': 'CacheManager',
    'ConfigManager': 'TradingConfig',
    'get_config': 'TradingConfig.from_env',
    'get_logger': 'logging.getLogger'
}

def fix_imports_in_file(filepath):
    """Corrige imports incorretos em um arquivo"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Aplicar todas as correções
        for old_name, new_name in class_mappings.items():
            if old_name in content:
                # Corrigir imports
                content = re.sub(f'from [\\w.]+ import .*{old_name}.*', 
                               lambda m: m.group(0).replace(old_name, new_name), 
                               content)
                
                # Corrigir uso no código
                content = re.sub(f'\\b{old_name}\\b', new_name, content)
        
        # Salvar se mudou
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corrigido: {filepath}")
            return True
            
    except Exception as e:
        print(f"❌ Erro em {filepath}: {e}")
    
    return False

# Corrigir todos os arquivos
print("🔧 Corrigindo nomes de classes...\n")

fixed_count = 0
for root, dirs, files in os.walk('trade_system'):
    if '__pycache__' in root:
        continue
        
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            if fix_imports_in_file(filepath):
                fixed_count += 1

print(f"\n✅ Total de arquivos corrigidos: {fixed_count}")

# Verificar main.py especificamente
print("\n🔍 Verificando main.py:")
try:
    with open('trade_system/main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines[:30]):  # Primeiras 30 linhas
        if 'import' in line:
            print(f"  Linha {i+1}: {line.strip()}")
            
except Exception as e:
    print(f"❌ Erro ao verificar main.py: {e}")

print("\n✅ Correção completa!")
