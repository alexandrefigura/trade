import os
import re

def check_file_classes(filepath):
    """Verifica classes em um arquivo"""
    print(f"\n📄 Verificando: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Encontrar definições de classe
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        
        if classes:
            print(f"   Classes encontradas: {', '.join(classes)}")
        else:
            print(f"   ❌ Nenhuma classe encontrada!")
            
        return classes
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return []

# Verificar arquivos problemáticos
files_to_check = [
    'trade_system/websocket_manager.py',
    'trade_system/alerts.py',
    'trade_system/cache.py',
    'trade_system/config.py'
]

print("🔍 Verificando classes nos arquivos principais:\n")

class_mapping = {}
for file in files_to_check:
    if os.path.exists(file):
        classes = check_file_classes(file)
        if classes:
            class_mapping[file] = classes

# Agora corrigir main.py
print("\n🔧 Corrigindo imports em main.py...")

try:
    with open('trade_system/main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Mapear imports errados para corretos baseado no que encontramos
    corrections = []
    
    # Verificar cada import
    import_lines = re.findall(r'from trade_system\.(\w+) import ([^\n]+)', content)
    
    for module, imports in import_lines:
        module_file = f'trade_system/{module}.py'
        
        if module_file in class_mapping:
            real_classes = class_mapping[module_file]
            imported_classes = [c.strip() for c in imports.split(',')]
            
            for imported in imported_classes:
                if imported not in real_classes and real_classes:
                    # Tentar encontrar a classe correta
                    print(f"\n⚠️  Import incorreto: {imported} de {module}")
                    print(f"   Classes disponíveis: {', '.join(real_classes)}")
                    
                    # Se houver apenas uma classe, usar ela
                    if len(real_classes) == 1:
                        corrections.append((imported, real_classes[0]))
                        print(f"   ✅ Corrigindo: {imported} → {real_classes[0]}")
    
    # Aplicar correções
    if corrections:
        original_content = content
        for wrong, correct in corrections:
            # Corrigir no import
            content = re.sub(f'import\\s+{wrong}', f'import {correct}', content)
            # Corrigir no uso
            content = re.sub(f'\\b{wrong}\\b', correct, content)
        
        if content != original_content:
            with open('trade_system/main.py', 'w', encoding='utf-8') as f:
                f.write(content)
            print("\n✅ main.py corrigido!")
    
except Exception as e:
    print(f"\n❌ Erro ao corrigir main.py: {e}")

print("\n✅ Verificação completa!")
