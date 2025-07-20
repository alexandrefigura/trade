import os
import re
import json

print("🔍 ANÁLISE COMPLETA DO PROJETO")
print("=" * 60)

# 1. Analisar estrutura completa
print("\n📁 Analisando estrutura do projeto...")

project_structure = {}
imports_map = {}
classes_map = {}
errors_found = []

def analyze_file(filepath):
    """Analisa um arquivo Python"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Encontrar imports
        imports = re.findall(r'from\s+([\w.]+)\s+import\s+([^\n]+)', content)
        
        # Encontrar classes definidas
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        
        # Encontrar erros de sintaxe simples
        if 'import logging.getLogger' in content:
            errors_found.append(f"Erro de sintaxe em {filepath}: import logging.getLogger")
        
        return {
            'imports': imports,
            'classes': classes,
            'content': content
        }
    except Exception as e:
        return {'error': str(e)}

# Analisar todos os arquivos
for root, dirs, files in os.walk('trade_system'):
    if '__pycache__' in root:
        continue
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            relative_path = filepath.replace('\\', '/')
            
            analysis = analyze_file(filepath)
            project_structure[relative_path] = analysis
            
            if 'classes' in analysis:
                for cls in analysis['classes']:
                    classes_map[cls] = relative_path
            
            if 'imports' in analysis:
                imports_map[relative_path] = analysis['imports']

# 2. Identificar problemas
print("\n⚠️ Problemas encontrados:")

# Classes que estão sendo importadas mas não existem
missing_classes = set()
for file, imports in imports_map.items():
    for module, items in imports:
        if module.startswith('trade_system'):
            items_list = [item.strip() for item in items.split(',')]
            for item in items_list:
                item = item.split(' as ')[0].strip()
                if item not in classes_map and item not in ['Any', 'Dict', 'List', 'Optional', 'Tuple']:
                    missing_classes.add((item, file))
                    print(f"  - {item} importado em {os.path.basename(file)} mas não existe")

# 3. Gerar relatório
print("\n📊 Resumo da Análise:")
print(f"  - Total de arquivos: {len(project_structure)}")
print(f"  - Total de classes: {len(classes_map)}")
print(f"  - Imports problemáticos: {len(missing_classes)}")
print(f"  - Erros de sintaxe: {len(errors_found)}")

# 4. Salvar análise
analysis_report = {
    'structure': project_structure,
    'classes': classes_map,
    'missing': list(missing_classes),
    'errors': errors_found
}

with open('analysis_report.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_report, f, indent=2)

print("\n📝 Relatório salvo em: analysis_report.json")

# 5. Criar script de correção baseado na análise
print("\n🔧 Criando script de correção...")

fix_script = f'''"""Correções baseadas na análise do projeto"""
import os
import re

# Mapeamento de correções baseado na análise
corrections = {{
    {chr(10).join([f'    "{item}": "SUBSTITUIR_POR_CLASSE_CORRETA",  # em {file}' for item, file in missing_classes])}
}}

# Classes encontradas no projeto:
available_classes = {list(classes_map.keys())}

print("Classes disponíveis no projeto:")
for cls in available_classes:
    print(f"  - {{cls}}")

print("\\nPor favor, atualize o mapeamento 'corrections' com as classes corretas!")
'''

with open('generated_fix.py', 'w', encoding='utf-8') as f:
    f.write(fix_script)

print("✅ Script de correção gerado: generated_fix.py")

# 6. Tentar correções automáticas óbvias
print("\n🤖 Aplicando correções automáticas...")

auto_corrections = {
    'import logging.getLogger': 'import logging',
    'ParallelOrderbookAnalyzer': 'OrderbookAnalyzer',
    'UltraFast': '',  # Remover prefixo UltraFast
}

fixed_count = 0
for filepath, data in project_structure.items():
    if 'content' in data:
        content = data['content']
        original = content
        
        for wrong, correct in auto_corrections.items():
            content = content.replace(wrong, correct)
        
        if content != original:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Corrigido: {filepath}")
                fixed_count += 1
            except Exception as e:
                print(f"❌ Erro ao corrigir {filepath}: {e}")

print(f"\n✅ Total de correções automáticas: {fixed_count}")
print("\n🎯 Próximos passos:")
print("1. Revise o arquivo analysis_report.json")
print("2. Execute python direct_paper_trading.py para testar")
