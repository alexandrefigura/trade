import os
import re

def find_class_names_in_file(filepath):
    """Encontra todas as classes definidas em um arquivo"""
    classes = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Encontrar definições de classe
        class_pattern = r'^class\s+(\w+)'
        matches = re.finditer(class_pattern, content, re.MULTILINE)
        
        for match in matches:
            classes.append(match.group(1))
            
    except Exception as e:
        print(f"Erro ao ler {filepath}: {e}")
    
    return classes

def find_all_classes():
    """Mapeia todos os arquivos e suas classes"""
    class_map = {}
    
    for root, dirs, files in os.walk('trade_system'):
        if '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                classes = find_class_names_in_file(filepath)
                
                if classes:
                    module_name = filepath.replace('\\', '.').replace('/', '.').replace('.py', '')
                    for class_name in classes:
                        class_map[class_name] = module_name
                        print(f"Encontrado: {class_name} em {module_name}")
    
    return class_map

def fix_imports():
    """Corrige todos os imports baseado nas classes reais"""
    print("\n🔍 Mapeando todas as classes...\n")
    class_map = find_all_classes()
    
    print("\n🔧 Corrigindo imports...\n")
    
    # Nomes incorretos conhecidos
    wrong_names = {
        'AlertSystem': 'AlertManager',
        'UltraFastCache': 'CacheManager',
        'UltraFastWebSocketManager': 'WebSocketManager',
        'ConfigManager': 'TradingConfig',
        'MLEngine': 'MLPredictor',
        'TechnicalAnalysis': 'TechnicalAnalyzer',
        'OrderbookAnalysis': 'OrderbookAnalyzer',
        'SignalManager': 'SignalAggregator',
        'RiskControl': 'RiskManager'
    }
    
    # Corrigir cada arquivo
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
                    
                    # Aplicar correções
                    for wrong, correct in wrong_names.items():
                        if wrong in content:
                            # Corrigir imports
                            content = re.sub(
                                f'from\\s+[\\w.]+\\s+import\\s+.*{wrong}.*',
                                lambda m: m.group(0).replace(wrong, correct),
                                content
                            )
                            
                            # Corrigir uso
                            content = re.sub(f'\\b{wrong}\\b(?!\\s*=)', correct, content)
                    
                    # Salvar se mudou
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"✅ Corrigido: {filepath}")
                        
                except Exception as e:
                    print(f"❌ Erro em {filepath}: {e}")

# Executar correções
fix_imports()

print("\n📝 Verificando main.py...")
try:
    with open('trade_system/main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extrair todos os imports
    import_pattern = r'^from\s+trade_system\.\w+\s+import\s+(.+)$'
    imports = re.findall(import_pattern, content, re.MULTILINE)
    
    print("\nImports em main.py:")
    for imp in imports:
        print(f"  - {imp}")
        
except Exception as e:
    print(f"Erro: {e}")

print("\n✅ Correção completa!")
