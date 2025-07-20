import os
import re

def fix_file(filepath, replacements):
    """Corrige imports em um arquivo"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corrigido: {filepath}")
        
    except Exception as e:
        print(f"❌ Erro em {filepath}: {e}")

# Correções necessárias
main_fixes = {
    'from trade_system.logging_config import setup_logging, get_logger': 
        'from trade_system.logging_config import setup_logging',
    'get_logger': 'logging.getLogger',
    'from trade_system.config import get_config': 
        'from trade_system.config import TradingConfig',
    'get_config()': 'TradingConfig.from_env()',
}

# Corrigir main.py
fix_file('trade_system/main.py', main_fixes)

# Corrigir cli.py
cli_fixes = {
    'from trade_system.logging_config import setup_logging, get_logger': 
        'from trade_system.logging_config import setup_logging',
    'get_logger': 'logging.getLogger',
}
fix_file('trade_system/cli.py', cli_fixes)

# Verificar todos os arquivos Python
for root, dirs, files in os.walk('trade_system'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            
            # Correções gerais
            general_fixes = {
                ', get_logger': '',
                'get_logger(__name__)': 'logging.getLogger(__name__)',
                'get_logger(': 'logging.getLogger(',
            }
            
            fix_file(filepath, general_fixes)

print("\n✅ Correções concluídas!")
print("\n📝 Adicionando import logging onde necessário...")

# Adicionar import logging se necessário
for root, dirs, files in os.walk('trade_system'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Se usa logging.getLogger mas não tem import logging
                if 'logging.getLogger' in content and 'import logging' not in content:
                    # Adicionar após os primeiros imports
                    lines = content.split('\n')
                    import_added = False
                    
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            # Adicionar após o primeiro bloco de imports
                            if i + 1 < len(lines) and not (lines[i + 1].startswith('import ') or lines[i + 1].startswith('from ')):
                                lines.insert(i + 1, 'import logging')
                                import_added = True
                                break
                    
                    if import_added:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(lines))
                        print(f"✅ Added logging import to: {filepath}")
                        
            except Exception as e:
                print(f"❌ Erro processando {filepath}: {e}")

print("\n✅ Todas as correções foram aplicadas!")
