import os

# Corrigir o import no main.py
file_path = 'trade_system/main.py'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Substituir imports errados
    content = content.replace('from trade_system.config import get_config', 
                            'from trade_system.config import TradingConfig')
    
    # Se tiver get_config() no código, substituir
    if 'get_config()' in content:
        content = content.replace('get_config()', 'TradingConfig.from_env()')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Import corrigido em main.py!")
    
    # Verificar __init__.py também
    init_path = 'trade_system/__init__.py'
    with open(init_path, 'r', encoding='utf-8') as f:
        init_content = f.read()
    
    if 'get_config' in init_content:
        init_content = init_content.replace('get_config', 'TradingConfig')
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(init_content)
        print("✅ Import corrigido em __init__.py!")

except Exception as e:
    print(f"❌ Erro: {e}")
