import os

# Corrigir o import em main.py
main_file = 'trade_system/main.py'

try:
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substituir UltraFastCache por CacheManager
    content = content.replace('from trade_system.cache import UltraFastCache', 
                            'from trade_system.cache import CacheManager')
    content = content.replace('UltraFastCache(', 'CacheManager(')
    
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ main.py corrigido!")
    
except Exception as e:
    print(f"❌ Erro: {e}")

# Verificar outros arquivos que podem ter o mesmo problema
for root, dirs, files in os.walk('trade_system'):
    for file in files:
        if file.endswith('.py') and file != 'cache.py':
            filepath = os.path.join(root, file)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'UltraFastCache' in content:
                    content = content.replace('UltraFastCache', 'CacheManager')
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"✅ Corrigido: {filepath}")
                    
            except Exception as e:
                print(f"❌ Erro em {filepath}: {e}")

print("\n✅ Todas as correções aplicadas!")
