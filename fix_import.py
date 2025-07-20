# Criar um arquivo de correção
@"
import sys
import os

# Corrigir o import no main.py
file_path = 'trade_system/main.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Substituir import errado
content = content.replace('from trade_system.config import get_config', 'from trade_system.config import TradingConfig')
content = content.replace('get_config()', 'TradingConfig.from_env()')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Import corrigido!")
"@ | Out-File -FilePath fix_import.py -Encoding UTF8

# Executar a correção
python fix_import.py
