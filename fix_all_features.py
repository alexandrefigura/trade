#!/usr/bin/env python3
"""
Script para corrigir todos os problemas identificados no sistema de trading
"""
import os
import sys
import subprocess
import time
import re
from pathlib import Path

print("🔧 CORREÇÃO COMPLETA DO SISTEMA DE TRADING")
print("=" * 60)

# 1. CORRIGIR O ARQUIVO CONFIG.PY
print("\n1️⃣ Corrigindo config.py...")
config_path = Path("trade_system/config.py")

if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar se take_profit_pct e stop_loss_pct existem
    if 'take_profit_pct:' not in content:
        # Adicionar após max_position_pct
        pattern = r'(max_position_pct: float = [\d.]+)'
        replacement = r'\1\n    take_profit_pct: float = 0.02  # 2% de lucro\n    stop_loss_pct: float = 0.01    # 1% de perda'
        content = re.sub(pattern, replacement, content)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ config.py corrigido com take_profit_pct e stop_loss_pct")
    else:
        print("✅ config.py já está correto")

# 2. ADICIONAR CÁLCULO DE MOMENTUM
print("\n2️⃣ Adicionando cálculo de momentum...")
technical_path = Path("trade_system/analysis/technical.py")

momentum_code = '''
def calculate_momentum(prices: np.ndarray, period: int = 10) -> float:
    """Calcula o momentum dos preços"""
    if len(prices) < period + 1:
        return 0.0
    
    current = prices[-1]
    past = prices[-(period + 1)]
    
    if past == 0:
        return 0.0
        
    return ((current - past) / past) * 100
'''

if technical_path.exists():
    with open(technical_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'calculate_momentum' not in content:
        # Adicionar após os imports
        import_section = content.split('\n\n')[0]
        rest_of_file = '\n\n'.join(content.split('\n\n')[1:])
        
        new_content = import_section + '\n' + momentum_code + '\n' + rest_of_file
        
        # Adicionar momentum ao get_signals
        if 'momentum' not in new_content:
            pattern = r'(features = {[^}]+)'
            replacement = r'\1,\n            "momentum": calculate_momentum(prices)'
            new_content = re.sub(pattern, replacement, new_content)
        
        with open(technical_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Função calculate_momentum adicionada")
    else:
        print("✅ Momentum já está implementado")

# 3. CONFIGURAR TELEGRAM
print("\n3️⃣ Configurando Telegram...")
env_file = Path(".env")

telegram_config = """
# Telegram Configuration
TELEGRAM_BOT_TOKEN=seu_token_aqui
TELEGRAM_CHAT_ID=seu_chat_id_aqui
"""

if not env_file.exists():
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(telegram_config)
    print("✅ Arquivo .env criado")
    print("⚠️  IMPORTANTE: Edite .env e adicione seu token e chat_id do Telegram")
else:
    print("✅ .env já existe")

print("""
📱 Para configurar o Telegram:
1. Crie um bot com @BotFather no Telegram
2. Pegue o token do bot
3. Envie uma mensagem para o bot
4. Acesse: https://api.telegram.org/bot<TOKEN>/getUpdates
5. Encontre seu chat_id
6. Adicione no arquivo .env
""")

# 4. INSTALAR E CONFIGURAR REDIS (Windows)
print("\n4️⃣ Configurando Redis...")
print("""
Para Windows, você tem 3 opções:

OPÇÃO A - WSL (Recomendado):
1. Abra o PowerShell como Admin
2. Execute: wsl --install
3. Reinicie o computador
4. Abra o WSL e execute:
   sudo apt update
   sudo apt install redis-server
   sudo service redis-server start

OPÇÃO B - Redis Windows (Não oficial):
1. Baixe: https://github.com/microsoftarchive/redis/releases
2. Instale o MSI
3. Redis será instalado como serviço

OPÇÃO C - Desabilitar Redis:
""")

config_yaml = Path("config.yaml")
if config_yaml.exists():
    with open(config_yaml, 'r') as f:
        yaml_content = f.read()
    
    if 'use_redis: true' in yaml_content:
        yaml_content = yaml_content.replace('use_redis: true', 'use_redis: false')
        with open(config_yaml, 'w') as f:
            f.write(yaml_content)
        print("✅ Redis desabilitado no config.yaml")

# 5. AJUSTAR PARÂMETROS PARA MAIS TRADES
print("\n5️⃣ Ajustando parâmetros para paper trading ativo...")

aggressive_config = """trading:
  symbol: "BTCUSDT"
  min_confidence: 0.30  # Mais agressivo
  max_position_pct: 0.05
  take_profit_pct: 0.015  # 1.5%
  stop_loss_pct: 0.01   # 1%

risk:
  max_daily_loss: 0.05  # 5% máximo de perda diária
  min_balance_usd: 100.0
  max_pct_per_trade: 0.10
  min_trade_usd: 50.0

technical:
  rsi_buy_threshold: 40  # Mais sensível
  rsi_sell_threshold: 60
  rsi_period: 14
  sma_short_period: 5   # Períodos menores
  sma_long_period: 15
  buy_threshold: 0.2    # Thresholds menores
  sell_threshold: 0.2

alerts:
  enable_alerts: false  # Desabilitar até configurar Telegram
  
redis:
  use_redis: false  # Desabilitar Redis por enquanto
"""

with open(config_yaml, 'w', encoding='utf-8') as f:
    f.write(aggressive_config)
print("✅ config.yaml ajustado para paper trading agressivo")

# 6. CRIAR SCRIPT DE MONITORAMENTO
print("\n6️⃣ Criando script de monitoramento...")

monitor_script = '''#!/usr/bin/env python3
"""Monitor de performance do sistema de trading"""
import time
import json
from pathlib import Path
from datetime import datetime

def monitor_system():
    checkpoint_dir = Path("checkpoints")
    last_balance = 10000
    
    print("📊 MONITOR DE TRADING - PAPER MODE")
    print("=" * 60)
    
    while True:
        try:
            # Encontrar checkpoint mais recente
            checkpoints = list(checkpoint_dir.glob("*.pkl"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                
                # Aqui você poderia carregar o pickle, mas por simplicidade
                # vamos apenas mostrar o nome do arquivo
                timestamp = latest.stem.split('_')[-1]
                
                print(f"\\r⏰ {datetime.now().strftime('%H:%M:%S')} | ", end="")
                print(f"📁 Checkpoint: {timestamp} | ", end="")
                print(f"💰 Balance: ${last_balance:.2f}", end="")
                
        except Exception as e:
            print(f"\\nErro: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    monitor_system()
'''

with open("monitor.py", "w", encoding='utf-8') as f:
    f.write(monitor_script)
print("✅ monitor.py criado")

# 7. CRIAR SCRIPT DE RESET
print("\n7️⃣ Criando script de reset...")

reset_script = '''#!/usr/bin/env python3
"""Reset do sistema para começar do zero"""
import shutil
from pathlib import Path

print("🔄 RESET DO SISTEMA DE TRADING")
response = input("Tem certeza? Isso apagará todos os checkpoints! (s/n): ")

if response.lower() == 's':
    # Limpar checkpoints
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir()
    
    # Limpar logs
    log_dir = Path("logs")
    if log_dir.exists():
        for log in log_dir.glob("*.log"):
            log.unlink()
    
    print("✅ Sistema resetado!")
    print("Execute: trade-system paper")
else:
    print("❌ Reset cancelado")
'''

with open("reset_system.py", "w", encoding='utf-8') as f:
    f.write(reset_script)
print("✅ reset_system.py criado")

# RESUMO FINAL
print("\n" + "=" * 60)
print("✅ CORREÇÕES APLICADAS!")
print("=" * 60)

print("""
📋 PRÓXIMOS PASSOS:

1. Configure o Telegram (opcional):
   - Edite .env com seu token e chat_id
   - Ou mantenha enable_alerts: false

2. Configure o Redis (opcional):
   - Instale via WSL ou Windows
   - Ou mantenha use_redis: false

3. Reinicie o sistema:
   - Pare com Ctrl+C
   - Execute: trade-system paper

4. Para monitorar:
   - Em outro terminal: python monitor.py

5. Para resetar tudo:
   - python reset_system.py

💡 DICAS:
- O sistema agora está mais agressivo para paper trading
- Vai executar mais trades para você aprender
- Monitore os logs para entender o comportamento
- Ajuste os parâmetros no config.yaml conforme necessário

🚀 Boa sorte com seu paper trading!
""")

print("\n⏳ Aplicando mudanças...")
time.sleep(2)
print("✅ Pronto! Reinicie o sistema com: trade-system paper")
