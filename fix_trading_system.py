#!/usr/bin/env python3
"""
Script para corrigir o sistema de trading e forçar execução de trades
"""
import os
import yaml
import shutil
from pathlib import Path
import requests

print("🔧 CORRIGINDO SISTEMA DE TRADING")
print("=" * 60)

# 1. Limpar checkpoints antigos
print("\n1️⃣ Limpando checkpoints antigos...")
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    # Fazer backup do último checkpoint
    checkpoints = list(checkpoint_dir.glob("*.pkl"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        shutil.copy(latest, f"backup_{latest.name}")
        print(f"✅ Backup criado: backup_{latest.name}")
    
    # Limpar todos os checkpoints
    shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir()
    print("✅ Checkpoints limpos!")

# 2. Configurar parâmetros ULTRA agressivos
print("\n2️⃣ Configurando parâmetros ultra agressivos...")
ultra_aggressive_config = """trading:
  symbol: "BTCUSDT"
  min_confidence: 0.10      # Apenas 10% de confiança!
  max_position_pct: 0.20    # 20% do saldo por trade
  take_profit_pct: 0.003    # 0.3% de lucro (muito baixo)
  stop_loss_pct: 0.005      # 0.5% de perda

risk:
  max_daily_loss: 0.50      # 50% perda máxima (apenas para teste!)
  min_balance_usd: 10.0
  max_pct_per_trade: 0.30   # 30% por trade
  min_trade_usd: 5.0        # Trades mínimos de $5

technical:
  rsi_buy_threshold: 70     # Comprar mesmo com RSI alto
  rsi_sell_threshold: 30    # Vender mesmo com RSI baixo
  rsi_period: 5
  sma_short_period: 2
  sma_long_period: 5
  buy_threshold: 0.0001     # Threshold praticamente zero
  sell_threshold: 0.0001
  rsi_confidence: 0.1       # Confiança mínima
  sma_cross_confidence: 0.1
  bb_confidence: 0.1
  pattern_confidence: 0.1

alerts:
  enable_alerts: true       # Telegram habilitado

redis:
  use_redis: false

# Parâmetros especiais para forçar trades
force_trades: true
max_open_positions: 5      # Permitir múltiplas posições
close_partial: true         # Permitir fechamento parcial
"""

with open("config.yaml", "w") as f:
    f.write(ultra_aggressive_config)
print("✅ Configuração ultra agressiva aplicada!")

# 3. Testar Telegram
print("\n3️⃣ Testando Telegram...")
token = "8199550294:AAEMIRLicQ167ED7Wz4cc_u2xAjHvAzpVTM"
chat_id = "1025666426"

url = f"https://api.telegram.org/bot{token}/sendMessage"
test_msg = {
    "chat_id": chat_id,
    "text": "🚀 <b>Sistema Reconfigurado!</b>\n\n"
            "✅ Checkpoints limpos\n"
            "✅ Parâmetros ultra agressivos\n"
            "✅ Pronto para executar MUITOS trades!\n\n"
            "⚠️ <i>Apenas para Paper Trading!</i>",
    "parse_mode": "HTML"
}

try:
    response = requests.post(url, data=test_msg, timeout=5)
    if response.status_code == 200:
        print("✅ Telegram funcionando!")
    else:
        print(f"❌ Erro Telegram: {response.text}")
except Exception as e:
    print(f"❌ Erro ao testar Telegram: {e}")

# 4. Modificar main.py para permitir múltiplas posições
print("\n4️⃣ Modificando sistema para múltiplas posições...")
main_path = Path("trade_system/main.py")
if main_path.exists():
    with open(main_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Procurar por verificação de posição existente
    if "if self.position_info:" in content:
        # Comentar a verificação
        content = content.replace(
            "if self.position_info:",
            "if False:  # MODIFICADO: Permitir múltiplas posições\n        # if self.position_info:"
        )
        
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Sistema modificado para múltiplas posições!")

# 5. Criar script de monitoramento
print("\n5️⃣ Criando monitor de trades...")
monitor_script = '''#!/usr/bin/env python3
"""Monitor de trades em tempo real"""
import time
import pickle
from pathlib import Path
from datetime import datetime

def monitor():
    print("📊 MONITOR DE TRADES")
    print("=" * 40)
    
    trades_count = 0
    last_balance = 10000
    
    while True:
        try:
            # Contar checkpoints (indica atividade)
            checkpoints = list(Path("checkpoints").glob("*.pkl"))
            
            # Tentar ler o último checkpoint
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest, "rb") as f:
                        data = pickle.load(f)
                        balance = data.get("balance", last_balance)
                        position = data.get("position_info")
                        
                        if balance != last_balance:
                            trades_count += 1
                            profit = balance - last_balance
                            print(f"\\n💰 TRADE #{trades_count}")
                            print(f"   Lucro/Prejuízo: ${profit:.2f}")
                            print(f"   Saldo: ${balance:.2f}")
                            if position:
                                print(f"   Posição: {position.get('side')} @ ${position.get('entry_price'):.2f}")
                            last_balance = balance
                except:
                    pass
            
            print(f"\\r⏱️  {datetime.now().strftime('%H:%M:%S')} | Trades: {trades_count} | Saldo: ${last_balance:.2f}", end="")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\\nErro: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    monitor()
'''

with open("monitor_trades.py", "w", encoding="utf-8") as f:
    f.write(monitor_script)
print("✅ Monitor criado!")

# 6. Instruções finais
print("\n" + "=" * 60)
print("✅ SISTEMA CORRIGIDO E PRONTO!")
print("=" * 60)

print("""
📋 PRÓXIMOS PASSOS:

1. Reinicie o sistema:
   trade-system paper

2. Em outro terminal, monitore os trades:
   python monitor_trades.py

3. O sistema agora vai:
   ✅ Executar MUITOS trades (confiança de apenas 10%)
   ✅ Permitir múltiplas posições simultâneas
   ✅ Enviar alertas no Telegram
   ✅ Take profit de apenas 0.3% (trades rápidos)

⚠️  ATENÇÃO: Configuração EXTREMAMENTE agressiva!
   Apenas para testes em Paper Trading!

💡 Se ainda não executar trades, verifique:
   - Se há saldo suficiente
   - Se o mercado está ativo
   - Os logs para mensagens de erro
""")

# 7. Criar um forçador de trades manual
force_trade_script = '''#!/usr/bin/env python3
"""Força a abertura de uma posição manualmente"""
import asyncio
from trade_system.main import TradingSystem
from trade_system.config import get_config

async def force_trade():
    config = get_config()
    system = TradingSystem(config, paper_trading=True)
    
    # Forçar abertura de posição
    await system._open_position(
        {"price": 119000},  # Preço atual aproximado
        "BUY",
        0.90  # 90% de confiança
    )
    print("✅ Posição forçada!")

if __name__ == "__main__":
    asyncio.run(force_trade())
'''

with open("force_trade.py", "w", encoding="utf-8") as f:
    f.write(force_trade_script)
print("\n💡 Criado script force_trade.py para forçar trades manualmente!")
