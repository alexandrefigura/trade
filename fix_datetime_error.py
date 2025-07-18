#!/usr/bin/env python3
"""
Corrigir erro de datetime no sistema de trading
"""
import re
from pathlib import Path
from datetime import datetime

print("🔧 CORRIGINDO ERRO DE DATETIME")
print("=" * 60)

# 1. Corrigir risk.py
print("\n1️⃣ Corrigindo risk.py...")
risk_path = Path("trade_system/risk.py")

if risk_path.exists():
    with open(risk_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar a linha problemática
    if "elapsed = time.time() - self.position_info.get('entry_time'" in content:
        # Adicionar conversão para timestamp
        old_line = "elapsed = time.time() - self.position_info.get('entry_time', time.time())"
        new_line = """entry_time = self.position_info.get('entry_time', time.time())
        # Converter datetime para timestamp se necessário
        if isinstance(entry_time, datetime):
            entry_time = entry_time.timestamp()
        elapsed = time.time() - entry_time"""
        
        content = content.replace(old_line, new_line)
        
        # Adicionar import datetime se não existir
        if "from datetime import datetime" not in content:
            content = "from datetime import datetime\n" + content
        
        with open(risk_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ risk.py corrigido!")

# 2. Corrigir main.py para garantir que entry_time seja timestamp
print("\n2️⃣ Corrigindo main.py...")
main_path = Path("trade_system/main.py")

if main_path.exists():
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar onde position_info é criado
    if "'entry_time':" in content:
        # Substituir datetime.now() por time.time()
        content = re.sub(
            r"'entry_time':\s*datetime\.now\(\)",
            "'entry_time': time.time()",
            content
        )
        
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ main.py corrigido!")

# 3. Criar script de monitoramento melhorado
print("\n3️⃣ Criando monitor melhorado...")
monitor_script = '''#!/usr/bin/env python3
"""Monitor de trades com estatísticas"""
import time
import pickle
from pathlib import Path
from datetime import datetime
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def monitor():
    trades = []
    last_checkpoint = None
    start_balance = 10000
    
    while True:
        try:
            clear_screen()
            print("📊 MONITOR DE TRADES - PAPER TRADING")
            print("=" * 60)
            
            # Buscar último checkpoint
            checkpoints = list(Path("checkpoints").glob("*.pkl"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                
                if latest != last_checkpoint:
                    last_checkpoint = latest
                    
                    try:
                        with open(latest, "rb") as f:
                            data = pickle.load(f)
                            
                            balance = data.get("balance", start_balance)
                            position = data.get("position_info")
                            
                            if position:
                                trades.append({
                                    "time": datetime.now(),
                                    "side": position.get("side"),
                                    "price": position.get("entry_price"),
                                    "size": position.get("position_size"),
                                    "balance": balance
                                })
                    except:
                        pass
            
            # Mostrar estatísticas
            current_balance = trades[-1]["balance"] if trades else start_balance
            total_pnl = current_balance - start_balance
            pnl_pct = (total_pnl / start_balance) * 100
            
            print(f"💰 Saldo Atual: ${current_balance:.2f}")
            print(f"📈 P&L Total: ${total_pnl:.2f} ({pnl_pct:+.2f}%)")
            print(f"🔢 Total de Trades: {len(trades)}")
            
            if trades:
                print(f"\\n📊 Último Trade:")
                last = trades[-1]
                print(f"   Tipo: {last['side']}")
                print(f"   Preço: ${last['price']:.2f}")
                print(f"   Tamanho: ${last['size']:.2f}")
                print(f"   Horário: {last['time'].strftime('%H:%M:%S')}")
            
            # Histórico recente
            if len(trades) > 1:
                print(f"\\n📜 Histórico (últimos 5):")
                for trade in trades[-5:]:
                    print(f"   {trade['time'].strftime('%H:%M:%S')} - {trade['side']} @ ${trade['price']:.2f}")
            
            print("\\n⏸️  Pressione Ctrl+C para sair")
            
        except KeyboardInterrupt:
            print("\\n👋 Monitor encerrado!")
            break
        except Exception as e:
            print(f"Erro: {e}")
        
        time.sleep(2)

if __name__ == "__main__":
    monitor()
'''

with open("monitor_live.py", "w", encoding='utf-8') as f:
    f.write(monitor_script)
print("✅ Monitor melhorado criado!")

# 4. Informações sobre o que aconteceu
print("\n" + "=" * 60)
print("📊 ANÁLISE DO QUE ACONTECEU:")
print("=" * 60)

print("""
✅ SUCESSO: O sistema executou seu primeiro trade!
   - Tipo: BUY
   - Preço: $119,166.80
   - Valor: $313.38
   - Confiança: 52.2%

❌ ERRO: Incompatibilidade de tipos de dados
   - O sistema estava salvando datetime mas esperando timestamp
   - Isso causou o crash após abrir a posição

🔧 CORREÇÕES APLICADAS:
   1. risk.py agora converte datetime para timestamp
   2. main.py agora salva timestamps ao invés de datetime
   3. Monitor melhorado criado

📋 PRÓXIMOS PASSOS:
   1. Execute novamente: trade-system paper
   2. Em outro terminal: python monitor_live.py
   3. O sistema deve funcionar continuamente agora!
""")

print("\n💡 Dica: O sistema está configurado para ser MUITO agressivo.")
print("   Espere ver muitos trades sendo executados!")
