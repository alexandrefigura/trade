#!/usr/bin/env python3
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
                            print(f"\n💰 TRADE #{trades_count}")
                            print(f"   Lucro/Prejuízo: ${profit:.2f}")
                            print(f"   Saldo: ${balance:.2f}")
                            if position:
                                print(f"   Posição: {position.get('side')} @ ${position.get('entry_price'):.2f}")
                            last_balance = balance
                except:
                    pass
            
            print(f"\r⏱️  {datetime.now().strftime('%H:%M:%S')} | Trades: {trades_count} | Saldo: ${last_balance:.2f}", end="")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nErro: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    monitor()
