#!/usr/bin/env python3
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
                print(f"\n📊 Último Trade:")
                last = trades[-1]
                print(f"   Tipo: {last['side']}")
                print(f"   Preço: ${last['price']:.2f}")
                print(f"   Tamanho: ${last['size']:.2f}")
                print(f"   Horário: {last['time'].strftime('%H:%M:%S')}")
            
            # Histórico recente
            if len(trades) > 1:
                print(f"\n📜 Histórico (últimos 5):")
                for trade in trades[-5:]:
                    print(f"   {trade['time'].strftime('%H:%M:%S')} - {trade['side']} @ ${trade['price']:.2f}")
            
            print("\n⏸️  Pressione Ctrl+C para sair")
            
        except KeyboardInterrupt:
            print("\n👋 Monitor encerrado!")
            break
        except Exception as e:
            print(f"Erro: {e}")
        
        time.sleep(2)

if __name__ == "__main__":
    monitor()
