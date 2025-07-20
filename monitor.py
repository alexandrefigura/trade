#!/usr/bin/env python3
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
                
                print(f"\r⏰ {datetime.now().strftime('%H:%M:%S')} | ", end="")
                print(f"📁 Checkpoint: {timestamp} | ", end="")
                print(f"💰 Balance: ${last_balance:.2f}", end="")
                
        except Exception as e:
            print(f"\nErro: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    monitor_system()
