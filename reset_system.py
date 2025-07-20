#!/usr/bin/env python3
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
