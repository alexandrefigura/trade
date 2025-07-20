#!/usr/bin/env python3
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
