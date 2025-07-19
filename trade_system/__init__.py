"""Ultra Trading System - Bot de trading algorítmico de alta performance"""

__version__ = "2.0.0"
__author__ = "Alexandre Figura"

from trade_system.config import TradingConfig
from trade_system.main import TradingSystem, run_paper_trading

__all__ = ['TradingConfig', 'TradingSystem', 'run_paper_trading']
