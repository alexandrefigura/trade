"""Configuração centralizada do sistema"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
import yaml
import json

# Carregar variáveis de ambiente
from dotenv import load_dotenv
load_dotenv()


@dataclass
class TradingConfig:
    """Configuração do sistema de trading"""
    # API
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    
    # Trading
    symbol: str = "BTCUSDT"
    base_balance: float = 10000.0
    min_confidence: float = 0.75
    max_position_pct: float = 0.02
    min_order_size: float = 10.0
    
    # Risk Management
    risk: Dict[str, Any] = field(default_factory=lambda: {
        'max_volatility': 0.05,
        'max_spread_bps': 20,
        'max_daily_loss': 0.02,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.03,
        'trailing_stop_pct': 0.01,
        'max_positions': 1,
        'position_timeout_hours': 24
    })
    
    # Technical Analysis
    ta: Dict[str, Any] = field(default_factory=lambda: {
        'rsi_period': 14,
        'rsi_buy_threshold': 30,
        'rsi_sell_threshold': 70,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2.0,
        'ema_short': 9,
        'ema_long': 21,
        'atr_period': 14,
        'momentum_period': 10
    })
    
    # Machine Learning
    ml: Dict[str, Any] = field(default_factory=lambda: {
        'features': [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
            'bb_middle', 'volume_ratio', 'price_change', 'volatility',
            'momentum', 'support', 'resistance', 'trend_strength',
            'volume_profile', 'ema_diff', 'atr'
        ],
        'lookback': 100,
        'retrain_interval': 1000,
        'model_type': 'gradient_boosting',
        'test_size': 0.2,
        'n_estimators': 100
    })
    
    # WebSocket
    websocket: Dict[str, Any] = field(default_factory=lambda: {
        'buffer_size': 1000,
        'reconnect_delay': 5,
        'ping_interval': 30,
        'streams': ['trade', 'depth20@100ms', 'kline_1m']
    })
    
    # System
    system: Dict[str, Any] = field(default_factory=lambda: {
        'log_level': 'INFO',
        'checkpoint_interval': 300,
        'health_check_interval': 60,
        'cache_ttl': 60,
        'rate_limit_per_minute': 1200
    })
    
    # Alerts
    alerts: Dict[str, Any] = field(default_factory=lambda: {
        'telegram_enabled': True,
        'telegram_token': '',
        'telegram_chat_id': '',
        'email_enabled': False,
        'email_smtp': '',
        'email_from': '',
        'email_to': ''
    })
    
    @classmethod
    def from_file(cls, filepath: str = "config.yaml") -> 'TradingConfig':
        """Carrega configuração de arquivo"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
                return cls(**data) if data else cls()
        return cls()
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Carrega configuração de variáveis de ambiente"""
        config = cls()
        config.api_key = os.getenv('BINANCE_API_KEY', '')
        config.api_secret = os.getenv('BINANCE_API_SECRET', '')
        config.alerts['telegram_token'] = os.getenv('TELEGRAM_BOT_TOKEN', '')
        config.alerts['telegram_chat_id'] = os.getenv('TELEGRAM_CHAT_ID', '')
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'testnet': self.testnet,
            'symbol': self.symbol,
            'base_balance': self.base_balance,
            'min_confidence': self.min_confidence,
            'max_position_pct': self.max_position_pct,
            'min_order_size': self.min_order_size,
            'risk': self.risk,
            'ta': self.ta,
            'ml': self.ml,
            'websocket': self.websocket,
            'system': self.system,
            'alerts': self.alerts
        }
    
    def save(self, filepath: str = "config.yaml"):
        """Salva configuração em arquivo"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
