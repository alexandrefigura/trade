"""
Configuração centralizada do sistema de trading
"""
from dotenv import load_dotenv
load_dotenv()

import os
import yaml
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    # Credenciais
    api_key: str = ""
    api_secret: str = ""

    # Trading
    symbol: str = "BTCUSDT"
    min_confidence: float = 0.75
    max_position_pct: float = 0.05
    fee_rate: float = 0.001  # 0.1% por operação

    # Technical Analysis
    ta_interval_ms: int = 5000
    sma_short_period: int = 9
    sma_long_period: int = 20
    ema_short_period: int = 9
    ema_long_period: int = 20
    rsi_period: int = 14
    rsi_buy_threshold: float = 30.0
    rsi_sell_threshold: float = 70.0
    rsi_confidence: float = 0.8
    sma_cross_confidence: float = 0.75
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_confidence: float = 0.7
    pattern_confidence: float = 0.85
    buy_threshold: float = 0.3
    sell_threshold: float = 0.3

    # Filtros
    min_volume_multiplier: float = 1.0
    max_recent_volatility: float = 0.1

    # Risco / ATR
    atr_period: int = 14
    tp_multiplier: float = 1.5
    sl_multiplier: float = 1.0

    # Gestão de Posição (CORREÇÃO ADICIONADA)
    take_profit_pct: float = 0.02  # 2% de lucro
    stop_loss_pct: float = 0.01    # 1% de perda
    
    # Redis
    use_redis: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Processamento paralelo
    num_workers: int = mp.cpu_count()
    batch_size: int = 1000

    # Buffers
    price_buffer_size: int = 10000
    orderbook_buffer_size: int = 100

    # Intervalos
    main_loop_interval_ms: int = 1000
    gc_interval_cycles: int = 1000

    # Limites
    rate_limit_window: int = 60
    rate_limit_max_calls: int = 1200

    # Validações de mercado
    max_volatility: float = 0.1
    max_spread_bps: float = 20.0
    min_volume_24h: int = 1_000_000
    min_market_score: float = 50.0

    # Alertas
    enable_alerts: bool = True
    telegram_token: str = ""
    telegram_chat_id: str = ""
    alert_email: str = ""

    # Risco diário
    max_daily_loss: float = 0.02
    min_balance_usd: float = 100.0
    max_pct_per_trade: float = 0.10
    min_trade_usd: float = 50.0

    # Parâmetros de fechamento de posição
    trailing_start_pct: float = 0.005
    trailing_pct: float = 0.7
    max_position_duration: int = 3600
    time_stop_pct: float = 0.002

    # Debug
    debug_mode: bool = True


def load_config_from_yaml(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Carrega configurações de um arquivo YAML"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"{config_path} não encontrado, usando configurações padrão")
        return {}
    except Exception as e:
        logger.error(f"Erro ao carregar {config_path}: {e}")
        return {}


def get_config(debug_mode: bool = False) -> TradingConfig:
    """
    Obtém a configuração do sistema
    
    Args:
        debug_mode: Se True, usa parâmetros agressivos para testes
        
    Returns:
        TradingConfig: Configuração completa do sistema
    """
    base = TradingConfig()
    
    # Modo debug com parâmetros agressivos
    if debug_mode:
        base.debug_mode = True
        base.min_confidence = 0.40
        base.rsi_buy_threshold = 45.0
        base.rsi_sell_threshold = 55.0
        base.rsi_confidence = 0.50
        base.bb_confidence = 0.50
        base.sma_cross_confidence = 0.50
        base.pattern_confidence = 0.50
        base.buy_threshold = 0.05
        base.sell_threshold = 0.05
        base.min_volume_multiplier = 0.1
        base.max_recent_volatility = 0.50
        base.max_volatility = 0.50
        base.take_profit_pct = 0.01  # 1% no modo debug
        base.stop_loss_pct = 0.005   # 0.5% no modo debug
        logger.warning("⚠️ MODO DEBUG - Parâmetros ultra-agressivos ativados!")

    # Carrega configurações do YAML
    yaml_cfg = load_config_from_yaml()
    if yaml_cfg:
        for section in yaml_cfg:
            values = yaml_cfg.get(section, {})
            if isinstance(values, dict):
                for key, val in values.items():
                    if hasattr(base, key):
                        setattr(base, key, val)
            elif hasattr(base, section):
                setattr(base, section, values)

    # Sobrescreve com variáveis de ambiente
    base.api_key = os.getenv("BINANCE_API_KEY", base.api_key)
    base.api_secret = os.getenv("BINANCE_API_SECRET", base.api_secret)
    base.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", base.telegram_token)
    base.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", base.telegram_chat_id)

    # Validações básicas
    if not base.api_key or not base.api_secret:
        logger.warning("⚠️ API Keys não configuradas - rodando em modo simulação")
    
    if base.enable_alerts and (not base.telegram_token or not base.telegram_chat_id):
        logger.warning("⚠️ Telegram não configurado - alertas desabilitados")
        base.enable_alerts = False

    return base


def create_default_yaml(path: str = 'config.yaml'):
    """Cria um arquivo YAML de configuração padrão"""
    default_config = {
        'trading': {
            'symbol': 'BTCUSDT',
            'min_confidence': 0.75,
            'max_position_pct': 0.02,
            'take_profit_pct': 0.02,
            'stop_loss_pct': 0.01
        },
        'risk': {
            'max_daily_loss': 0.02,
            'min_balance_usd': 100.0,
            'max_pct_per_trade': 0.10,
            'min_trade_usd': 50.0
        },
        'technical': {
            'rsi_buy_threshold': 30,
            'rsi_sell_threshold': 70,
            'rsi_period': 14,
            'sma_short_period': 9,
            'sma_long_period': 20
        },
        'alerts': {
            'enable_alerts': True
        }
    }
    
    try:
        with open(path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        logger.info(f"✅ Arquivo de configuração criado: {path}")
    except Exception as e:
        logger.error(f"Erro ao criar arquivo de configuração: {e}")


# Para testes
if __name__ == "__main__":
    config = get_config()
    print("Configuração carregada:")
    print(f"- Symbol: {config.symbol}")
    print(f"- Take Profit: {config.take_profit_pct * 100}%")
    print(f"- Stop Loss: {config.stop_loss_pct * 100}%")
    print(f"- Min Confidence: {config.min_confidence}")
    print(f"- Debug Mode: {config.debug_mode}")
