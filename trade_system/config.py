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
class UltraConfigV5:
    """Configuração para máxima performance"""
    api_key: str = ""
    api_secret: str = ""

    # Trading
    symbol: str = "BTCUSDT"
    min_confidence: float = 0.75
    max_position_pct: float = 0.05  # Aumentado para facilitar testes

    # TA
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
    max_recent_volatility: float = 0.1  # Relaxado para facilitar simulações

    # ATR / Risk
    atr_period: int = 14
    tp_multiplier: float = 1.5
    sl_multiplier: float = 1.0

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

    # Rate Limiting
    rate_limit_window: int = 60
    rate_limit_max_calls: int = 1200

    # Validações de mercado
    max_volatility: float = 0.1  # Relaxado
    max_spread_bps: float = 20.0
    min_volume_24h: int = 1_000_000

    # Alertas
    enable_alerts: bool = True
    telegram_token: str = ""
    telegram_chat_id: str = ""
    alert_email: str = ""

    # Risco
    max_daily_loss: float = 0.02

    # Debug
    debug_mode: bool = True  # Ativado por padrão para simulações


def load_config_from_yaml(config_path: str = 'config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"{config_path} não encontrado, usando configurações padrão")
        return {}
    except Exception as e:
        logger.error(f"Erro ao carregar {config_path}: {e}")
        return {}


def create_debug_config() -> UltraConfigV5:
    config = UltraConfigV5()
    config.min_confidence = 0.40
    config.rsi_buy_threshold = 45.0
    config.rsi_sell_threshold = 55.0
    config.rsi_confidence = 0.50
    config.bb_confidence = 0.50
    config.sma_cross_confidence = 0.50
    config.pattern_confidence = 0.50
    config.buy_threshold = 0.05
    config.sell_threshold = 0.05
    config.min_volume_multiplier = 0.1
    config.max_recent_volatility = 0.50
    config.max_volatility = 0.50
    config.debug_mode = True
    logger.warning("⚠️ MODO DEBUG - Parâmetros ultra-agressivos ativados!")
    return config


def get_config(debug_mode: bool = False) -> UltraConfigV5:
    config = create_debug_config() if debug_mode else UltraConfigV5()
    yaml_cfg = load_config_from_yaml()

    if yaml_cfg:
        for key in ('symbol', 'min_confidence', 'max_position_pct'):
            if key in yaml_cfg.get('trading', {}):
                setattr(config, key, yaml_cfg['trading'][key])

        for key in ('max_volatility', 'max_spread_bps', 'max_daily_loss',
                    'atr_period', 'tp_multiplier', 'sl_multiplier'):
            if key in yaml_cfg.get('risk', {}):
                setattr(config, key, yaml_cfg['risk'][key])

        for key, val in yaml_cfg.get('ta', {}).items():
            if hasattr(config, key):
                setattr(config, key, val)

        for key in ('min_volume_multiplier', 'max_recent_volatility'):
            if key in yaml_cfg.get('filters', {}):
                setattr(config, key, yaml_cfg['filters'][key])

        for key in ('ta_interval_ms', 'main_loop_interval_ms'):
            if key in yaml_cfg.get('performance', {}):
                setattr(config, key, yaml_cfg['performance'][key])

        if 'debug_mode' in yaml_cfg:
            config.debug_mode = yaml_cfg['debug_mode']

    config.api_key = os.getenv('BINANCE_API_KEY', config.api_key)
    config.api_secret = os.getenv('BINANCE_API_SECRET', config.api_secret)
    config.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', config.telegram_token)
    config.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', config.telegram_chat_id)

    return config


def create_example_config(config_path: str = 'config.yaml'):
    example = """# Configuração do Sistema de Trading v5.2
debug_mode: true

trading:
  symbol: "BTCUSDT"
  min_confidence: 0.75
  max_position_pct: 0.05

risk:
  max_volatility: 0.1
  max_spread_bps: 20
  max_daily_loss: 0.02
  atr_period: 14
  tp_multiplier: 1.5
  sl_multiplier: 1.0

ta:
  rsi_period: 14
  rsi_buy_threshold: 30
  rsi_sell_threshold: 70
  rsi_confidence: 0.8
  sma_short_period: 9
  sma_long_period: 20
  ema_short_period: 9
  ema_long_period: 20
  sma_cross_confidence: 0.75
  bb_period: 20
  bb_std_dev: 2.0
  bb_confidence: 0.7
  pattern_confidence: 0.85
  buy_threshold: 0.3
  sell_threshold: 0.3

filters:
  min_volume_multiplier: 1.0
  max_recent_volatility: 0.1

performance:
  ta_interval_ms: 5000
  main_loop_interval_ms: 1000
"""
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(example)
    logger.info(f"Arquivo {config_path} criado com sucesso!")
