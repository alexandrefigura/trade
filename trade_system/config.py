"""
Configuração centralizada do sistema de trading
"""
from dotenv import load_dotenv
load_dotenv()  # carrega as variáveis do .env para os.environ

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
    # APIs
    api_key: str = os.getenv('BINANCE_API_KEY', '')
    api_secret: str = os.getenv('BINANCE_API_SECRET', '')
    
    # Trading
    symbol: str = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
    min_confidence: float = float(os.getenv('MIN_CONFIDENCE', 0.75))
    max_position_pct: float = float(os.getenv('MAX_POSITION_PCT', 0.02))

    # Parâmetros de Technical Analysis
    ta_interval_ms: int = int(os.getenv('TA_INTERVAL_MS', 5000))
    sma_short_period: int = int(os.getenv('SMA_SHORT_PERIOD', 9))
    sma_long_period: int = int(os.getenv('SMA_LONG_PERIOD', 20))
    ema_short_period: int = int(os.getenv('EMA_SHORT_PERIOD', 9))
    ema_long_period: int = int(os.getenv('EMA_LONG_PERIOD', 20))
    rsi_period: int = int(os.getenv('RSI_PERIOD', 14))
    rsi_buy_threshold: float = float(os.getenv('RSI_BUY_THRESHOLD', 30.0))
    rsi_sell_threshold: float = float(os.getenv('RSI_SELL_THRESHOLD', 70.0))
    rsi_confidence: float = float(os.getenv('RSI_CONFIDENCE', 0.8))
    sma_cross_confidence: float = float(os.getenv('SMA_CROSS_CONFIDENCE', 0.75))
    bb_period: int = int(os.getenv('BB_PERIOD', 20))
    bb_std_dev: float = float(os.getenv('BB_STD_DEV', 2.0))
    bb_confidence: float = float(os.getenv('BB_CONFIDENCE', 0.7))
    pattern_confidence: float = float(os.getenv('PATTERN_CONFIDENCE', 0.85))
    buy_threshold: float = float(os.getenv('BUY_THRESHOLD', 0.3))
    sell_threshold: float = float(os.getenv('SELL_THRESHOLD', 0.3))

    # Filtros
    min_volume_multiplier: float = float(os.getenv('MIN_VOLUME_MULTIPLIER', 1.0))
    max_recent_volatility: float = float(os.getenv('MAX_RECENT_VOLATILITY', 0.05))

    # ATR parameters
    atr_period: int = int(os.getenv('ATR_PERIOD', 14))
    tp_multiplier: float = float(os.getenv('TP_MULTIPLIER', 1.5))
    sl_multiplier: float = float(os.getenv('SL_MULTIPLIER', 1.0))

    # Performance
    use_redis: bool = os.getenv('USE_REDIS', 'True').lower() in ('true', '1', 'yes')
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', 6379))

    # Processamento paralelo
    num_workers: int = int(os.getenv('NUM_WORKERS', mp.cpu_count()))
    batch_size: int = int(os.getenv('BATCH_SIZE', 1000))

    # Buffers otimizados
    price_buffer_size: int = int(os.getenv('PRICE_BUFFER_SIZE', 10000))
    orderbook_buffer_size: int = int(os.getenv('ORDERBOOK_BUFFER_SIZE', 100))

    # Timing
    main_loop_interval_ms: int = int(os.getenv('MAIN_LOOP_INTERVAL_MS', 1000))
    gc_interval_cycles: int = int(os.getenv('GC_INTERVAL_CYCLES', 1000))

    # Configuração avançada
    rate_limit_window: int = int(os.getenv('RATE_LIMIT_WINDOW', 60))
    rate_limit_max_calls: int = int(os.getenv('RATE_LIMIT_MAX_CALLS', 1200))

    # Proteções de mercado
    max_volatility: float = float(os.getenv('MAX_VOLATILITY', 0.05))
    max_spread_bps: float = float(os.getenv('MAX_SPREAD_BPS', 20.0))
    min_volume_24h: int = int(os.getenv('MIN_VOLUME_24H', 1_000_000))

    # Alertas
    enable_alerts: bool = os.getenv('ENABLE_ALERTS', 'True').lower() in ('true', '1', 'yes')
    telegram_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    alert_email: str = os.getenv('ALERT_EMAIL', '')

    # Risk management
    max_daily_loss: float = float(os.getenv('MAX_DAILY_LOSS', 0.02))

    # Debug mode
    debug_mode: bool = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 'yes')


def load_config_from_yaml(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Carrega configurações do arquivo YAML"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"{config_path} não encontrado, usando configurações padrão")
        return {}
    except Exception as e:
        logger.error(f"Erro ao carregar {config_path}: {e}")
        return {}


def create_debug_config() -> UltraConfigV5:
    """Cria configuração ultra-agressiva para debug e testes"""
    config = UltraConfigV5()
    
    # Parâmetros extremamente agressivos para garantir sinais
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
    """
    Retorna configuração completa do sistema
    Prioridade: ENV > YAML > Debug > Default
    """
    if debug_mode:
        config = create_debug_config()
    else:
        config = UltraConfigV5()
    
    # Carregar do YAML
    yaml_cfg = load_config_from_yaml()
    if yaml_cfg:
        # Trading
        trading = yaml_cfg.get('trading', {})
        config.symbol = trading.get('symbol', config.symbol)
        config.min_confidence = trading.get('min_confidence', config.min_confidence)
        config.max_position_pct = trading.get('max_position_pct', config.max_position_pct)
        
        # Risk
        risk = yaml_cfg.get('risk', {})
        config.max_volatility = risk.get('max_volatility', config.max_volatility)
        config.max_spread_bps = risk.get('max_spread_bps', config.max_spread_bps)
        config.max_daily_loss = risk.get('max_daily_loss', config.max_daily_loss)
        config.atr_period = risk.get('atr_period', config.atr_period)
        config.tp_multiplier = risk.get('tp_multiplier', config.tp_multiplier)
        config.sl_multiplier = risk.get('sl_multiplier', config.sl_multiplier)
        
        # TA parameters
        ta = yaml_cfg.get('ta', {})
        for key, value in ta.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Filters
        filters = yaml_cfg.get('filters', {})
        config.min_volume_multiplier = filters.get('min_volume_multiplier', config.min_volume_multiplier)
        config.max_recent_volatility = filters.get('max_recent_volatility', config.max_volatility)
        
        # Performance
        performance = yaml_cfg.get('performance', {})
        config.ta_interval_ms = performance.get('ta_interval_ms', config.ta_interval_ms)
        config.main_loop_interval_ms = performance.get('main_loop_interval_ms', config.main_loop_interval_ms)
        
        # Debug do YAML tem prioridade sobre parâmetro
        config.debug_mode = yaml_cfg.get('debug_mode', debug_mode)
    
    # ENV vars têm prioridade máxima
    config.api_key = os.getenv('BINANCE_API_KEY', config.api_key)
    config.api_secret = os.getenv('BINANCE_API_SECRET', config.api_secret)
    config.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', config.telegram_token)
    config.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', config.telegram_chat_id)
    
    return config


def create_example_config(config_path: str = 'config.yaml'):
    """Cria um arquivo config.yaml de exemplo"""
    example_config = """# Configuração do Sistema de Trading v5.2

# Modo debug (parâmetros mais agressivos)
debug_mode: false

trading:
  symbol: "BTCUSDT"
  min_confidence: 0.75  # Confiança mínima para abrir posição (75%)
  max_position_pct: 0.02  # Máximo 2% do balance por posição

risk:
  max_volatility: 0.05  # Máxima volatilidade aceita (5%)
  max_spread_bps: 20  # Máximo spread em basis points
  max_daily_loss: 0.02  # Stop loss diário (2%)
  atr_period: 14  # Período para cálculo do ATR
  tp_multiplier: 1.5  # Take profit em 1.5x ATR
  sl_multiplier: 1.0  # Stop loss em 1x ATR

ta:
  # RSI
  rsi_period: 14
  rsi_buy_threshold: 30  # Compra quando RSI < 30
  rsi_sell_threshold: 70  # Venda quando RSI > 70
  rsi_confidence: 0.8
  
  # Moving Averages
  sma_short_period: 9
  sma_long_period: 20
  ema_short_period: 9
  ema_long_period: 20
  sma_cross_confidence: 0.75
  
  # Bollinger Bands
  bb_period: 20
  bb_std_dev: 2.0
  bb_confidence: 0.7
  
  # Pattern Detection
  pattern_confidence: 0.85
  
  # Thresholds para decisão final
  buy_threshold: 0.3
  sell_threshold: 0.3

filters:
  min_volume_multiplier: 1.0  # Volume mínimo em relação à média
  max_recent_volatility: 0.05  # Máxima volatilidade recente (5%)

performance:
  ta_interval_ms: 5000  # Intervalo mínimo entre análises técnicas
  main_loop_interval_ms: 1000  # Intervalo do loop principal
"""
    
    with open(config_path, 'w') as f:
        f.write(example_config)
    logger.info(f"Arquivo {config_path} criado com sucesso!")
