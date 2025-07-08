"""
Configuração centralizada do sistema de trading - V5.2 Balanceada
"""
import os
import yaml
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Carrega variáveis do arquivo .env
load_dotenv()


@dataclass
class UltraConfigV5:
    """Configuração balanceada para trading em produção"""
    # APIs
    api_key: str = ""
    api_secret: str = ""
    
    # Trading
    symbol: str = "BTCUSDT"
    min_confidence: float = 0.70  # 70% para produção
    max_position_pct: float = 0.015  # 1.5% conservador

    # Parâmetros de Technical Analysis - BALANCEADOS
    ta_interval_ms: int = 5000
    sma_short_period: int = 12
    sma_long_period: int = 26
    ema_short_period: int = 12
    ema_long_period: int = 26
    rsi_period: int = 14
    rsi_buy_threshold: float = 30.0   # Real oversold
    rsi_sell_threshold: float = 70.0  # Real overbought
    rsi_confidence: float = 0.6
    sma_cross_confidence: float = 0.65
    bb_period: int = 20
    bb_std_dev: float = 2.5
    bb_confidence: float = 0.6
    pattern_confidence: float = 0.7
    buy_threshold: float = 0.45    # Mais conservador
    sell_threshold: float = 0.45   # POSITIVO para simetria

    # Filtros
    min_volume_multiplier: float = 1.5
    max_recent_volatility: float = 0.025

    # Risk Management - DINÂMICO
    stop_loss_base: float = 0.01      # 1% base
    take_profit_base: float = 0.02    # 2% base
    use_dynamic_stops: bool = True
    enable_trailing: bool = True
    trailing_activation: float = 0.01
    trailing_distance: float = 0.005
    max_position_duration: int = 7200  # 2 horas
    breakeven_time: int = 1800        # 30 min

    # ATR parameters
    atr_period: int = 14
    tp_multiplier: float = 2.0  # Aumentado para melhor R:R
    sl_multiplier: float = 1.5  # Stop mais apertado

    # Performance
    use_redis: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Processamento
    num_workers: int = mp.cpu_count()
    batch_size: int = 1000

    # Buffers
    price_buffer_size: int = 10000
    orderbook_buffer_size: int = 100

    # Timing
    main_loop_interval_ms: int = 1000
    signal_cooldown_ms: int = 10000  # 10s entre sinais
    gc_interval_cycles: int = 1000

    # Rate limiting
    rate_limit_window: int = 60
    rate_limit_max_calls: int = 1200

    # Proteções de mercado
    max_volatility: float = 0.04  # 4%
    max_spread_bps: float = 15.0
    min_volume_24h: int = 1_000_000
    min_liquidity_depth: float = 100_000

    # Alertas
    enable_alerts: bool = True
    telegram_token: str = ""
    telegram_chat_id: str = ""
    alert_email: str = ""

    # Risk management
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.08
    max_consecutive_losses: int = 3

    # Anti-bias
    antibias_enabled: bool = True
    max_consecutive_signals: int = 3
    force_balance: bool = True
    max_bias_ratio: float = 0.6

    # Debug mode
    debug_mode: bool = False
    force_first_trade: bool = False
    use_simulated_data: bool = False


def load_config_from_yaml(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Carrega configurações do arquivo YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"{config_path} não encontrado, usando configurações padrão")
        return {}
    except Exception as e:
        logger.error(f"Erro ao carregar {config_path}: {e}")
        return {}


def create_debug_config() -> UltraConfigV5:
    """Cria configuração de debug BALANCEADA"""
    config = UltraConfigV5()
    
    # Debug mais moderado - não ultra agressivo
    config.min_confidence = 0.60  # 60% em debug
    
    # RSI - valores razoáveis
    config.rsi_buy_threshold = 35.0   # Oversold moderado
    config.rsi_sell_threshold = 65.0  # Overbought moderado
    config.rsi_confidence = 0.5
    
    # Outros indicadores
    config.bb_confidence = 0.5
    config.sma_cross_confidence = 0.5
    config.pattern_confidence = 0.5
    
    # Thresholds simétricos
    config.buy_threshold = 0.3
    config.sell_threshold = 0.3  # POSITIVO
    
    # Filtros mais relaxados mas não desativados
    config.min_volume_multiplier = 1.0
    config.max_recent_volatility = 0.05
    config.max_volatility = 0.05
    
    # Anti-bias ativo mesmo em debug
    config.antibias_enabled = True
    config.max_consecutive_signals = 5  # Um pouco mais permissivo
    
    # Flags de debug
    config.debug_mode = True
    config.force_first_trade = False  # NÃO forçar
    config.use_simulated_data = False
    
    # Intervalos
    config.ta_interval_ms = 3000      # 3s
    config.main_loop_interval_ms = 1000  # 1s
    
    # Risk
    config.max_position_pct = 0.02    # 2% em debug
    config.max_daily_loss = 0.05      # 5% em debug
    
    logger.warning("⚠️ MODO DEBUG - Parâmetros moderados ativados")
    logger.info(f"""
🔧 DEBUG CONFIG (BALANCEADO):
- min_confidence: {config.min_confidence}
- rsi_buy: {config.rsi_buy_threshold} / rsi_sell: {config.rsi_sell_threshold}
- buy_threshold: {config.buy_threshold} / sell_threshold: {config.sell_threshold}
- anti-bias: ATIVO
- force_first_trade: {config.force_first_trade}
    """)
    
    return config


def get_config(debug_mode: bool = False) -> UltraConfigV5:
    """
    Retorna configuração completa do sistema
    Prioridade: ENV > YAML > Debug > Default
    """
    # Carrega .env primeiro
    load_dotenv()
    
    # Base config
    if debug_mode:
        config = create_debug_config()
    else:
        config = UltraConfigV5()
    
    # Tentar carregar YAML
    yaml_cfg = load_config_from_yaml()
    
    # Se YAML existe e NÃO estamos em debug mode, aplicar valores
    if yaml_cfg and not debug_mode:
        # Trading
        trading = yaml_cfg.get('trading', {})
        config.symbol = trading.get('symbol', config.symbol)
        config.min_confidence = trading.get('min_confidence', config.min_confidence)
        config.max_position_pct = trading.get('max_position_pct', config.max_position_pct)
        
        # Risk
        risk = yaml_cfg.get('risk', {})
        config.stop_loss_base = risk.get('stop_loss_base', config.stop_loss_base)
        config.take_profit_base = risk.get('take_profit_base', config.take_profit_base)
        config.use_dynamic_stops = risk.get('use_dynamic_stops', config.use_dynamic_stops)
        config.enable_trailing = risk.get('enable_trailing', config.enable_trailing)
        config.max_volatility = risk.get('max_volatility', config.max_volatility)
        config.max_daily_loss = risk.get('max_daily_loss', config.max_daily_loss)
        config.max_drawdown = risk.get('max_drawdown', config.max_drawdown)
        
        # TA parameters
        ta = yaml_cfg.get('ta', {})
        for key, value in ta.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Garantir que sell_threshold é positivo
        if hasattr(config, 'sell_threshold') and config.sell_threshold < 0:
            logger.warning(f"sell_threshold negativo ({config.sell_threshold}) convertido para positivo")
            config.sell_threshold = abs(config.sell_threshold)
        
        # Filters
        filters = yaml_cfg.get('filters', {})
        config.min_volume_multiplier = filters.get('min_volume_multiplier', config.min_volume_multiplier)
        config.max_recent_volatility = filters.get('max_recent_volatility', config.max_recent_volatility)
        config.min_liquidity_depth = filters.get('min_liquidity_depth', config.min_liquidity_depth)
        
        # ML
        ml = yaml_cfg.get('ml', {})
        config.force_balance = ml.get('force_balance', config.force_balance)
        config.max_bias_ratio = ml.get('max_bias_ratio', config.max_bias_ratio)
        
        # Anti-bias
        antibias = yaml_cfg.get('antibias', {})
        config.antibias_enabled = antibias.get('enabled', config.antibias_enabled)
        config.max_consecutive_signals = antibias.get('max_consecutive_signals', config.max_consecutive_signals)
        
        # Performance
        performance = yaml_cfg.get('performance', {})
        config.ta_interval_ms = performance.get('ta_interval_ms', config.ta_interval_ms)
        config.main_loop_interval_ms = performance.get('main_loop_interval_ms', config.main_loop_interval_ms)
        config.signal_cooldown_ms = performance.get('signal_cooldown_ms', config.signal_cooldown_ms)
        
        # Debug
        debug_section = yaml_cfg.get('debug', {})
        config.force_first_trade = debug_section.get('force_first_trade', config.force_first_trade)
        config.use_simulated_data = debug_section.get('use_simulated_data', config.use_simulated_data)
        
        # Se YAML especifica debug_mode=true, usar config debug
        if yaml_cfg.get('debug_mode', False) and not debug_mode:
            logger.info("YAML especifica debug_mode=true, usando config debug")
            return get_config(debug_mode=True)
    
    # ENV vars SEMPRE têm prioridade máxima
    config.api_key = os.getenv('BINANCE_API_KEY', config.api_key)
    config.api_secret = os.getenv('BINANCE_API_SECRET', config.api_secret)
    config.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', config.telegram_token)
    config.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', config.telegram_chat_id)
    
    # Validações importantes
    if config.api_key and config.api_secret:
        logger.info("✅ Credenciais Binance carregadas")
    else:
        logger.warning("⚠️ Credenciais Binance não encontradas")
    
    # Validar configurações críticas
    validate_config(config)
    
    return config


def validate_config(config: UltraConfigV5) -> bool:
    """Valida e corrige configuração"""
    valid = True
    
    # RSI thresholds
    if config.rsi_buy_threshold >= config.rsi_sell_threshold:
        logger.error("RSI thresholds inválidos - corrigindo")
        config.rsi_buy_threshold = 30.0
        config.rsi_sell_threshold = 70.0
        valid = False
    
    # Buy/Sell thresholds devem ser positivos
    if config.sell_threshold < 0:
        logger.warning(f"sell_threshold negativo ({config.sell_threshold}) - corrigindo para positivo")
        config.sell_threshold = abs(config.sell_threshold)
    
    # Percentuais
    if not 0 < config.min_confidence <= 1:
        logger.error(f"min_confidence inválida: {config.min_confidence}")
        config.min_confidence = 0.7
        valid = False
    
    if not 0 < config.max_position_pct <= 0.1:
        logger.error(f"max_position_pct inválida: {config.max_position_pct}")
        config.max_position_pct = 0.02
        valid = False
    
    # Avisos
    if config.min_confidence < 0.6:
        logger.warning(f"⚠️ min_confidence muito baixa: {config.min_confidence}")
    
    if config.max_position_pct > 0.05:
        logger.warning(f"⚠️ max_position_pct muito alta: {config.max_position_pct}")
    
    if config.force_first_trade:
        logger.warning("⚠️ force_first_trade está ATIVO - pode gerar trades forçados")
    
    return valid


def create_example_config(config_path: str = 'config.yaml'):
    """Cria um arquivo config.yaml de exemplo balanceado"""
    with open('config_template.yaml', 'r', encoding='utf-8') as template:
        example_config = template.read()
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(example_config)
    
    logger.info(f"✅ Arquivo {config_path} criado com sucesso!")
    logger.info("📝 Config balanceada para evitar viés de sinais")