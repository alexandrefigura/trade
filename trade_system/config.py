"""
Configuração centralizada do sistema de trading - V5.3 com Hyperopt e Versionamento
"""
import os
import yaml
import logging
import hashlib
import json
import multiprocessing as mp
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Carrega variáveis do arquivo .env
load_dotenv()


@dataclass
class HyperoptConfig:
    """Configuração para otimização de hiperparâmetros"""
    enabled: bool = True
    engine: str = "optuna"  # optuna ou hyperopt
    n_trials: int = 100
    direction: str = "maximize"  # maximize ou minimize
    objective: str = "sharpe_ratio"  # métrica a otimizar
    sampler: str = "TPESampler"
    pruner: str = "MedianPruner"
    
    # Walk-forward durante otimização
    use_walk_forward: bool = False
    train_days: int = 30
    test_days: int = 7
    
    # Espaço de busca
    search_space: Dict[str, Dict] = field(default_factory=dict)
    
    # Constraints
    constraints: List[str] = field(default_factory=list)
    
    # Persistência
    save_best_params: bool = True
    params_file: str = "best_hyperparameters.json"
    save_all_trials: bool = True
    trials_database: str = "sqlite:///optuna_trials.db"
    
    # Adaptação online
    adaptive_optimization: bool = True
    reoptimize_days: int = 7
    min_performance_drop: float = 0.15
    use_online_learning: bool = True


@dataclass
class FeatureFlags:
    """Feature flags para controle granular do sistema"""
    # Análise técnica
    enable_technical: bool = True
    enable_rsi: bool = True
    enable_moving_averages: bool = True
    enable_bollinger_bands: bool = True
    enable_pattern_detection: bool = True
    enable_atr_stops: bool = True
    
    # Orderbook
    enable_orderbook: bool = True
    enable_imbalance_detection: bool = True
    enable_liquidity_analysis: bool = True
    enable_market_impact: bool = True
    
    # Machine Learning
    enable_ml: bool = True
    enable_advanced_ml: bool = True
    enable_online_learning: bool = True
    enable_feature_engineering: bool = True
    
    # Risk Management
    enable_dynamic_stops: bool = True
    enable_trailing_stop: bool = True
    enable_kill_switch: bool = True
    enable_position_sizing_kelly: bool = True
    enable_volatility_adjustment: bool = True
    
    # Sinais
    enable_signal_consolidation: bool = True
    enable_adaptive_weights: bool = True
    enable_consensus_requirement: bool = False
    require_min_signals: int = 2
    
    # Performance
    enable_redis_cache: bool = True
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = False
    
    # Alertas e Notificações
    enable_telegram_alerts: bool = True
    enable_email_alerts: bool = False
    enable_discord_alerts: bool = False
    enable_webhook_alerts: bool = False
    
    # Debug e Logging
    enable_debug_mode: bool = False
    enable_performance_profiling: bool = False
    enable_detailed_logging: bool = True
    save_all_signals: bool = True
    
    # Experimental
    enable_sentiment_analysis: bool = False
    enable_news_integration: bool = False
    enable_social_signals: bool = False
    enable_multi_exchange: bool = False


@dataclass
class ConfigVersion:
    """Informações de versão e rastreabilidade"""
    version: str = "5.3.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_hash: str = ""
    git_commit: str = ""
    environment: str = "development"  # development, staging, production
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def calculate_hash(self, config_dict: Dict) -> str:
        """Calcula hash SHA256 da configuração"""
        # Remover campos voláteis
        config_copy = config_dict.copy()
        for field in ['timestamp', 'config_hash', 'run_id']:
            config_copy.pop(field, None)
        
        # Serializar e calcular hash
        config_str = json.dumps(config_copy, sort_keys=True)
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return self.config_hash
    
    def get_git_commit(self) -> str:
        """Obtém o commit git atual se disponível"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.git_commit = result.stdout.strip()[:8]
        except:
            self.git_commit = "unknown"
        return self.git_commit


@dataclass
class UltraConfigV5:
    """Configuração completa do sistema de trading v5.3"""
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
    stop_loss_pct: float = 0.015      # Para compatibilidade
    take_profit_pct: float = 0.025    # Para compatibilidade
    trailing_stop_pct: float = 0.01   # 1% trailing
    use_dynamic_stops: bool = True
    enable_trailing: bool = True
    trailing_activation: float = 0.01
    trailing_distance: float = 0.005
    max_position_duration: int = 7200  # 2 horas
    breakeven_time: int = 1800        # 30 min

    # Kelly Criterion
    kelly_multiplier: float = 0.25
    min_kelly_fraction: float = 0.01
    max_kelly_fraction: float = 0.25
    
    # Drawdown Protection
    max_daily_drawdown: float = 0.02
    max_weekly_drawdown: float = 0.05
    max_monthly_drawdown: float = 0.10
    daily_loss_kill_switch: float = 0.03
    max_consecutive_losses: int = 3

    # ATR parameters
    atr_period: int = 14
    tp_multiplier: float = 2.0  # Aumentado para melhor R:R
    sl_multiplier: float = 1.5  # Stop mais apertado

    # Signal Weights
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        'technical': 0.40,
        'orderbook': 0.40,
        'ml': 0.20
    })

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
    min_market_score: int = 60

    # Alertas
    enable_alerts: bool = True
    telegram_token: str = ""
    telegram_chat_id: str = ""
    alert_email: str = ""

    # Orderbook
    orderbook_imbalance_threshold: float = 0.90
    min_cycles_before_trade: int = 3
    orderbook_depth_levels: int = 20

    # Anti-bias
    antibias_enabled: bool = True
    max_consecutive_signals: int = 3
    force_balance: bool = True
    max_bias_ratio: float = 0.6

    # Debug mode
    debug_mode: bool = False
    force_first_trade: bool = False
    use_simulated_data: bool = False
    
    # Sub-configurações
    hyperopt: HyperoptConfig = field(default_factory=HyperoptConfig)
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    version_info: ConfigVersion = field(default_factory=ConfigVersion)
    
    # Metadados
    _config_source: str = field(default="default", init=False)
    _load_timestamp: datetime = field(default_factory=datetime.now, init=False)


def load_config_from_yaml(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Carrega configurações do arquivo YAML com suporte completo"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            
        logger.info(f"✅ Configuração carregada de {config_path}")
        
        # Log de seções encontradas
        sections = list(yaml_data.keys())
        logger.debug(f"Seções encontradas: {sections}")
        
        return yaml_data
        
    except FileNotFoundError:
        logger.warning(f"{config_path} não encontrado, usando configurações padrão")
        return {}
    except Exception as e:
        logger.error(f"Erro ao carregar {config_path}: {e}")
        return {}


def create_debug_config() -> UltraConfigV5:
    """Cria configuração de debug BALANCEADA com features específicas"""
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
    
    # Feature flags para debug
    config.feature_flags.enable_debug_mode = True
    config.feature_flags.enable_performance_profiling = True
    config.feature_flags.save_all_signals = True
    config.feature_flags.enable_detailed_logging = True
    
    # Desabilitar features complexas em debug
    config.feature_flags.enable_advanced_ml = False
    config.feature_flags.enable_consensus_requirement = False
    
    # Hyperopt desabilitado em debug por padrão
    config.hyperopt.enabled = False
    
    # Versioning
    config.version_info.environment = "debug"
    
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
- environment: {config.version_info.environment}
    """)
    
    return config


def apply_yaml_config(config: UltraConfigV5, yaml_cfg: Dict[str, Any]) -> None:
    """Aplica configurações do YAML ao objeto config"""
    # Trading
    trading = yaml_cfg.get('trading', {})
    for key, value in trading.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Risk
    risk = yaml_cfg.get('risk', {})
    for key, value in risk.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Technical Analysis
    ta = yaml_cfg.get('ta', {})
    for key, value in ta.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Orderbook
    orderbook = yaml_cfg.get('orderbook', {})
    for key, value in orderbook.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Signal weights
    if 'signal_weights' in yaml_cfg:
        config.signal_weights = yaml_cfg['signal_weights']
    
    # Filters
    filters = yaml_cfg.get('filters', {})
    for key, value in filters.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # ML/Antibias
    ml = yaml_cfg.get('ml', {})
    for key, value in ml.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    antibias = yaml_cfg.get('antibias', {})
    for key, value in antibias.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Performance
    performance = yaml_cfg.get('performance', {})
    for key, value in performance.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Debug
    debug_section = yaml_cfg.get('debug', {})
    for key, value in debug_section.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Hyperopt configuration
    if 'hyperopt' in yaml_cfg:
        hyperopt_cfg = yaml_cfg['hyperopt']
        config.hyperopt = HyperoptConfig()
        
        for key, value in hyperopt_cfg.items():
            if hasattr(config.hyperopt, key):
                setattr(config.hyperopt, key, value)
        
        logger.info("📊 Configuração Hyperopt carregada")
    
    # Feature flags
    if 'feature_flags' in yaml_cfg:
        flags_cfg = yaml_cfg['feature_flags']
        
        for key, value in flags_cfg.items():
            if hasattr(config.feature_flags, key):
                setattr(config.feature_flags, key, value)
        
        logger.info("🚩 Feature flags carregadas")
    
    # Version info
    if 'version_info' in yaml_cfg:
        version_cfg = yaml_cfg['version_info']
        
        for key, value in version_cfg.items():
            if hasattr(config.version_info, key):
                setattr(config.version_info, key, value)


def get_config(debug_mode: bool = False, config_path: str = 'config.yaml') -> UltraConfigV5:
    """
    Retorna configuração completa do sistema com suporte para hyperopt e versionamento
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
    yaml_cfg = load_config_from_yaml(config_path)
    
    # Se YAML existe e NÃO estamos em debug mode, aplicar valores
    if yaml_cfg and not debug_mode:
        apply_yaml_config(config, yaml_cfg)
        config._config_source = "yaml"
        
        # Se YAML especifica debug_mode=true, usar config debug
        if yaml_cfg.get('debug_mode', False) and not debug_mode:
            logger.info("YAML especifica debug_mode=true, usando config debug")
            return get_config(debug_mode=True, config_path=config_path)
    
    # ENV vars SEMPRE têm prioridade máxima
    config.api_key = os.getenv('BINANCE_API_KEY', config.api_key)
    config.api_secret = os.getenv('BINANCE_API_SECRET', config.api_secret)
    config.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', config.telegram_token)
    config.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', config.telegram_chat_id)
    
    # Configurar ambiente baseado em ENV
    env = os.getenv('TRADING_ENV', 'development')
    config.version_info.environment = env
    
    # Gerar hash e metadados
    config_dict = asdict(config)
    config.version_info.calculate_hash(config_dict)
    config.version_info.get_git_commit()
    
    # Validações importantes
    if config.api_key and config.api_secret:
        logger.info("✅ Credenciais Binance carregadas")
    else:
        logger.warning("⚠️ Credenciais Binance não encontradas")
    
    # Validar configurações críticas
    validate_config(config)
    
    # Log version info
    logger.info(f"""
📋 Configuração Carregada:
- Versão: {config.version_info.version}
- Ambiente: {config.version_info.environment}
- Hash: {config.version_info.config_hash}
- Run ID: {config.version_info.run_id}
- Git: {config.version_info.git_commit}
- Fonte: {config._config_source}
    """)
    
    # Salvar snapshot da configuração
    save_config_snapshot(config)
    
    return config


def validate_config(config: UltraConfigV5) -> bool:
    """Valida e corrige configuração com validações expandidas"""
    valid = True
    warnings = []
    errors = []
    
    # RSI thresholds
    if config.rsi_buy_threshold >= config.rsi_sell_threshold:
        errors.append("RSI thresholds inválidos")
        config.rsi_buy_threshold = 30.0
        config.rsi_sell_threshold = 70.0
        valid = False
    
    # Buy/Sell thresholds devem ser positivos
    if config.sell_threshold < 0:
        warnings.append(f"sell_threshold negativo ({config.sell_threshold})")
        config.sell_threshold = abs(config.sell_threshold)
    
    # Percentuais
    if not 0 < config.min_confidence <= 1:
        errors.append(f"min_confidence inválida: {config.min_confidence}")
        config.min_confidence = 0.7
        valid = False
    
    if not 0 < config.max_position_pct <= 0.1:
        errors.append(f"max_position_pct inválida: {config.max_position_pct}")
        config.max_position_pct = 0.02
        valid = False
    
    # Stop loss vs Take profit
    if config.stop_loss_base >= config.take_profit_base:
        errors.append("stop_loss deve ser menor que take_profit")
        config.stop_loss_base = 0.01
        config.take_profit_base = 0.02
        valid = False
    
    # Signal weights devem somar 1
    if config.signal_weights:
        total = sum(config.signal_weights.values())
        if abs(total - 1.0) > 0.01:
            warnings.append(f"signal_weights somam {total:.2f}, normalizando")
            for key in config.signal_weights:
                config.signal_weights[key] /= total
    
    # Avisos de configuração
    if config.min_confidence < 0.6:
        warnings.append(f"min_confidence muito baixa: {config.min_confidence}")
    
    if config.max_position_pct > 0.05:
        warnings.append(f"max_position_pct muito alta: {config.max_position_pct}")
    
    if config.force_first_trade:
        warnings.append("force_first_trade está ATIVO")
    
    if config.daily_loss_kill_switch > 0.05:
        warnings.append(f"daily_loss_kill_switch muito alto: {config.daily_loss_kill_switch*100:.1f}%")
    
    # Feature flags warnings
    if config.feature_flags.enable_experimental_features := [
        config.feature_flags.enable_sentiment_analysis,
        config.feature_flags.enable_news_integration,
        config.feature_flags.enable_social_signals,
        config.feature_flags.enable_multi_exchange
    ]:
        if any(config.feature_flags.enable_experimental_features):
            warnings.append("Features experimentais habilitadas")
    
    # Log resultados
    if errors:
        logger.error("❌ Erros de configuração:")
        for error in errors:
            logger.error(f"  - {error}")
    
    if warnings:
        logger.warning("⚠️ Avisos de configuração:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    if valid and not warnings:
        logger.info("✅ Configuração validada sem problemas")
    
    return valid


def save_config_snapshot(config: UltraConfigV5):
    """Salva snapshot da configuração para auditoria"""
    snapshot_dir = Path('config_snapshots')
    snapshot_dir.mkdir(exist_ok=True)
    
    # Nome do arquivo com timestamp e hash
    filename = f"config_{config.version_info.run_id}_{config.version_info.config_hash}.json"
    filepath = snapshot_dir / filename
    
    # Converter para dict e salvar
    config_dict = asdict(config)
    
    # Adicionar metadados extras
    config_dict['_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'python_version': os.sys.version,
        'platform': os.sys.platform,
        'hostname': os.environ.get('HOSTNAME', 'unknown')
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.debug(f"📸 Config snapshot salvo: {filepath}")
        
        # Limpar snapshots antigos (manter últimos 100)
        snapshots = sorted(snapshot_dir.glob('config_*.json'))
        if len(snapshots) > 100:
            for old_snapshot in snapshots[:-100]:
                old_snapshot.unlink()
                
    except Exception as e:
        logger.error(f"Erro salvando snapshot: {e}")


def export_hyperopt_config(config: UltraConfigV5) -> Dict[str, Any]:
    """Exporta configuração de hyperopt para uso no CLI"""
    if not config.hyperopt.enabled:
        return {}
    
    hyperopt_config = asdict(config.hyperopt)
    
    # Adicionar parâmetros atuais como valores iniciais
    current_params = {
        'max_position_pct': config.max_position_pct,
        'stop_loss_pct': config.stop_loss_pct,
        'take_profit_pct': config.take_profit_pct,
        'trailing_stop_pct': config.trailing_stop_pct,
        'min_confidence': config.min_confidence,
        'rsi_period': config.rsi_period,
        'rsi_buy_threshold': config.rsi_buy_threshold,
        'rsi_sell_threshold': config.rsi_sell_threshold,
        'sma_short_period': config.sma_short_period,
        'sma_long_period': config.sma_long_period,
        'bb_period': config.bb_period,
        'bb_std_dev': config.bb_std_dev,
        'orderbook_imbalance_threshold': config.orderbook_imbalance_threshold,
        'signal_weights': config.signal_weights
    }
    
    hyperopt_config['current_params'] = current_params
    
    logger.info(f"📊 Configuração Hyperopt exportada: {config.hyperopt.n_trials} trials")
    
    return hyperopt_config


def get_feature_flags(config: UltraConfigV5) -> FeatureFlags:
    """Retorna feature flags ativas"""
    return config.feature_flags


def create_example_config(config_path: str = 'config.yaml'):
    """Cria um arquivo config.yaml de exemplo completo"""
    example_yaml = """# Sistema de Trading v5.3 - Configuração Completa
# Com suporte para Hyperopt e Feature Flags

# Modo de operação
debug_mode: false

# Trading Principal
trading:
  symbol: "BTCUSDT"
  min_confidence: 0.65
  max_position_pct: 0.02

# Análise Técnica
ta:
  rsi_period: 14
  rsi_buy_threshold: 30
  rsi_sell_threshold: 70
  rsi_confidence: 0.6
  
  sma_short_period: 12
  sma_long_period: 26
  ema_short_period: 12
  ema_long_period: 26
  sma_cross_confidence: 0.65
  
  bb_period: 20
  bb_std_dev: 2.5
  bb_confidence: 0.6
  
  pattern_confidence: 0.7
  buy_threshold: 0.45
  sell_threshold: 0.45

# Orderbook
orderbook:
  imbalance_threshold: 0.90
  min_cycles_before_trade: 3
  depth_levels: 20

# Pesos dos Sinais
signal_weights:
  technical: 0.40
  orderbook: 0.40
  ml: 0.20

# Gestão de Risco
risk:
  stop_loss_pct: 0.015
  take_profit_pct: 0.025
  trailing_stop_pct: 0.01
  
  max_daily_drawdown: 0.02
  max_weekly_drawdown: 0.05
  max_monthly_drawdown: 0.10
  daily_loss_kill_switch: 0.03
  max_consecutive_losses: 5
  
  kelly_multiplier: 0.25
  min_kelly_fraction: 0.01
  max_kelly_fraction: 0.25

# Otimização de Hiperparâmetros
hyperopt:
  enabled: true
  engine: "optuna"
  n_trials: 100
  direction: "maximize"
  objective: "sharpe_ratio"
  sampler: "TPESampler"
  pruner: "MedianPruner"
  
  # Walk-forward durante otimização
  use_walk_forward: false
  train_days: 30
  test_days: 7
  
  # Espaço de busca
  search_space:
    rsi_period:
      type: "int"
      low: 10
      high: 30
      step: 2
    
    max_position_pct:
      type: "float"
      low: 0.005
      high: 0.05
      step: 0.005
    
    stop_loss_pct:
      type: "float"
      low: 0.005
      high: 0.03
      step: 0.005
  
  # Persistência
  save_best_params: true
  params_file: "best_hyperparameters.json"
  trials_database: "sqlite:///optuna_trials.db"
  
  # Otimização adaptativa
  adaptive_optimization: true
  reoptimize_days: 7
  min_performance_drop: 0.15

# Feature Flags
feature_flags:
  # Análise técnica
  enable_technical: true
  enable_rsi: true
  enable_moving_averages: true
  enable_bollinger_bands: true
  enable_pattern_detection: true
  enable_atr_stops: true
  
  # Orderbook
  enable_orderbook: true
  enable_imbalance_detection: true
  enable_liquidity_analysis: true
  
  # Machine Learning
  enable_ml: true
  enable_advanced_ml: true
  enable_online_learning: true
  
  # Risk Management
  enable_dynamic_stops: true
  enable_trailing_stop: true
  enable_kill_switch: true
  enable_position_sizing_kelly: true
  
  # Sinais
  enable_signal_consolidation: true
  enable_adaptive_weights: true
  enable_consensus_requirement: false
  require_min_signals: 2
  
  # Performance
  enable_redis_cache: true
  enable_parallel_processing: true
  
  # Alertas
  enable_telegram_alerts: true
  enable_email_alerts: false
  
  # Debug
  enable_debug_mode: false
  enable_performance_profiling: false
  save_all_signals: true
  
  # Experimental
  enable_sentiment_analysis: false
  enable_news_integration: false

# Informações de Versão
version_info:
  version: "5.3.0"
  environment: "production"  # development, staging, production

# Filtros de Mercado
filters:
  min_volume_multiplier: 1.5
  max_spread_bps: 20
  max_recent_volatility: 0.03

# Performance
performance:
  ta_interval_ms: 5000
  main_loop_interval_ms: 1000
  orderbook_update_ms: 100
  signal_cooldown_ms: 10000

# Validação de Mercado
market_validation:
  check_interval: 60
  min_market_score: 60
  required_checks:
    - volume_above_average
    - spread_acceptable
    - volatility_in_range
    - orderbook_depth_sufficient

# Alertas (usar variáveis de ambiente)
alerts:
  enabled: true
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"

# Debug
debug:
  force_first_trade: false
  use_simulated_data: false
  save_all_signals: true
  log_orderbook_changes: false
"""
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(example_yaml)
    
    logger.info(f"✅ Arquivo {config_path} criado com sucesso!")
    logger.info("📝 Config v5.3 com Hyperopt e Feature Flags")
    
    # Criar arquivo de feature flags separado
    feature_flags_path = Path('feature_flags.yaml')
    if not feature_flags_path.exists():
        feature_flags_yaml = """# Feature Flags - Controle granular do sistema
# Pode ser usado para A/B testing e rollout gradual

# Flags de produção
production:
  enable_kill_switch: true
  enable_position_sizing_kelly: true
  enable_adaptive_weights: true
  enable_consensus_requirement: false
  max_position_pct_override: null

# Flags de staging
staging:
  enable_advanced_ml: true
  enable_online_learning: true
  enable_sentiment_analysis: true
  enable_experimental_features: true

# Flags de desenvolvimento
development:
  enable_debug_mode: true
  enable_performance_profiling: true
  save_all_signals: true
  enable_detailed_logging: true
  force_trades_in_debug: false

# A/B Testing
ab_tests:
  test_new_ml_model:
    enabled: false
    percentage: 10  # 10% dos usuários
    flags:
      enable_advanced_ml: true
      ml_model_version: "v2"
  
  test_aggressive_trading:
    enabled: false
    percentage: 5  # 5% dos usuários
    flags:
      max_position_pct_override: 0.05
      min_confidence_override: 0.55
"""
        
        with open(feature_flags_path, 'w') as f:
            f.write(feature_flags_yaml)
        
        logger.info("📝 Arquivo feature_flags.yaml criado")


# Manter compatibilidade com versão anterior
def get_hyperopt_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Função auxiliar para obter apenas configuração de hyperopt"""
    config = get_config(config_path=config_path)
    return export_hyperopt_config(config)