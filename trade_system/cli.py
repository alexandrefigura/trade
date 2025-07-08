"""
Interface de linha de comando aprimorada com otimização automática
"""
import os
import sys
import asyncio
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_system.config import get_config, create_example_config
from trade_system.logging_config import setup_logging, get_logger

# Importação condicional do Optuna
try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = get_logger(__name__)


def create_parser():
    parser = argparse.ArgumentParser(
        description='''
╔═══════════════════════════════════════════════════════════╗
║         Sistema de Trading Ultra-Otimizado v5.3           ║
║                  With Hyperparameter Tuning               ║
╚═══════════════════════════════════════════════════════════╝
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
    
    # Optimize (novo)
    optimize_parser = subparsers.add_parser(
        'optimize', 
        help='Otimiza hiperparâmetros usando Optuna',
        description='Executa otimização automática de parâmetros'
    )
    optimize_parser.add_argument('--trials', type=int, default=50, 
                                help='Número de trials (padrão: 50)')
    optimize_parser.add_argument('--metric', 
                                choices=['sharpe', 'return', 'profit_factor', 'calmar', 'sortino'],
                                default='sharpe',
                                help='Métrica a otimizar (padrão: sharpe)')
    optimize_parser.add_argument('--days', type=int, default=30,
                                help='Dias de dados para otimização (padrão: 30)')
    optimize_parser.add_argument('--walk-forward', action='store_true',
                                help='Usar walk-forward durante otimização')
    optimize_parser.add_argument('--parallel', action='store_true',
                                help='Executar trials em paralelo')
    optimize_parser.add_argument('--study-name', type=str,
                                help='Nome do estudo (para continuar otimização)')
    optimize_parser.add_argument('--storage', type=str,
                                help='URL do banco de dados (ex: sqlite:///optuna.db)')
    optimize_parser.add_argument('--save-config', action='store_true',
                                help='Salvar melhores parâmetros em config.yaml')
    optimize_parser.add_argument('--export-report', action='store_true',
                                help='Exportar relatório detalhado')
    
    # Walk-forward (novo)
    walk_parser = subparsers.add_parser(
        'walk-forward',
        help='Executa análise walk-forward completa',
        description='Walk-forward analysis com múltiplas janelas'
    )
    walk_parser.add_argument('--days', type=int, default=90,
                            help='Total de dias de dados (padrão: 90)')
    walk_parser.add_argument('--train-days', type=int, default=30,
                            help='Dias para treinamento (padrão: 30)')
    walk_parser.add_argument('--test-days', type=int, default=7,
                            help='Dias para teste (padrão: 7)')
    walk_parser.add_argument('--step-days', type=int, default=1,
                            help='Dias para avançar (padrão: 1)')
    walk_parser.add_argument('--parallel', action='store_true',
                            help='Processar janelas em paralelo')
    
    # Backtest
    backtest_parser = subparsers.add_parser('backtest', help='Executa backtest da estratégia')
    backtest_parser.add_argument('--debug', action='store_true', help='Modo debug')
    backtest_parser.add_argument('--days', type=int, default=7, help='Dias de dados (padrão: 7)')
    backtest_parser.add_argument('--symbol', type=str, help='Par de trading')
    backtest_parser.add_argument('--config', type=str, help='Arquivo de configuração customizado')
    
    # Paper Trading
    paper_parser = subparsers.add_parser('paper', help='Inicia paper trading')
    paper_parser.add_argument('--debug', action='store_true', help='Modo debug')
    paper_parser.add_argument('--no-backtest', action='store_true', help='Pular backtest')
    paper_parser.add_argument('--balance', type=float, default=10000, help='Balance inicial')
    paper_parser.add_argument('--config', type=str, help='Arquivo de configuração customizado')
    
    # Config
    config_parser = subparsers.add_parser('config', help='Gerenciar configurações')
    config_parser.add_argument('--create', action='store_true', help='Criar config.yaml')
    config_parser.add_argument('--show', action='store_true', help='Mostrar configuração')
    config_parser.add_argument('--validate', action='store_true', help='Validar configuração')
    
    # Dashboard (novo)
    dashboard_parser = subparsers.add_parser(
        'dashboard',
        help='Visualizar resultados e métricas',
        description='Dashboard interativo com resultados'
    )
    dashboard_parser.add_argument('--port', type=int, default=8080,
                                 help='Porta para o servidor web (padrão: 8080)')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Nível de logging')
    
    return parser


async def run_optimize_command(args):
    """Executa otimização de hiperparâmetros com Optuna"""
    if not OPTUNA_AVAILABLE:
        print("❌ ERRO: Optuna não está instalado!")
        print("Instale com: pip install optuna")
        sys.exit(1)
    
    print("="*60)
    print("           OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("="*60)
    
    print(f"Trials: {args.trials}")
    print(f"Métrica: {args.metric}")
    print(f"Período: {args.days} dias")
    print(f"Walk-forward: {'SIM' if args.walk_forward else 'NÃO'}")
    print(f"Paralelo: {'SIM' if args.parallel else 'NÃO'}")
    print()
    
    # Configurar Optuna
    if args.study_name:
        study_name = args.study_name
    else:
        study_name = f"trading_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if args.storage:
        storage = args.storage
    else:
        storage = f"sqlite:///optuna_studies.db"
    
    # Configurar logging do Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Criar ou carregar estudo
    print(f"📊 Criando/carregando estudo: {study_name}")
    
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        print(f"✅ Estudo carregado. Trials existentes: {len(study.trials)}")
        
    except Exception as e:
        logger.error(f"Erro criando estudo: {e}")
        sys.exit(1)
    
    # Função objetivo
    objective = create_objective_function(args)
    
    # Callback para mostrar progresso
    def show_progress(study, trial):
        print(f"\rTrial {trial.number + 1}/{args.trials} | "
              f"Melhor {args.metric}: {study.best_value:.4f}", end="")
    
    # Executar otimização
    print("\n🔄 Iniciando otimização...\n")
    
    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            n_jobs=4 if args.parallel else 1,
            callbacks=[show_progress],
            show_progress_bar=True
        )
        
        print("\n\n✅ Otimização concluída!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Otimização interrompida pelo usuário")
    
    # Mostrar resultados
    print_optimization_results(study, args.metric)
    
    # Salvar melhores parâmetros
    if args.save_config:
        save_best_params_to_config(study)
    
    # Exportar relatório
    if args.export_report:
        export_optimization_report(study, args.metric)
    
    # Gerar visualizações
    generate_optimization_plots(study, study_name)


def create_objective_function(args):
    """Cria função objetivo para o Optuna"""
    from trade_system.backtester import AdvancedBacktester, BacktestConfig
    from trade_system.config import get_config
    from binance.client import Client
    import pandas as pd
    
    # Carregar dados uma vez
    config = get_config()
    client = Client(config.api_key, config.api_secret)
    
    # Obter dados históricos
    interval = Client.KLINE_INTERVAL_15MINUTE
    limit = min(args.days * 96, 1000)
    
    klines = client.get_klines(
        symbol=config.symbol,
        interval=interval,
        limit=limit
    )
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    print(f"📊 Dados carregados: {len(df)} candles")
    
    def objective(trial):
        """Função objetivo para otimização"""
        # Sugerir hiperparâmetros
        params = suggest_hyperparameters(trial)
        
        # Criar configuração modificada
        config_copy = get_config()
        
        # Aplicar parâmetros sugeridos
        for key, value in params.items():
            if hasattr(config_copy, key):
                setattr(config_copy, key, value)
        
        # Configurar backtester
        backtest_config = BacktestConfig(
            max_position_pct=params.get('max_position_pct', 0.02),
            slippage_pct=0.0005,
            export_metrics=False,
            generate_plots=False
        )
        
        backtester = AdvancedBacktester(backtest_config, config_copy)
        
        # Executar backtest
        try:
            if args.walk_forward:
                # Walk-forward
                results = asyncio.run(
                    backtester.run_walk_forward_analysis(
                        df, 
                        initial_balance=10000,
                        parallel=False
                    )
                )
                
                if results and 'summary' in results:
                    summary = results['summary']
                    
                    # Selecionar métrica
                    if args.metric == 'sharpe':
                        value = summary.get('sharpe_mean', 0)
                    elif args.metric == 'return':
                        value = summary.get('return_mean', 0)
                    elif args.metric == 'profit_factor':
                        value = summary.get('profit_factor', 0)
                    elif args.metric == 'calmar':
                        value = summary.get('return_mean', 0) / max(summary.get('max_drawdown', 0.01), 0.01)
                    elif args.metric == 'sortino':
                        value = summary.get('sortino_mean', summary.get('sharpe_mean', 0))
                    else:
                        value = 0
                else:
                    value = 0
                    
            else:
                # Backtest simples
                result = backtester._run_backtest_core(
                    df,
                    initial_balance=10000,
                    ml_predictor=None
                )
                
                # Selecionar métrica
                if args.metric == 'sharpe':
                    value = result.get('sharpe_ratio', 0)
                elif args.metric == 'return':
                    value = result.get('total_return', 0)
                elif args.metric == 'profit_factor':
                    value = result.get('profit_factor', 0)
                elif args.metric == 'calmar':
                    value = result.get('calmar_ratio', 0)
                elif args.metric == 'sortino':
                    value = result.get('sortino_ratio', 0)
                else:
                    value = 0
            
            # Penalizar se não houver trades
            if result.get('num_trades', 0) < 5:
                value *= 0.1
            
            return value
            
        except Exception as e:
            logger.error(f"Erro no trial {trial.number}: {e}")
            return 0
    
    return objective


def suggest_hyperparameters(trial):
    """Sugere hiperparâmetros para o trial"""
    params = {
        # Parâmetros de risco
        'max_position_pct': trial.suggest_float('max_position_pct', 0.005, 0.05, step=0.005),
        'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.005, 0.03, step=0.005),
        'take_profit_pct': trial.suggest_float('take_profit_pct', 0.01, 0.05, step=0.005),
        'trailing_stop_pct': trial.suggest_float('trailing_stop_pct', 0.005, 0.02, step=0.005),
        
        # Parâmetros de sinal
        'min_confidence': trial.suggest_float('min_confidence', 0.5, 0.8, step=0.05),
        
        # Parâmetros técnicos
        'rsi_period': trial.suggest_int('rsi_period', 10, 30, step=2),
        'rsi_buy_threshold': trial.suggest_int('rsi_buy_threshold', 20, 40, step=5),
        'rsi_sell_threshold': trial.suggest_int('rsi_sell_threshold', 60, 80, step=5),
        
        'sma_short_period': trial.suggest_int('sma_short_period', 5, 15),
        'sma_long_period': trial.suggest_int('sma_long_period', 20, 50, step=5),
        
        'bb_period': trial.suggest_int('bb_period', 15, 25),
        'bb_std_dev': trial.suggest_float('bb_std_dev', 1.5, 2.5, step=0.1),
        
        # Parâmetros de orderbook
        'orderbook_imbalance_threshold': trial.suggest_float('orderbook_imbalance_threshold', 0.8, 0.95, step=0.05),
        'min_cycles_before_trade': trial.suggest_int('min_cycles_before_trade', 1, 5),
        
        # Pesos dos sinais
        'signal_weight_technical': trial.suggest_float('signal_weight_technical', 0.2, 0.6, step=0.1),
        'signal_weight_orderbook': trial.suggest_float('signal_weight_orderbook', 0.2, 0.6, step=0.1),
        'signal_weight_ml': trial.suggest_float('signal_weight_ml', 0.1, 0.4, step=0.1),
    }
    
    # Garantir que os pesos somem 1
    total_weight = (params['signal_weight_technical'] + 
                   params['signal_weight_orderbook'] + 
                   params['signal_weight_ml'])
    
    params['signal_weight_technical'] /= total_weight
    params['signal_weight_orderbook'] /= total_weight
    params['signal_weight_ml'] /= total_weight
    
    # Garantir stop loss < take profit
    if params['stop_loss_pct'] >= params['take_profit_pct']:
        params['take_profit_pct'] = params['stop_loss_pct'] * 1.5
    
    # Garantir SMA curta < SMA longa
    if params['sma_short_period'] >= params['sma_long_period']:
        params['sma_long_period'] = params['sma_short_period'] + 10
    
    return params


def print_optimization_results(study, metric):
    """Imprime resultados da otimização"""
    print("\n" + "="*60)
    print("              RESULTADOS DA OTIMIZAÇÃO")
    print("="*60)
    
    print(f"\n📊 Estatísticas:")
    print(f"  - Trials completados: {len(study.trials)}")
    print(f"  - Melhor {metric}: {study.best_value:.4f}")
    print(f"  - Trial do melhor resultado: {study.best_trial.number}")
    
    print(f"\n🏆 Melhores parâmetros:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")
    
    # Top 5 trials
    print(f"\n📈 Top 5 trials:")
    df_trials = study.trials_dataframe()
    top_trials = df_trials.nlargest(5, 'value')[['number', 'value', 'datetime_complete']]
    
    for idx, row in top_trials.iterrows():
        print(f"  {int(row['number']):3d} | {metric}: {row['value']:.4f} | {row['datetime_complete']}")
    
    # Importância dos parâmetros
    try:
        importances = optuna.importance.get_param_importances(study)
        
        print(f"\n🎯 Importância dos parâmetros:")
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        for param, importance in sorted_importances[:10]:
            bar_length = int(importance * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {param:30s} [{bar}] {importance:.2%}")
            
    except Exception as e:
        logger.debug(f"Não foi possível calcular importância: {e}")


def save_best_params_to_config(study):
    """Salva os melhores parâmetros no config.yaml"""
    print("\n💾 Salvando melhores parâmetros...")
    
    # Carregar config atual
    config_path = Path('config.yaml')
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {}
    
    # Mapear parâmetros para estrutura do config
    best_params = study.best_params
    
    # Criar backup
    backup_path = config_path.with_suffix('.yaml.bak')
    if config_path.exists():
        import shutil
        shutil.copy(config_path, backup_path)
        print(f"  - Backup criado: {backup_path}")
    
    # Atualizar configuração
    if 'trading' not in config_data:
        config_data['trading'] = {}
    
    config_data['trading']['min_confidence'] = best_params.get('min_confidence', 0.65)
    config_data['trading']['max_position_pct'] = best_params.get('max_position_pct', 0.02)
    
    # Análise técnica
    if 'ta' not in config_data:
        config_data['ta'] = {}
    
    config_data['ta']['rsi_period'] = best_params.get('rsi_period', 14)
    config_data['ta']['rsi_buy_threshold'] = best_params.get('rsi_buy_threshold', 30)
    config_data['ta']['rsi_sell_threshold'] = best_params.get('rsi_sell_threshold', 70)
    config_data['ta']['sma_short_period'] = best_params.get('sma_short_period', 9)
    config_data['ta']['sma_long_period'] = best_params.get('sma_long_period', 21)
    config_data['ta']['bb_period'] = best_params.get('bb_period', 20)
    config_data['ta']['bb_std_dev'] = best_params.get('bb_std_dev', 2.0)
    
    # Orderbook
    if 'orderbook' not in config_data:
        config_data['orderbook'] = {}
    
    config_data['orderbook']['imbalance_threshold'] = best_params.get('orderbook_imbalance_threshold', 0.9)
    config_data['orderbook']['min_cycles_before_trade'] = best_params.get('min_cycles_before_trade', 3)
    
    # Pesos dos sinais
    if 'signal_weights' not in config_data:
        config_data['signal_weights'] = {}
    
    config_data['signal_weights']['technical'] = round(best_params.get('signal_weight_technical', 0.4), 2)
    config_data['signal_weights']['orderbook'] = round(best_params.get('signal_weight_orderbook', 0.4), 2)
    config_data['signal_weights']['ml'] = round(best_params.get('signal_weight_ml', 0.2), 2)
    
    # Risk management
    if 'risk' not in config_data:
        config_data['risk'] = {}
    
    config_data['risk']['stop_loss_pct'] = best_params.get('stop_loss_pct', 0.015)
    config_data['risk']['take_profit_pct'] = best_params.get('take_profit_pct', 0.025)
    config_data['risk']['trailing_stop_pct'] = best_params.get('trailing_stop_pct', 0.01)
    
    # Adicionar comentário com informações da otimização
    config_data['_optimization_info'] = {
        'timestamp': datetime.now().isoformat(),
        'best_value': study.best_value,
        'metric': study.best_trial.user_attrs.get('metric', 'sharpe'),
        'trials': len(study.trials),
        'study_name': study.study_name
    }
    
    # Salvar config
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Configuração atualizada: {config_path}")
    print("  - Use --validate para verificar a nova configuração")


def export_optimization_report(study, metric):
    """Exporta relatório detalhado da otimização"""
    print("\n📄 Exportando relatório...")
    
    # Criar diretório
    report_dir = Path('optimization_reports')
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON com todos os dados
    report_data = {
        'study_name': study.study_name,
        'metric': metric,
        'timestamp': timestamp,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
        'total_trials': len(study.trials),
        'trials': []
    }
    
    # Adicionar dados de cada trial
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() 
                       if trial.datetime_complete and trial.datetime_start else None
        }
        report_data['trials'].append(trial_data)
    
    # Salvar JSON
    json_file = report_dir / f'optimization_report_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"  - JSON: {json_file}")
    
    # CSV resumido
    import pandas as pd
    
    df_trials = study.trials_dataframe()
    csv_file = report_dir / f'optimization_trials_{timestamp}.csv'
    df_trials.to_csv(csv_file, index=False)
    
    print(f"  - CSV: {csv_file}")
    
    # Markdown com resumo
    md_file = report_dir / f'optimization_summary_{timestamp}.md'
    
    with open(md_file, 'w') as f:
        f.write(f"# Relatório de Otimização\n\n")
        f.write(f"**Data**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Estudo**: {study.study_name}\n")
        f.write(f"**Métrica**: {metric}\n")
        f.write(f"**Trials**: {len(study.trials)}\n\n")
        
        f.write(f"## Melhor Resultado\n\n")
        f.write(f"- **Valor**: {study.best_value:.4f}\n")
        f.write(f"- **Trial**: #{study.best_trial.number}\n\n")
        
        f.write(f"### Parâmetros Ótimos\n\n")
        f.write("| Parâmetro | Valor |\n")
        f.write("|-----------|-------|\n")
        
        for key, value in sorted(study.best_params.items()):
            if isinstance(value, float):
                f.write(f"| {key} | {value:.4f} |\n")
            else:
                f.write(f"| {key} | {value} |\n")
        
        # Importância dos parâmetros
        try:
            importances = optuna.importance.get_param_importances(study)
            
            f.write(f"\n## Importância dos Parâmetros\n\n")
            f.write("| Parâmetro | Importância |\n")
            f.write("|-----------|------------|\n")
            
            for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {param} | {importance:.2%} |\n")
                
        except:
            pass
    
    print(f"  - Markdown: {md_file}")


def generate_optimization_plots(study, study_name):
    """Gera visualizações da otimização"""
    print("\n📊 Gerando visualizações...")
    
    plots_dir = Path('optimization_plots')
    plots_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # História da otimização
        fig = plot_optimization_history(study)
        fig.write_html(plots_dir / f'optimization_history_{timestamp}.html')
        print(f"  - História: optimization_history_{timestamp}.html")
        
        # Importância dos parâmetros
        if len(study.trials) > 10:
            fig = plot_param_importances(study)
            fig.write_html(plots_dir / f'param_importances_{timestamp}.html')
            print(f"  - Importância: param_importances_{timestamp}.html")
            
    except Exception as e:
        logger.debug(f"Erro gerando plots: {e}")
        print("  - Visualizações requerem mais trials")


async def run_walk_forward_command(args):
    """Executa análise walk-forward"""
    from trade_system.backtester import run_walk_forward_validation
    
    print("="*60)
    print("            WALK-FORWARD ANALYSIS")
    print("="*60)
    
    print(f"Período total: {args.days} dias")
    print(f"Janela de treino: {args.train_days} dias")
    print(f"Janela de teste: {args.test_days} dias")
    print(f"Passo: {args.step_days} dias")
    print(f"Paralelo: {'SIM' if args.parallel else 'NÃO'}")
    print()
    
    results = await run_walk_forward_validation(
        days=args.days,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days
    )
    
    if results:
        print("\n✅ Walk-forward concluído!")
        print("Verifique os relatórios em 'backtest_results/'")
    else:
        print("\n❌ Falha no walk-forward")


async def run_backtest_command(args):
    from trade_system.backtester import run_backtest_validation
    
    print("="*60)
    print("                    MODO BACKTEST")
    print("="*60)
    
    config = get_config(debug_mode=args.debug)
    
    if args.config:
        # Carregar config customizada
        with open(args.config, 'r') as f:
            custom_config = yaml.safe_load(f)
        # Aplicar configurações customizadas
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    if args.symbol:
        config.symbol = args.symbol
    
    print(f"Symbol: {config.symbol}")
    print(f"Período: {args.days} dias")
    print(f"Debug: {'SIM' if args.debug else 'NÃO'}")
    print()
    
    results = await run_backtest_validation(
        config=config,
        days=args.days,
        debug_mode=args.debug
    )
    
    if results:
        print("\n✅ Backtest concluído com sucesso!")
    else:
        print("\n❌ Falha no backtest")
        sys.exit(1)


async def run_paper_trading_command(args):
    from trade_system.main import run_paper_trading
    from trade_system.backtester import run_backtest_validation
    
    print("="*60)
    print("              PAPER TRADING MODE")
    print("         Execução simulada com dados reais")
    print("="*60)
    
    # Verificar credenciais
    if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_API_SECRET'):
        print("❌ ERRO: Credenciais Binance não configuradas!")
        print("\nConfigure as variáveis de ambiente no arquivo .env")
        sys.exit(1)
    
    config = get_config(debug_mode=args.debug)
    
    if args.config:
        # Carregar config customizada
        with open(args.config, 'r') as f:
            custom_config = yaml.safe_load(f)
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    print(f"Balance inicial: ${args.balance:,.2f}")
    print(f"Símbolo: {config.symbol}")
    print(f"Debug: {'SIM' if args.debug else 'NÃO'}")
    
    if not args.no_backtest:
        print("\n🔄 Executando backtest de validação...")
        
        results = await run_backtest_validation(config=config, debug_mode=args.debug)
        if not results:
            print("\n❌ Backtest falhou. Abortando...")
            sys.exit(1)
        
        if results.get('profit_factor', 0) < 1.0 and not args.debug:
            confirm = input("\n⚠️ WARNING: Profit factor < 1.0. Continuar? (s/n): ")
            if confirm.lower() != 's':
                print("Operação cancelada.")
                sys.exit(0)
    
    print("\n▶️ Iniciando Paper Trading...")
    print("Pressione Ctrl+C para parar\n")
    
    try:
        await run_paper_trading(
            config=config,
            initial_balance=args.balance,
            debug_mode=args.debug
        )
    except KeyboardInterrupt:
        print("\n\n⏹️ Paper Trading interrompido")


def run_config_command(args):
    if args.create:
        if os.path.exists('config.yaml'):
            confirm = input("config.yaml já existe. Sobrescrever? (s/n): ")
            if confirm.lower() != 's':
                print("Operação cancelada.")
                return
        
        create_example_config()
        print("✅ config.yaml criado com sucesso!")
        print("\nEdite o arquivo para personalizar")
    
    elif args.show:
        config = get_config()
        print("\n📋 Configuração atual:")
        print(f"Symbol: {config.symbol}")
        print(f"Min confidence: {config.min_confidence}")
        print(f"Max position: {config.max_position_pct*100}%")
        print(f"Debug mode: {config.debug_mode}")
        print(f"\nPara ver todas as configurações, abra config.yaml")
    
    elif args.validate:
        print("\n🔍 Validando configuração...")
        
        try:
            config = get_config()
            
            # Validações
            errors = []
            warnings = []
            
            # Verificar limites
            if config.max_position_pct > 0.1:
                warnings.append("max_position_pct > 10% (muito alto)")
            
            if config.stop_loss_pct >= config.take_profit_pct:
                errors.append("stop_loss deve ser menor que take_profit")
            
            if config.min_confidence < 0.5:
                warnings.append("min_confidence < 50% (muito baixo)")
            
            # Resultados
            if errors:
                print("\n❌ ERROS encontrados:")
                for error in errors:
                    print(f"  - {error}")
            
            if warnings:
                print("\n⚠️ AVISOS:")
                for warning in warnings:
                    print(f"  - {warning}")
            
            if not errors and not warnings:
                print("\n✅ Configuração válida!")
                
        except Exception as e:
            print(f"\n❌ Erro ao validar: {e}")


async def run_dashboard_command(args):
    """Inicia dashboard web para visualização"""
    print("="*60)
    print("              DASHBOARD WEB")
    print("="*60)
    
    print(f"\n🌐 Iniciando servidor na porta {args.port}...")
    print(f"Acesse: http://localhost:{args.port}")
    print("\nPressione Ctrl+C para parar")
    
    # TODO: Implementar dashboard web com Flask/FastAPI
    print("\n⚠️ Dashboard ainda não implementado")


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    setup_logging(log_level=args.log_level)
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    try:
        if args.command == 'optimize':
            asyncio.run(run_optimize_command(args))
        elif args.command == 'walk-forward':
            asyncio.run(run_walk_forward_command(args))
        elif args.command == 'backtest':
            asyncio.run(run_backtest_command(args))
        elif args.command == 'paper':
            asyncio.run(run_paper_trading_command(args))
        elif args.command == 'config':
            run_config_command(args)
        elif args.command == 'dashboard':
            asyncio.run(run_dashboard_command(args))
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Operação interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()