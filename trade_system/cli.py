"""Interface de linha de comando"""
import asyncio
import click
import sys
from pathlib import Path
from datetime import datetime
import yaml

from trade_system.config import TradingConfig
from trade_system.logging_config import setup_logging
from trade_system.main import run_paper_trading
from trade_system.backtester import Backtester

@click.group()
@click.version_option(version='2.0.0')
def cli():
    """Ultra Trading System - Bot de trading algorítmico de alta performance"""
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Arquivo de configuração')
@click.option('--balance', '-b', type=float, 
              help='Balance inicial para paper trading')
@click.option('--debug', is_flag=True, 
              help='Modo debug com logs detalhados')
def paper(config, balance, debug):
    """Executa o sistema em modo paper trading"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  PAPER TRADING MODE                          ║
║              Execução simulada com dados reais               ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(run_paper_trading_command(config, balance, debug))
    except KeyboardInterrupt:
        print("\n\n⏹️ Sistema interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        sys.exit(1)

@cli.command()
@click.option('--symbol', '-s', default='BTCUSDT', 
              help='Par de trading')
@click.option('--days', '-d', type=int, default=7, 
              help='Dias de dados históricos')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Arquivo de configuração')
def backtest(symbol, days, config):
    """Executa backtest com dados históricos"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      BACKTEST MODE                           ║
║                 Testando com {days} dias de dados                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(run_backtest_command(symbol, days, config))
    except Exception as e:
        print(f"\n❌ Erro no backtest: {e}")
        sys.exit(1)

@cli.command()
@click.option('--create', is_flag=True, help='Criar arquivo de configuração')
@click.option('--show', is_flag=True, help='Mostrar configuração atual')
@click.option('--validate', is_flag=True, help='Validar configuração')
def config(create, show, validate):
    """Gerencia configurações do sistema"""
    if create:
        create_config()
    elif show:
        show_config()
    elif validate:
        validate_config()
    else:
        click.echo("Use --create, --show ou --validate")

async def run_paper_trading_command(config_file, balance, debug):
    """Executa paper trading com parâmetros do CLI"""
    from trade_system.backtester import Backtester
    
    # Configurar logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level)
    
    # Carregar configuração
    if config_file:
        config = TradingConfig.from_file(config_file)
    else:
        config = TradingConfig.from_env()
    
    # Sobrescrever balance se fornecido
    if balance:
        config.base_balance = balance
    
    # Executar backtest de validação
    print("\n🔬 Executando backtest de validação (7 dias)...")
    backtester = Backtester(config)
    await backtester.run(days=7)
    
    metrics = backtester.get_metrics()
    print(f"""
📊 Resultados do Backtest:
   Trades: {metrics['total_trades']}
   Win Rate: {metrics['win_rate']:.2%}
   Profit Factor: {metrics['profit_factor']:.2f}
   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
   Max Drawdown: {metrics['max_drawdown']:.2%}
   Retorno: {metrics['total_return']:.2%}
    """)
    
    # Perguntar se deseja continuar
    if metrics['total_trades'] == 0:
        print("\n⚠️ Nenhum trade foi executado no backtest!")
        print("Isso pode indicar que os parâmetros estão muito restritivos.")
    
    response = input("\nDeseja continuar com paper trading? (s/n): ")
    if response.lower() != 's':
        return
    
    # Executar paper trading
    await run_paper_trading(config_file, debug, balance)

async def run_backtest_command(symbol, days, config_file):
    """Executa backtest"""
    # Configurar logging
    setup_logging("INFO")
    
    # Carregar configuração
    if config_file:
        config = TradingConfig.from_file(config_file)
    else:
        config = TradingConfig()
    
    config.symbol = symbol
    
    # Executar backtest
    backtester = Backtester(config)
    await backtester.run(days=days)
    
    # Mostrar resultados
    backtester.print_results()

def create_config():
    """Cria arquivo de configuração padrão"""
    config = TradingConfig()
    config.save("config.yaml")
    
    print("✅ Arquivo config.yaml criado!")
    print("\nPróximos passos:")
    print("1. Edite config.yaml com suas configurações")
    print("2. Configure as variáveis de ambiente:")
    print("   export BINANCE_API_KEY='sua_chave'")
    print("   export BINANCE_API_SECRET='seu_secret'")
    print("3. Execute: trade-system paper")

def show_config():
    """Mostra configuração atual"""
    if Path("config.yaml").exists():
        config = TradingConfig.from_file("config.yaml")
    else:
        config = TradingConfig.from_env()
    
    # Ocultar secrets
    config_dict = config.to_dict()
    if config_dict['api_key']:
        config_dict['api_key'] = config_dict['api_key'][:8] + "..."
    if config_dict['api_secret']:
        config_dict['api_secret'] = config_dict['api_secret'][:8] + "..."
    
    print("Configuração atual:")
    print(yaml.dump(config_dict, default_flow_style=False))

def validate_config():
    """Valida configuração"""
    if Path("config.yaml").exists():
        config = TradingConfig.from_file("config.yaml")
    else:
        config = TradingConfig.from_env()
    
    issues = []
    
    # Verificar API keys
    if not config.api_key or not config.api_secret:
        issues.append("❌ API Keys não configuradas")
    
    # Verificar parâmetros
    if config.min_confidence > 0.9:
        issues.append("⚠️ min_confidence muito alto (> 0.9)")
    
    if config.max_position_pct > 0.1:
        issues.append("⚠️ max_position_pct muito alto (> 10%)")
    
    if issues:
        print("Problemas encontrados:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Configuração válida!")

def main():
    """Função principal do CLI"""
    cli()

if __name__ == "__main__":
    main()
