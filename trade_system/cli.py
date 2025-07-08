"""
Interface de linha de comando para o sistema de trading
"""
import os
import sys
import asyncio
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_system.config import get_config, create_example_config
from trade_system.logging_config import setup_logging


def create_parser():
    parser = argparse.ArgumentParser(
        description='Sistema de Trading Ultra-Otimizado v5.2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponiveis')
    
    # Backtest
    backtest_parser = subparsers.add_parser('backtest', help='Executa backtest da estrategia')
    backtest_parser.add_argument('--debug', action='store_true', help='Modo debug')
    backtest_parser.add_argument('--days', type=int, default=7, help='Dias de dados (padrao: 7)')
    backtest_parser.add_argument('--symbol', type=str, help='Par de trading')
    
    # Paper Trading
    paper_parser = subparsers.add_parser('paper', help='Inicia paper trading')
    paper_parser.add_argument('--debug', action='store_true', help='Modo debug')
    paper_parser.add_argument('--no-backtest', action='store_true', help='Pular backtest')
    paper_parser.add_argument('--balance', type=float, default=10000, help='Balance inicial')
    
    # Config
    config_parser = subparsers.add_parser('config', help='Gerenciar configuracoes')
    config_parser.add_argument('--create', action='store_true', help='Criar config.yaml')
    config_parser.add_argument('--show', action='store_true', help='Mostrar configuracao')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Nivel de logging')
    
    return parser


async def run_backtest_command(args):
    from trade_system.backtester import run_backtest_validation
    
    print("="*60)
    print("                    MODO BACKTEST")
    print("="*60)
    
    config = get_config(debug_mode=args.debug)
    if args.symbol:
        config.symbol = args.symbol
    
    print(f"Symbol: {config.symbol}")
    print(f"Periodo: {args.days} dias")
    print(f"Debug: {'SIM' if args.debug else 'NAO'}")
    print()
    
    results = await run_backtest_validation(
        config=config,
        days=args.days,
        debug_mode=args.debug
    )
    
    if results:
        print("\nBacktest concluido com sucesso!")
    else:
        print("\nFalha no backtest")
        sys.exit(1)


async def run_paper_trading_command(args):
    from trade_system.main import run_paper_trading
    from trade_system.backtester import run_backtest_validation
    
    print("="*60)
    print("              PAPER TRADING MODE")
    print("         Execucao simulada com dados reais")
    print("="*60)
    
    # Verificar credenciais
    if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_API_SECRET'):
        print("ERRO: Credenciais Binance nao configuradas!")
        print("\nConfigure as variaveis de ambiente no arquivo .env")
        sys.exit(1)
    
    config = get_config(debug_mode=args.debug)
    
    print(f"Balance inicial: ${args.balance:,.2f}")
    print(f"Simbolo: {config.symbol}")
    print(f"Debug: {'SIM' if args.debug else 'NAO'}")
    
    if not args.no_backtest:
        print("\nExecutando backtest de validacao...")
        
        results = await run_backtest_validation(config=config, debug_mode=args.debug)
        if not results:
            print("\nBacktest falhou. Abortando...")
            sys.exit(1)
        
        if results.get('profit_factor', 0) < 1.0 and not args.debug:
            confirm = input("\nWARNING: Profit factor < 1.0. Continuar? (s/n): ")
            if confirm.lower() != 's':
                print("Operacao cancelada.")
                sys.exit(0)
    
    print("\nIniciando Paper Trading...")
    print("Pressione Ctrl+C para parar\n")
    
    try:
        await run_paper_trading(
            config=config,
            initial_balance=args.balance,
            debug_mode=args.debug
        )
    except KeyboardInterrupt:
        print("\n\nPaper Trading interrompido")


def run_config_command(args):
    if args.create:
        if os.path.exists('config.yaml'):
            confirm = input("config.yaml ja existe. Sobrescrever? (s/n): ")
            if confirm.lower() != 's':
                print("Operacao cancelada.")
                return
        
        create_example_config()
        print("config.yaml criado com sucesso!")
        print("\nEdite o arquivo para personalizar")
    
    elif args.show:
        config = get_config()
        print("\nConfiguracao atual:")
        print(f"Symbol: {config.symbol}")
        print(f"Min confidence: {config.min_confidence}")
        print(f"Max position: {config.max_position_pct*100}%")
        print(f"Debug mode: {config.debug_mode}")
        print(f"\nPara ver todas as configuracoes, abra config.yaml")


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    setup_logging(log_level=args.log_level)
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == 'backtest':
        asyncio.run(run_backtest_command(args))
    elif args.command == 'paper':
        asyncio.run(run_paper_trading_command(args))
    elif args.command == 'config':
        run_config_command(args)


if __name__ == '__main__':
    main()
