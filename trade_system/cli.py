"""
Interface de linha de comando para o sistema de trading
"""
import os
import sys
import asyncio
import argparse

# adiciona o diretório raiz do projeto ao PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_system.config import get_config, create_example_config
from trade_system.logging_config import setup_logging
from trade_system.main import run_paper_trading, run_live_trading  # ou apenas run_paper_trading/backtest

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Sistema de Trading Ultra-Otimizado v5.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  trade-system backtest
  trade-system paper
  trade-system config --create
        """
    )
    sub = parser.add_subparsers(dest='command', help='subcomandos')

    # backtest
    bt = sub.add_parser('backtest', help='Executa backtest de validação')
    bt.add_argument('--debug', action='store_true', help='modo debug')
    bt.add_argument('--days', type=int, default=7, help='dias de histórico')

    # paper
    pt = sub.add_parser('paper', help='Inicia paper trading')
    pt.add_argument('--debug', action='store_true', help='modo debug')
    pt.add_argument('--no-backtest', action='store_true', help='pula validação')
    pt.add_argument('--balance', type=float, default=10000, help='capital inicial')

    # config
    cfg = sub.add_parser('config', help='Gerencia config.yaml/.env')
    cfg.add_argument('--create', action='store_true', help='cria config.yaml exemplo')
    cfg.add_argument('--show', action='store_true', help='mostra valores atuais')
    cfg.add_argument('--file', type=str, default='config.yaml', help='caminho ao YAML')

    # globais
    parser.add_argument('--log-level', choices=['DEBUG','INFO','WARNING','ERROR'], default='INFO')

    return parser


async def _cmd_backtest(args):
    from trade_system.backtester import run_backtest_validation
    cfg = get_config(debug_mode=args.debug)
    print("=== MODO BACKTEST ===")
    res = await run_backtest_validation(config=cfg, days=args.days, debug_mode=args.debug)
    if res:
        print(f"✅ Backtest: {res['num_trades']} trades | ROI {res.get('total_return',0):.2%}")


async def _cmd_paper(args):
    cfg = get_config(debug_mode=args.debug)
    print("=== PAPER TRADING ===")
    if not args.no_backtest:
        from trade_system.backtester import run_backtest_validation
        await run_backtest_validation(config=cfg, days=7, debug_mode=args.debug)

    await run_paper_trading(
        config=cfg,
        initial_balance=args.balance,
        debug_mode=args.debug
    )


def _cmd_config(args):
    if args.create:
        if os.path.exists(args.file):
            c = input(f"{args.file} já existe. Sobrescrever? (s/n): ")
            if c.lower()!='s':
                print("Cancelado.")
                return
        create_example_config(args.file)
        print(f"✅ {args.file} criado.")
        return
    if args.show:
        cfg = get_config(debug_mode=False)
        print("CONFIG ATUAL:")
        for k,v in vars(cfg).items():
            print(f"  {k}: {v}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    setup_logging(log_level=args.log_level)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == 'backtest':
        asyncio.run(_cmd_backtest(args))
    elif args.command == 'paper':
        asyncio.run(_cmd_paper(args))
    elif args.command == 'config':
        _cmd_config(args)


if __name__ == '__main__':
    main()
