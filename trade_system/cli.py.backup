"""
Interface de linha de comando para o sistema de trading
"""
import os
import sys
import asyncio
import argparse
from datetime import datetime
from typing import Optional

# Adicionar diretório pai ao path se necessário
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_system.config import get_config, create_example_config
from trade_system.logging_config import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Cria parser de argumentos"""
    parser = argparse.ArgumentParser(
        description='Sistema de Trading Ultra-Otimizado v5.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python -m trade_system.cli backtest              # Executa backtest
  python -m trade_system.cli backtest --debug      # Backtest em modo debug
  python -m trade_system.cli paper                 # Inicia paper trading
  python -m trade_system.cli paper --no-backtest   # Paper trading sem backtest
  python -m trade_system.cli config                # Cria config.yaml exemplo
        """
    )
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
    
    # Backtest
    backtest_parser = subparsers.add_parser('backtest', help='Executa backtest da estratégia')
    backtest_parser.add_argument('--debug', action='store_true', help='Modo debug com parâmetros agressivos')
    backtest_parser.add_argument('--days', type=int, default=7, help='Dias de dados históricos (padrão: 7)')
    backtest_parser.add_argument('--symbol', type=str, help='Par de trading (ex: BTCUSDT)')
    
    # Paper Trading
    paper_parser = subparsers.add_parser('paper', help='Inicia paper trading com dados reais')
    paper_parser.add_argument('--debug', action='store_true', help='Modo debug')
    paper_parser.add_argument('--no-backtest', action='store_true', help='Pular backtest inicial')
    paper_parser.add_argument('--balance', type=float, default=10000, help='Balance inicial (padrão: 10000)')
    
    # Config
    config_parser = subparsers.add_parser('config', help='Gerenciar configurações')
    config_parser.add_argument('--create', action='store_true', help='Criar config.yaml exemplo')
    config_parser.add_argument('--show', action='store_true', help='Mostrar configuração atual')
    
    # Opções globais
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Nível de logging')
    parser.add_argument('--config-file', type=str, default='config.yaml', 
                       help='Arquivo de configuração')
    
    return parser


async def run_backtest_command(args):
    """Executa comando de backtest"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    MODO BACKTEST                             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # TODO: Implementar backtest
    print("❌ Backtest ainda não implementado nesta versão modular")
    print("Em desenvolvimento...")


async def run_paper_trading_command(args):
    """Executa comando de paper trading"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  PAPER TRADING MODE                          ║
║              Execução simulada com dados reais               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # TODO: Implementar paper trading
    print("❌ Paper Trading ainda não implementado nesta versão modular")
    print("Em desenvolvimento...")


def run_config_command(args):
    """Executa comando de configuração"""
    if args.create:
        if os.path.exists('config.yaml'):
            confirm = input("config.yaml já existe. Sobrescrever? (s/n): ")
            if confirm.lower() != 's':
                print("Operação cancelada.")
                return
        
        create_example_config()
        print("✅ config.yaml criado com sucesso!")
        print("\n📝 Edite o arquivo para personalizar os parâmetros")
    
    elif args.show:
        config = get_config()
        print("\n📋 Configuração atual:")
        print(f"Symbol: {config.symbol}")
        print(f"Min confidence: {config.min_confidence}")
        print(f"Max position: {config.max_position_pct*100}%")
        print(f"Debug mode: {config.debug_mode}")
        print(f"\nPara ver todas as configurações, abra config.yaml")


def main():
    """Função principal do CLI"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging(log_level=args.log_level)
    
    # Verificar comando
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Executar comando apropriado
    if args.command == 'backtest':
        asyncio.run(run_backtest_command(args))
    
    elif args.command == 'paper':
        asyncio.run(run_paper_trading_command(args))
    
    elif args.command == 'config':
        run_config_command(args)


if __name__ == '__main__':
    main()
