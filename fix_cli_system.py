#!/usr/bin/env python3
"""
Script para diagnosticar e corrigir o sistema CLI original
"""
import os
import subprocess
import sys

def check_cli_setup():
    """Verifica se o CLI está instalado corretamente"""
    print("🔍 VERIFICANDO SISTEMA CLI ORIGINAL")
    print("=" * 60)
    
    # 1. Verificar se setup.py existe
    if os.path.exists('setup.py'):
        print("✅ setup.py encontrado")
        
        # Tentar instalar em modo desenvolvimento
        print("\n📦 Instalando sistema em modo desenvolvimento...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
            print("✅ Sistema instalado com sucesso!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro na instalação: {e}")
            return False
    else:
        print("❌ setup.py não encontrado")
        create_setup_py()
    
    # 2. Verificar se o comando trade-system está disponível
    print("\n🔧 Verificando comando trade-system...")
    try:
        result = subprocess.run(["trade-system", "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Comando trade-system funcionando!")
            print("\nComandos disponíveis:")
            print(result.stdout)
        else:
            print("❌ Comando trade-system não encontrado")
    except FileNotFoundError:
        print("❌ Comando trade-system não está no PATH")
        
    # 3. Verificar estrutura de diretórios
    print("\n📁 Verificando estrutura...")
    required_files = [
        'trade_system/__init__.py',
        'trade_system/cli.py',
        'trade_system/main.py',
        'trade_system/config.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} não encontrado")
    
    return True

def create_setup_py():
    """Cria setup.py se não existir"""
    print("\n📝 Criando setup.py...")
    
    setup_content = '''from setuptools import setup, find_packages

setup(
    name="ultra-trading-system",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "python-binance>=1.0.0",
        "asyncio-throttle>=1.0.0",
        "TA-Lib>=0.4.24",
        "numba>=0.56.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "aiohttp>=3.8.0",
        "websockets>=10.0",
        "redis>=4.0.0",
        "requests>=2.26.0",
        "click>=8.0.0",
        "colorlog>=6.6.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "psutil>=5.8.0",
    ],
    entry_points={
        'console_scripts': [
            'trade-system=trade_system.cli:main',
        ],
    },
)
'''
    
    with open('setup.py', 'w') as f:
        f.write(setup_content)
    print("✅ setup.py criado!")

def fix_cli_main():
    """Corrige o arquivo cli.py para funcionar corretamente"""
    cli_path = 'trade_system/cli.py'
    
    if not os.path.exists(cli_path):
        print(f"\n📝 Criando {cli_path}...")
        
        cli_content = '''#!/usr/bin/env python3
"""
CLI para o Sistema de Trading
"""
import click
import asyncio
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

@click.group()
def cli():
    """Ultra Trading System - CLI"""
    pass

@cli.command()
@click.option('--balance', default=10000, help='Balance inicial para paper trading')
@click.option('--debug', is_flag=True, help='Modo debug com mais informações')
def paper(balance, debug):
    """Inicia paper trading"""
    click.echo(f"🚀 Iniciando Paper Trading com balance ${balance:,.2f}")
    
    # Verificar API keys
    api_key = os.getenv('BINANCE_API_KEY')
    if not api_key:
        click.echo("❌ BINANCE_API_KEY não encontrada no .env")
        return
    
    # Importar e executar o sistema principal
    try:
        from trade_system.main import TradingSystem
        from trade_system.config import TradingConfig
        
        # Criar config
        config = TradingConfig()
        config.INITIAL_BALANCE = balance
        
        # Criar e executar sistema
        system = TradingSystem(config, paper_trading=True)
        
        # Executar
        asyncio.run(system.run())
        
    except KeyboardInterrupt:
        click.echo("\\n⏹️ Sistema interrompido")
    except Exception as e:
        click.echo(f"❌ Erro: {e}")
        if debug:
            import traceback
            traceback.print_exc()

@cli.command()
@click.option('--days', default=7, help='Dias de histórico para backtest')
@click.option('--symbol', default='BTCUSDT', help='Par para backtest')
def backtest(days, symbol):
    """Executa backtest"""
    click.echo(f"📊 Executando backtest de {days} dias para {symbol}")
    
    try:
        from trade_system.backtester import Backtester
        from trade_system.config import TradingConfig
        
        config = TradingConfig()
        config.SYMBOL = symbol
        
        backtester = Backtester(config)
        asyncio.run(backtester.run(days=days))
        
    except Exception as e:
        click.echo(f"❌ Erro no backtest: {e}")

@cli.command()
@click.option('--create', is_flag=True, help='Criar arquivo de configuração')
def config(create):
    """Gerencia configurações"""
    if create:
        config_content = """# Configuração do Trading System
symbol: BTCUSDT
initial_balance: 10000

# Risk Management
risk:
  stop_loss_percent: 2.0
  take_profit_percent: 3.0
  max_position_percent: 95.0

# Technical Analysis
technical:
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70
  ma_short: 5
  ma_long: 20
"""
        
        with open('config.yaml', 'w') as f:
            f.write(config_content)
        
        click.echo("✅ config.yaml criado!")
    else:
        click.echo("Use --create para criar arquivo de configuração")

def main():
    """Entry point"""
    cli()

if __name__ == '__main__':
    main()
'''
        
        os.makedirs('trade_system', exist_ok=True)
        with open(cli_path, 'w') as f:
            f.write(cli_content)
        print("✅ cli.py criado!")

def main():
    """Função principal"""
    print("🔧 CORRIGINDO SISTEMA CLI ORIGINAL")
    print("=" * 60)
    
    # 1. Verificar e corrigir setup
    check_cli_setup()
    
    # 2. Corrigir CLI se necessário
    fix_cli_main()
    
    print("\n" + "=" * 60)
    print("✅ CORREÇÕES APLICADAS!")
    print("\n🚀 Para usar o sistema:")
    print("   1. pip install -e .")
    print("   2. trade-system paper")
    print("   3. trade-system backtest")
    print("   4. trade-system config --create")
    
    print("\n💡 Se o comando não funcionar, tente:")
    print("   python -m trade_system.cli paper")

if __name__ == "__main__":
    main()
