import click
import os
import sys
import asyncio
from pathlib import Path

# Fix para Windows event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@click.group()
def cli():
    """Ultra Trading System CLI"""
    pass

@cli.command()
@click.option('--interval', default=5, help='Seconds between decisions')
@click.option('--dry-run', is_flag=True, help='Paper trading mode')
def hft(interval, dry_run):
    """Start HFT mode"""
    click.echo(f"Starting HFT mode (interval={interval}s, dry_run={dry_run})")
    click.echo("HFT implementation coming soon...")
    click.echo("")
    click.echo("Modo HFT será implementado com:")
    click.echo("- Análise de microestrutura de mercado")
    click.echo("- Decisões a cada 5 segundos")
    click.echo("- Machine Learning online com River")
    click.echo("- Operações de até R$20 ou 0.00003 BTC")

@cli.command()
def test_connection():
    """Test API connection"""
    click.echo("Testing Binance API connection...")
    try:
        # Carrega variáveis de ambiente do arquivo .env se existir
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        from binance.client import Client
        
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or api_key == 'your_api_key_here':
            click.echo("")
            click.echo("❌ API keys not found or not configured!")
            click.echo("")
            click.echo("Por favor:")
            click.echo("1. Copie .env.example para .env")
            click.echo("2. Edite .env e adicione suas chaves da Binance")
            click.echo("3. Tente novamente")
            return
            
        client = Client(api_key, api_secret)
        
        # Testa conexão
        click.echo("Conectando...")
        account = client.get_account()
        
        click.echo("")
        click.echo("✅ Connection successful!")
        click.echo(f"Account type: {account['accountType']}")
        click.echo(f"Can trade: {account['canTrade']}")
        
        # Mostra saldos não-zero
        click.echo("")
        click.echo("Balances:")
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                click.echo(f"  {balance['asset']}: free={free}, locked={locked}")
                
    except ImportError:
        click.echo("❌ python-binance not installed!")
        click.echo("Run: pip install python-binance")
    except Exception as e:
        click.echo(f"❌ Connection failed: {str(e)}")
        if "APIError" in str(e):
            click.echo("")
            click.echo("Possíveis causas:")
            click.echo("- Chaves API inválidas")
            click.echo("- API não habilitada na Binance")
            click.echo("- IP não autorizado")

@cli.command()
@click.option('--mode', type=click.Choice(['standard', 'hft']), default='standard')
@click.option('--balance', default=1000.0, help='Initial balance')
def paper(mode, balance):
    """Start paper trading mode"""
    click.echo(f"Starting paper trading in {mode} mode")
    click.echo(f"Initial balance: R$ {balance}")
    click.echo("")
    click.echo("Paper trading será implementado com:")
    click.echo("- Simulação completa de ordens")
    click.echo("- Tracking de P&L")
    click.echo("- Métricas de performance")
    click.echo("- Dados reais de mercado")

@cli.command()
@click.option('--symbol', default='BTCBRL')
@click.option('--days', default=7)
def backtest(symbol, days):
    """Run backtesting"""
    click.echo(f"Running backtest for {symbol} over {days} days")
    click.echo("")
    click.echo("Backtest será implementado com:")
    click.echo("- Download de dados históricos")
    click.echo("- Simulação de estratégias")
    click.echo("- Relatório detalhado")
    click.echo("- Gráficos de performance")

@cli.command()
def config():
    """Show configuration"""
    click.echo("Current configuration:")
    click.echo("")
    
    # Verifica arquivos
    files = {
        '.env': 'Environment variables',
        'config/base.yaml': 'Base configuration',
        'config/hft.yaml': 'HFT configuration',
        'config/standard.yaml': 'Standard configuration'
    }
    
    for file, desc in files.items():
        if os.path.exists(file):
            click.echo(f"✅ {desc}: {file}")
        else:
            click.echo(f"❌ {desc}: {file} (not found)")
    
    click.echo("")
    click.echo("Para criar arquivos faltantes:")
    click.echo("- Copie .env.example para .env")
    click.echo("- Execute o script de instalação novamente")

def main():
    """Entry point"""
    cli()

if __name__ == '__main__':
    main()