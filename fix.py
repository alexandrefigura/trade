# fix.py - Salve este arquivo na raiz do projeto e execute
import os

cli_content = '''import click
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

@cli.command()
def test_connection():
    """Test API connection"""
    click.echo("Testing Binance API connection...")
    try:
        # Carrega .env se existir
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
            click.echo("❌ API keys not configured!")
            click.echo("Please edit .env file first")
            return
            
        client = Client(api_key, api_secret)
        account = client.get_account()
        
        click.echo("✅ Connection successful!")
        click.echo(f"Account type: {account['accountType']}")
        
    except ImportError:
        click.echo("❌ python-binance not installed!")
        click.echo("Run: pip install python-binance")
    except Exception as e:
        click.echo(f"❌ Connection failed: {str(e)}")

@cli.command()
@click.option('--mode', default='standard')
def paper(mode):
    """Start paper trading mode"""
    click.echo(f"Starting paper trading in {mode} mode")

def main():
    """Entry point"""
    cli()

if __name__ == '__main__':
    main()
'''

# Cria o arquivo
with open('trade_system/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli_content)

print("✅ Arquivo cli.py criado com sucesso!")
print("Agora você pode testar com: python -m trade_system.cli test-connection")