# =============================================================================
# Script de Implantação Windows - Sistema de Trading HFT
# =============================================================================
# Autor: Sistema de Trading Algorítmico
# Descrição: Configura ambiente Windows, instala dependências e prepara o sistema HFT
# Uso: Execute no PowerShell como Administrador
# =============================================================================

# Configurações
$ProjectName = "ultra-trading-system"
$PythonVersion = "3.10"
$VenvName = "venv"
$RedisVersion = "5.0.14.1"

# Cores para output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Log {
    param($Message)
    Write-ColorOutput Green "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

function Error-Exit {
    param($Message)
    Write-ColorOutput Red "[ERROR] $Message"
    exit 1
}

function Warning {
    param($Message)
    Write-ColorOutput Yellow "[WARNING] $Message"
}

function Info {
    param($Message)
    Write-ColorOutput Cyan "[INFO] $Message"
}

# =============================================================================
# 1. VERIFICAÇÃO DO SISTEMA
# =============================================================================
function Check-System {
    Log "Verificando sistema Windows..."
    
    # Verifica se está rodando como administrador
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Error-Exit "Este script precisa ser executado como Administrador!`nClique com botão direito no PowerShell e selecione 'Executar como Administrador'"
    }
    
    # Verifica versão do PowerShell
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Error-Exit "PowerShell 5.0 ou superior é necessário"
    }
    
    Log "Sistema Windows detectado - PowerShell $($PSVersionTable.PSVersion)"
}

# =============================================================================
# 2. INSTALAÇÃO DO CHOCOLATEY E DEPENDÊNCIAS
# =============================================================================
function Install-SystemDeps {
    Log "Instalando gerenciador de pacotes Chocolatey..."
    
    # Instala Chocolatey se não existir
    if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    }
    
    Log "Instalando dependências do sistema..."
    
    # Instala Python
    choco install python --version=$PythonVersion -y
    
    # Instala Git
    choco install git -y
    
    # Instala Redis
    choco install redis-64 -y
    
    # Instala Visual C++ Build Tools (necessário para compilar algumas libs Python)
    choco install visualstudio2022buildtools -y
    choco install visualstudio2022-workload-vctools -y
    
    # Atualiza PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    # Verifica instalações
    Log "Verificando instalações..."
    python --version
    git --version
}

# =============================================================================
# 3. CRIAÇÃO DA ESTRUTURA DO PROJETO
# =============================================================================
function Create-ProjectStructure {
    Log "Criando estrutura do projeto..."
    
    # Remove diretório se existir
    if (Test-Path $ProjectName) {
        $response = Read-Host "Diretório $ProjectName já existe. Deseja sobrescrever? (S/N)"
        if ($response -ne 'S' -and $response -ne 's') {
            Error-Exit "Instalação cancelada"
        }
        Remove-Item -Recurse -Force $ProjectName
    }
    
    # Cria diretórios
    $directories = @(
        "$ProjectName",
        "$ProjectName\trade_system",
        "$ProjectName\trade_system\modes",
        "$ProjectName\trade_system\analysis",
        "$ProjectName\trade_system\analysis\ml",
        "$ProjectName\trade_system\config",
        "$ProjectName\trade_system\utils",
        "$ProjectName\tests",
        "$ProjectName\tests\unit",
        "$ProjectName\tests\integration",
        "$ProjectName\logs",
        "$ProjectName\data",
        "$ProjectName\data\historical",
        "$ProjectName\data\checkpoints",
        "$ProjectName\notebooks",
        "$ProjectName\scripts",
        "$ProjectName\.github\workflows"
    )
    
    foreach ($dir in $directories) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        
        # Cria __init__.py onde necessário
        if ($dir -like "*trade_system*" -or $dir -like "*tests*") {
            New-Item -ItemType File -Force -Path "$dir\__init__.py" | Out-Null
        }
    }
    
    Set-Location $ProjectName
}

# =============================================================================
# 4. CRIAÇÃO DOS ARQUIVOS PRINCIPAIS
# =============================================================================
function Create-Files {
    Log "Criando arquivos do sistema..."
    
    # setup.py
    @'
from setuptools import setup, find_packages

setup(
    name="ultra-trading-system",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0",
        "python-binance>=1.0.17",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "numba>=0.56.0",
        "redis>=4.3.0",
        "river>=0.15.0",
        "scikit-learn>=1.1.0",
        "ta>=0.10.1",
        "websocket-client>=1.4.0",
        "aiohttp>=3.8.0",
        "asyncio>=3.4.3",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "prometheus-client>=0.15.0",
        "python-telegram-bot>=20.0",
        "pytest>=7.2.0",
        "pytest-asyncio>=0.20.0",
        "black>=22.10.0",
        "flake8>=5.0.0",
        "mypy>=0.990",
    ],
    entry_points={
        "console_scripts": [
            "trade-system=trade_system.cli:main",
        ],
    },
)
'@ | Out-File -FilePath "setup.py" -Encoding UTF8

    # requirements-dev.txt
    @'
jupyter>=1.0.0
notebook>=6.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
ipython>=8.7.0
autopep8>=2.0.0
pre-commit>=2.20.0
coverage>=7.0.0
'@ | Out-File -FilePath "requirements-dev.txt" -Encoding UTF8

    # .gitignore
    @'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
logs/
*.log
data/historical/*.csv
data/checkpoints/*.pkl
config/local.yaml
.env
.env.local

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# OS
.DS_Store
Thumbs.db
'@ | Out-File -FilePath ".gitignore" -Encoding UTF8

    # .env.example
    @'
# Binance API
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Telegram Alerts (opcional)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Email Alerts (opcional)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_TO=alerts@yourdomain.com

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Environment
ENV=development
LOG_LEVEL=INFO
'@ | Out-File -FilePath ".env.example" -Encoding UTF8

    # run.bat - Script de execução para Windows
    @'
@echo off
echo ===================================
echo Ultra Trading System - Windows
echo ===================================
echo.

if not exist venv (
    echo Criando ambiente virtual...
    python -m venv venv
)

echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo.
echo Comandos disponíveis:
echo   1. Instalar dependências
echo   2. Executar HFT (dry-run)
echo   3. Executar Standard
echo   4. Executar testes
echo   5. Sair
echo.

:menu
set /p choice="Escolha uma opção (1-5): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto hft
if "%choice%"=="3" goto standard
if "%choice%"=="4" goto test
if "%choice%"=="5" goto end

echo Opção inválida!
goto menu

:install
echo Instalando dependências...
pip install -e .
pip install -r requirements-dev.txt
echo Instalação concluída!
pause
goto menu

:hft
echo Iniciando modo HFT (dry-run)...
python -m trade_system.cli hft --dry-run --interval 5
pause
goto menu

:standard
echo Iniciando modo Standard...
python -m trade_system.cli paper --mode standard
pause
goto menu

:test
echo Executando testes...
pytest tests/ -v
pause
goto menu

:end
echo Saindo...
deactivate
'@ | Out-File -FilePath "run.bat" -Encoding ASCII

    # start-redis.bat
    @'
@echo off
echo Iniciando Redis Server...
redis-server --port 6379
'@ | Out-File -FilePath "start-redis.bat" -Encoding ASCII

    # README.md
    @'
# Ultra Trading System - HFT & Standard Modes

Sistema de trading algorítmico de alta performance com suporte para operações standard e high-frequency trading (HFT).

## 🚀 Quick Start no Windows

```cmd
# Clone o repositório
git clone https://github.com/yourusername/ultra-trading-system.git
cd ultra-trading-system

# Execute o script de configuração
run.bat

# Em outro terminal, inicie o Redis
start-redis.bat
```

## 📋 Requisitos Windows

- Windows 10/11
- Python 3.10+
- Redis para Windows
- Visual C++ Build Tools
- 4GB RAM mínimo

## 🔧 Instalação Manual

1. Instale Python 3.10 de python.org
2. Instale Redis: https://github.com/microsoftarchive/redis/releases
3. Clone este repositório
4. Execute: `pip install -e .`

## 🏃 Executando

### Usando run.bat (Recomendado)
```cmd
run.bat
# Escolha a opção desejada no menu
```

### Manualmente
```cmd
# Ative o ambiente virtual
venv\Scripts\activate

# Modo HFT (dry-run)
trade-system hft --dry-run --interval 5

# Modo Standard
trade-system paper --mode standard
```

## 📊 Monitoramento

Para monitoramento com Docker no Windows:
1. Instale Docker Desktop
2. Execute: `docker-compose up -d`
3. Acesse: http://localhost:3000 (Grafana)

## ⚠️ Notas Importantes para Windows

- Redis no Windows é uma versão não-oficial mas funciona bem para desenvolvimento
- Para produção, considere usar WSL2 ou um servidor Linux
- Alguns pacotes Python podem precisar do Visual C++ Build Tools

## 📈 Performance no Windows

- Use Windows Terminal para melhor performance do console
- Desative o Windows Defender para a pasta do projeto (cuidado!)
- Configure o Redis para persistir dados em SSD

## 🐛 Troubleshooting Windows

### Erro: "redis-server não é reconhecido"
- Adicione o Redis ao PATH do Windows
- Ou use o caminho completo: `C:\ProgramData\chocolatey\lib\redis-64\redis-server.exe`

### Erro ao instalar numba
- Instale Visual C++ Build Tools
- Use: `choco install visualstudio2022buildtools`

### Performance lenta
- Desative antivírus para a pasta do projeto
- Use SSD para melhor I/O
- Considere usar WSL2 para performance Linux-like

## 📄 Licença

MIT License - veja LICENSE para detalhes.
'@ | Out-File -FilePath "README.md" -Encoding UTF8

    # Configurações YAML
    New-Item -ItemType Directory -Force -Path "config" | Out-Null
    
    # config/base.yaml
    @'
# Configuração Base - Comum a todos os modos
api:
  exchange: "binance"
  testnet: false
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/trading.log"
  max_size: "100MB"
  backup_count: 5

cache:
  backend: "redis"
  ttl: 300
  prefix: "trade:"

alerts:
  enabled: true
  channels:
    - telegram
    - email
  
monitoring:
  prometheus_port: 9090
  health_check_interval: 60
'@ | Out-File -FilePath "config\base.yaml" -Encoding UTF8

    # config/hft.yaml
    @'
# Configuração para modo HFT
trading:
  mode: "hft"
  symbol: "BTCBRL"
  interval_seconds: 5
  max_position_btc: 0.00003
  max_position_brl: 20.0
  min_order_value: 10.0
  fee_pct: 0.0001
  buy_threshold: 0.65
  sell_threshold: 0.65
  dry_run: true

risk:
  max_concurrent_trades: 2
  cooldown_multiplier: 2
  take_profit_pct: 0.003
  stop_loss_pct: 0.002
  max_daily_drawdown: 0.02
  emergency_stop_loss: 0.005

features:
  use_microstructure: true
  lookback_seconds: 60
  momentum_lags: [5, 10, 30]
  book_depth_levels: 5

ml:
  model_type: "river"
  n_models: 10
  max_depth: 5
  grace_period: 200
  checkpoint_interval: 3600
  drift_detection: true
  warm_start_hours: 48
'@ | Out-File -FilePath "config\hft.yaml" -Encoding UTF8
}

# =============================================================================
# 5. CRIAÇÃO DO CÓDIGO PYTHON
# =============================================================================
function Create-PythonCode {
    Log "Criando código Python principal..."
    
    # trade_system/__init__.py
    @'
"""Ultra Trading System - High Performance Algorithmic Trading"""

__version__ = "2.0.0"
__author__ = "Trading System Team"

from trade_system.config.loader import ConfigLoader

__all__ = ["ConfigLoader"]
'@ | Out-File -FilePath "trade_system\__init__.py" -Encoding UTF8

    # trade_system/cli.py
    @'
import click
import asyncio
import signal
import logging
from pathlib import Path
import sys

# Fix para Windows event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from trade_system.config.loader import ConfigLoader

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
def cli(debug):
    """Ultra Trading System CLI"""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@cli.command()
@click.option('--interval', default=5, help='Seconds between decisions')
@click.option('--max-position', default=0.00003, help='Max BTC per trade')
@click.option('--dry-run', is_flag=True, help='Paper trading mode')
@click.option('--config', default='hft.yaml', help='Config file')
def hft(interval, max_position, dry_run, config):
    """Start High-Frequency Trading mode"""
    click.echo(f"Starting HFT mode (interval={interval}s, dry_run={dry_run})")
    click.echo("HFT mode implementation here...")
    # TODO: Implementar HFTTrader

@cli.command()
@click.option('--mode', type=click.Choice(['standard', 'hft']), default='standard')
@click.option('--balance', default=1000.0, help='Initial balance')
def paper(mode, balance):
    """Start paper trading"""
    click.echo(f"Starting paper trading in {mode} mode with balance {balance}")
    # TODO: Implementar paper trading

@cli.command()
def test_connection():
    """Test API connection"""
    click.echo("Testing Binance API connection...")
    try:
        from binance.client import Client
        import os
        
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            click.echo("❌ API keys not found in environment")
            return
            
        client = Client(api_key, api_secret)
        account = client.get_account()
        click.echo("✅ Connection successful!")
        click.echo(f"Account type: {account['accountType']}")
    except Exception as e:
        click.echo(f"❌ Connection failed: {e}")

def main():
    """Entry point"""
    cli()

if __name__ == '__main__':
    main()
'@ | Out-File -FilePath "trade_system\cli.py" -Encoding UTF8

    # Config Loader
    New-Item -ItemType Directory -Force -Path "trade_system\config" | Out-Null
    
    @'
from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path

class ConfigLoader:
    @staticmethod
    def load(config_file: str, mode: str = 'standard') -> DictConfig:
        """Load and merge configurations"""
        config_path = Path('config')
        
        # Load base config
        base_cfg = OmegaConf.load(config_path / 'base.yaml')
        
        # Load mode-specific config
        mode_cfg = OmegaConf.load(config_path / config_file)
        
        # Merge configs
        cfg = OmegaConf.merge(base_cfg, mode_cfg)
        
        # Add environment variables
        env_cfg = OmegaConf.create({
            'api': {
                'key': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_API_SECRET')
            }
        })
        
        return OmegaConf.merge(cfg, env_cfg)
'@ | Out-File -FilePath "trade_system\config\loader.py" -Encoding UTF8

    # Test básico
    New-Item -ItemType Directory -Force -Path "tests\unit" | Out-Null
    
    @'
import pytest

def test_import():
    """Test basic imports"""
    import trade_system
    assert trade_system.__version__ == "2.0.0"

def test_config_loader():
    """Test config loader"""
    from trade_system.config.loader import ConfigLoader
    # Basic test
    assert ConfigLoader is not None
'@ | Out-File -FilePath "tests\unit\test_basic.py" -Encoding UTF8
}

# =============================================================================
# 6. CONFIGURAÇÃO DO AMBIENTE VIRTUAL
# =============================================================================
function Setup-Venv {
    Log "Configurando ambiente virtual Python..."
    
    # Cria venv
    python -m venv $VenvName
    
    # Ativa venv
    & ".\$VenvName\Scripts\Activate.ps1"
    
    # Atualiza pip
    python -m pip install --upgrade pip setuptools wheel
    
    # Instala dependências básicas primeiro
    pip install numpy
    pip install numba
    
    # Instala o projeto
    pip install -e .
    
    # Instala dependências de desenvolvimento
    if (Test-Path "requirements-dev.txt") {
        pip install -r requirements-dev.txt
    }
}

# =============================================================================
# 7. CONFIGURAÇÃO DO REDIS
# =============================================================================
function Setup-Redis {
    Log "Configurando Redis..."
    
    # Cria arquivo de configuração Redis
    @'
# Redis configuration for Windows
port 6379
bind 127.0.0.1
protected-mode yes
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir ./data
'@ | Out-File -FilePath "redis.conf" -Encoding UTF8
    
    Info "Redis instalado. Use 'start-redis.bat' para iniciar o servidor"
}

# =============================================================================
# 8. CONFIGURAÇÃO DO GIT
# =============================================================================
function Setup-Git {
    Log "Inicializando repositório Git..."
    
    git init
    
    # Primeiro commit
    git add .
    git commit -m "Initial commit: Ultra Trading System v2.0 - Windows"
}

# =============================================================================
# 9. TESTES E VALIDAÇÃO
# =============================================================================
function Run-Tests {
    Log "Executando testes..."
    
    # Testa imports
    python -c "import trade_system; print(f'✓ Trade System v{trade_system.__version__}')"
    
    # Testa conexão Redis (se estiver rodando)
    try {
        $redis = New-Object System.Net.Sockets.TcpClient
        $redis.Connect("localhost", 6379)
        $redis.Close()
        Log "✓ Redis acessível na porta 6379"
    } catch {
        Warning "Redis não está rodando. Execute 'start-redis.bat' em outro terminal"
    }
}

# =============================================================================
# 10. INSTALAÇÃO DE ATALHOS
# =============================================================================
function Create-Shortcuts {
    Log "Criando atalhos..."
    
    # Cria atalho para abrir projeto no VS Code
    if (Get-Command code -ErrorAction SilentlyContinue) {
        @'
@echo off
code .
'@ | Out-File -FilePath "open-vscode.bat" -Encoding ASCII
    }
    
    # Cria atalho para Jupyter
    @'
@echo off
call venv\Scripts\activate.bat
jupyter notebook
'@ | Out-File -FilePath "start-jupyter.bat" -Encoding ASCII
}

# =============================================================================
# 11. FINALIZAÇÃO
# =============================================================================
function Show-Complete {
    # Banner de conclusão
    Write-Host ""
    Write-ColorOutput Green "═══════════════════════════════════════════════════════════════"
    Write-ColorOutput Green "✅ INSTALAÇÃO CONCLUÍDA COM SUCESSO!"
    Write-ColorOutput Green "═══════════════════════════════════════════════════════════════"
    Write-Host ""
    Write-Host "📁 Projeto criado em: $(Get-Location)"
    Write-Host ""
    Write-ColorOutput Yellow "🚀 Próximos passos:"
    Write-Host ""
    Write-Host "1. Configure suas credenciais:"
    Write-ColorOutput Cyan "   copy .env.example .env"
    Write-ColorOutput Cyan "   notepad .env"
    Write-Host ""
    Write-Host "2. Inicie o Redis (em outro terminal):"
    Write-ColorOutput Cyan "   start-redis.bat"
    Write-Host ""
    Write-Host "3. Execute o sistema:"
    Write-ColorOutput Cyan "   run.bat"
    Write-Host ""
    Write-Host "📊 Comandos úteis:"
    Write-ColorOutput Cyan "   run.bat           - Menu interativo"
    Write-ColorOutput Cyan "   start-redis.bat   - Iniciar Redis"
    Write-ColorOutput Cyan "   start-jupyter.bat - Iniciar Jupyter"
    Write-Host ""
    Write-ColorOutput Red "⚠️  IMPORTANTE: Sempre teste em modo dry-run antes de usar dinheiro real!"
    Write-Host ""
    Write-ColorOutput Green "💡 Documentação completa em README.md"
    Write-Host ""
    Write-ColorOutput Green "Boa sorte com seus trades! 🚀"
    Write-Host ""
    Write-Host "Pressione qualquer tecla para continuar..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
function Main {
    Clear-Host
    Write-ColorOutput Green "═══════════════════════════════════════════════════════════════"
    Write-ColorOutput Green "       ULTRA TRADING SYSTEM - INSTALAÇÃO WINDOWS"
    Write-ColorOutput Green "═══════════════════════════════════════════════════════════════"
    Write-Host ""
    
    try {
        Check-System
        Install-SystemDeps
        Create-ProjectStructure
        Create-Files
        Create-PythonCode
        Setup-Venv
        Setup-Redis
        Create-Shortcuts
        Setup-Git
        Run-Tests
        Show-Complete
    }
    catch {
        Error-Exit "Erro durante instalação: $_"
    }
}

# Executa
Main
