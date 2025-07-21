@echo off
REM =============================================================================
REM Script de Instalação Simplificado - Sistema Trading HFT
REM Para Windows CMD
REM =============================================================================

title Ultra Trading System - Instalacao Windows

echo ===============================================================
echo        ULTRA TRADING SYSTEM - INSTALACAO WINDOWS
echo ===============================================================
echo.

REM Verifica Python
echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado!
    echo Por favor instale Python 3.10 de https://python.org
    echo Certifique-se de marcar "Add Python to PATH" durante instalacao
    pause
    exit /b 1
)

echo [OK] Python encontrado
echo.

REM Nome do projeto
set PROJECT_NAME=ultra-trading-system

REM Verifica se diretório existe
if exist %PROJECT_NAME% (
    echo Diretorio %PROJECT_NAME% ja existe!
    set /p resp="Deseja sobrescrever? (S/N): "
    if /i not "%resp%"=="S" (
        echo Instalacao cancelada.
        pause
        exit /b 1
    )
    rmdir /s /q %PROJECT_NAME%
)

echo Criando estrutura do projeto...

REM Cria diretórios
mkdir %PROJECT_NAME%
cd %PROJECT_NAME%

mkdir trade_system
mkdir trade_system\modes
mkdir trade_system\analysis
mkdir trade_system\analysis\ml
mkdir trade_system\config
mkdir trade_system\utils
mkdir tests
mkdir tests\unit
mkdir tests\integration
mkdir logs
mkdir data
mkdir data\historical
mkdir data\checkpoints
mkdir config
mkdir notebooks
mkdir scripts

REM Cria arquivos __init__.py
echo. > trade_system\__init__.py
echo. > trade_system\modes\__init__.py
echo. > trade_system\analysis\__init__.py
echo. > trade_system\analysis\ml\__init__.py
echo. > trade_system\config\__init__.py
echo. > trade_system\utils\__init__.py
echo. > tests\__init__.py
echo. > tests\unit\__init__.py

echo.
echo Criando arquivos principais...

REM Cria setup.py
(
echo from setuptools import setup, find_packages
echo.
echo setup(
echo     name="ultra-trading-system",
echo     version="2.0.0",
echo     packages=find_packages(),
echo     install_requires=[
echo         "click>=8.0",
echo         "python-binance>=1.0.17",
echo         "pandas>=1.5.0",
echo         "numpy>=1.23.0",
echo         "redis>=4.3.0",
echo         "pyyaml>=6.0",
echo         "omegaconf>=2.3.0",
echo         "pytest>=7.2.0",
echo     ],
echo     entry_points={
echo         "console_scripts": [
echo             "trade-system=trade_system.cli:main",
echo         ],
echo     },
echo ^)
) > setup.py

REM Cria requirements.txt
(
echo click>=8.0
echo python-binance>=1.0.17
echo pandas>=1.5.0
echo numpy>=1.23.0
echo numba>=0.56.0
echo redis>=4.3.0
echo river>=0.15.0
echo scikit-learn>=1.1.0
echo ta>=0.10.1
echo websocket-client>=1.4.0
echo aiohttp>=3.8.0
echo pyyaml>=6.0
echo omegaconf>=2.3.0
echo prometheus-client>=0.15.0
echo python-telegram-bot>=20.0
echo pytest>=7.2.0
echo pytest-asyncio>=0.20.0
) > requirements.txt

REM Cria .env.example
(
echo # Binance API
echo BINANCE_API_KEY=your_api_key_here
echo BINANCE_API_SECRET=your_api_secret_here
echo.
echo # Telegram Alerts ^(opcional^)
echo TELEGRAM_BOT_TOKEN=your_bot_token_here
echo TELEGRAM_CHAT_ID=your_chat_id_here
echo.
echo # Redis
echo REDIS_HOST=localhost
echo REDIS_PORT=6379
echo REDIS_DB=0
echo.
echo # Environment
echo ENV=development
echo LOG_LEVEL=INFO
) > .env.example

REM Cria .gitignore
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo venv/
echo env/
echo .venv
echo.
echo # Project
echo logs/
echo *.log
echo data/historical/*.csv
echo data/checkpoints/*.pkl
echo .env
echo .env.local
echo.
echo # IDEs
echo .vscode/
echo .idea/
echo.
echo # OS
echo Thumbs.db
) > .gitignore

REM Cria script principal run.bat
(
echo @echo off
echo cls
echo echo ===================================
echo echo Ultra Trading System
echo echo ===================================
echo echo.
echo.
echo if not exist venv (
echo     echo Criando ambiente virtual...
echo     python -m venv venv
echo ^)
echo.
echo echo Ativando ambiente virtual...
echo call venv\Scripts\activate.bat
echo.
echo :menu
echo echo.
echo echo Escolha uma opcao:
echo echo   1. Instalar dependencias
echo echo   2. Configurar credenciais
echo echo   3. Testar conexao API
echo echo   4. Iniciar HFT ^(dry-run^)
echo echo   5. Iniciar modo Standard
echo echo   6. Executar testes
echo echo   7. Sair
echo echo.
echo.
echo set /p choice="Digite sua escolha (1-7): "
echo.
echo if "%%choice%%"=="1" goto install
echo if "%%choice%%"=="2" goto config
echo if "%%choice%%"=="3" goto testapi
echo if "%%choice%%"=="4" goto hft
echo if "%%choice%%"=="5" goto standard
echo if "%%choice%%"=="6" goto test
echo if "%%choice%%"=="7" goto end
echo.
echo echo Opcao invalida!
echo goto menu
echo.
echo :install
echo cls
echo echo Instalando dependencias...
echo pip install --upgrade pip
echo pip install -e .
echo echo.
echo echo Instalacao concluida!
echo pause
echo cls
echo goto menu
echo.
echo :config
echo cls
echo if not exist .env copy .env.example .env
echo echo Abrindo arquivo de configuracao...
echo notepad .env
echo echo.
echo echo Configuracao salva!
echo pause
echo cls
echo goto menu
echo.
echo :testapi
echo cls
echo echo Testando conexao com Binance...
echo python -m trade_system.cli test-connection
echo pause
echo cls
echo goto menu
echo.
echo :hft
echo cls
echo echo Iniciando modo HFT em dry-run...
echo python -m trade_system.cli hft --dry-run --interval 5
echo pause
echo cls
echo goto menu
echo.
echo :standard
echo cls
echo echo Iniciando modo Standard...
echo python -m trade_system.cli paper --mode standard
echo pause
echo cls
echo goto menu
echo.
echo :test
echo cls
echo echo Executando testes...
echo pytest tests/ -v
echo pause
echo cls
echo goto menu
echo.
echo :end
echo deactivate
echo exit
) > run.bat

REM Cria configurações básicas
echo Criando arquivos de configuracao...

REM config/base.yaml
(
echo # Configuracao Base
echo api:
echo   exchange: "binance"
echo   testnet: false
echo.  
echo logging:
echo   level: "INFO"
echo   file: "logs/trading.log"
echo.
echo cache:
echo   backend: "redis"
echo   ttl: 300
echo.
echo alerts:
echo   enabled: true
echo   channels:
echo     - telegram
) > config\base.yaml

REM config/hft.yaml
(
echo # Configuracao HFT
echo trading:
echo   mode: "hft"
echo   symbol: "BTCBRL"
echo   interval_seconds: 5
echo   max_position_btc: 0.00003
echo   max_position_brl: 20.0
echo   fee_pct: 0.0001
echo   dry_run: true
echo.
echo risk:
echo   take_profit_pct: 0.003
echo   stop_loss_pct: 0.002
echo   max_daily_drawdown: 0.02
echo.
echo features:
echo   lookback_seconds: 60
echo   momentum_lags: [5, 10, 30]
) > config\hft.yaml

REM Cria código Python básico
echo Criando codigo Python...

REM trade_system/__init__.py
(
echo """Ultra Trading System"""
echo __version__ = "2.0.0"
) > trade_system\__init__.py

REM trade_system/cli.py
(
echo import click
echo import os
echo.
echo @click.group^(^)
echo def cli^(^):
echo     """Ultra Trading System CLI"""
echo     pass
echo.
echo @cli.command^(^)
echo @click.option^('--interval', default=5^)
echo @click.option^('--dry-run', is_flag=True^)
echo def hft^(interval, dry_run^):
echo     """Start HFT mode"""
echo     click.echo^(f"Starting HFT mode ^(interval={interval}s, dry_run={dry_run}^)"^)
echo     click.echo^("HFT implementation coming soon..."^)
echo.
echo @cli.command^(^)
echo def test_connection^(^):
echo     """Test API connection"""
echo     try:
echo         from binance.client import Client
echo         api_key = os.getenv^('BINANCE_API_KEY'^)
echo         api_secret = os.getenv^('BINANCE_API_SECRET'^)
echo         
echo         if not api_key:
echo             click.echo^("API keys not found. Run 'config' first"^)
echo             return
echo             
echo         client = Client^(api_key, api_secret^)
echo         client.get_account^(^)
echo         click.echo^("Connection successful!"^)
echo     except Exception as e:
echo         click.echo^(f"Connection failed: {e}"^)
echo.
echo @cli.command^(^)
echo @click.option^('--mode', default='standard'^)
echo def paper^(mode^):
echo     """Paper trading mode"""
echo     click.echo^(f"Starting paper trading in {mode} mode"^)
echo.
echo def main^(^):
echo     cli^(^)
echo.
echo if __name__ == '__main__':
echo     main^(^)
) > trade_system\cli.py

REM Cria README
(
echo # Ultra Trading System
echo.
echo Sistema de trading algoritmico para Windows.
echo.
echo ## Instalacao
echo.
echo 1. Execute: run.bat
echo 2. Escolha opcao 1 para instalar dependencias
echo 3. Escolha opcao 2 para configurar credenciais
echo 4. Escolha opcao 3 para testar conexao
echo 5. Escolha opcao 4 para iniciar HFT em modo dry-run
echo.
echo ## Requisitos
echo.
echo - Windows 10/11
echo - Python 3.10+
echo - Redis ^(opcional^)
echo.
echo ## Suporte
echo.
echo Veja documentacao completa no GitHub.
) > README.md

REM Instala Redis para Windows
echo.
echo ===============================================================
echo INSTALACAO DO REDIS ^(OPCIONAL^)
echo ===============================================================
echo.
echo Para usar cache Redis, voce precisa instalar:
echo.
echo 1. Baixe Redis para Windows:
echo    https://github.com/microsoftarchive/redis/releases
echo.
echo 2. Ou instale via Chocolatey ^(se tiver^):
echo    choco install redis-64
echo.
echo 3. Para iniciar Redis:
echo    redis-server
echo.
echo ===============================================================
echo.

REM Cria ambiente virtual
echo Criando ambiente virtual...
python -m venv venv

REM Finalização
cls
echo ===============================================================
echo         INSTALACAO CONCLUIDA COM SUCESSO!
echo ===============================================================
echo.
echo Projeto criado em: %CD%
echo.
echo Proximos passos:
echo   1. Execute: run.bat
echo   2. Escolha opcao 1 para instalar dependencias
echo   3. Configure suas credenciais da Binance
echo.
echo IMPORTANTE: Sempre teste em modo dry-run primeiro!
echo.
echo ===============================================================
echo.
pause
