@echo off
cls
echo ===================================
echo Ultra Trading System
echo ===================================
echo.

if not exist venv (
    echo Criando ambiente virtual...
    python -m venv venv
)

echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

:menu
echo.
echo Escolha uma opcao:
echo   1. Instalar dependencias
echo   2. Configurar credenciais
echo   3. Testar conexao API
echo   4. Iniciar HFT (dry-run)
echo   5. Iniciar modo Standard
echo   6. Executar testes
echo   7. Sair
echo.

set /p choice="Digite sua escolha (1-7): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto config
if "%choice%"=="3" goto testapi
if "%choice%"=="4" goto hft
if "%choice%"=="5" goto standard
if "%choice%"=="6" goto test
if "%choice%"=="7" goto end

echo Opcao invalida!
goto menu

:install
cls
echo Instalando dependencias...
pip install --upgrade pip
pip install -e .
echo.
echo Instalacao concluida!
pause
cls
goto menu

:config
cls
if not exist .env copy .env.example .env
echo Abrindo arquivo de configuracao...
notepad .env
echo.
echo Configuracao salva!
pause
cls
goto menu

:testapi
cls
echo Testando conexao com Binance...
python -m trade_system.cli test-connection
pause
cls
goto menu

:hft
cls
echo Iniciando modo HFT em dry-run...
python -m trade_system.cli hft --dry-run --interval 5
pause
cls
goto menu

:standard
cls
echo Iniciando modo Standard...
python -m trade_system.cli paper --mode standard
pause
cls
goto menu

:test
cls
echo Executando testes...
pytest tests/ -v
pause
cls
goto menu

:end
deactivate
exit
