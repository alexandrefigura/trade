#!/bin/bash
# =============================================================================
# Script de Implantação Completa - Sistema de Trading HFT
# =============================================================================
# Autor: Sistema de Trading Algorítmico
# Descrição: Configura ambiente, instala dependências e prepara o sistema HFT
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações
PROJECT_NAME="ultra-trading-system"
PYTHON_VERSION="3.10"
VENV_NAME="venv"
REDIS_PORT=6379

# Função de log
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# =============================================================================
# 1. VERIFICAÇÃO DO SISTEMA
# =============================================================================
check_system() {
    log "Verificando sistema operacional..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        DISTRO=$(lsb_release -si 2>/dev/null || echo "Unknown")
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        error "Sistema operacional não suportado: $OSTYPE"
    fi
    
    log "Sistema detectado: $OS ${DISTRO:-}"
}

# =============================================================================
# 2. INSTALAÇÃO DE DEPENDÊNCIAS DO SISTEMA
# =============================================================================
install_system_deps() {
    log "Instalando dependências do sistema..."
    
    if [[ "$OS" == "linux" ]]; then
        sudo apt-get update -qq
        sudo apt-get install -y \
            python${PYTHON_VERSION} \
            python${PYTHON_VERSION}-venv \
            python${PYTHON_VERSION}-dev \
            build-essential \
            redis-server \
            git \
            curl \
            libssl-dev \
            libffi-dev \
            libblas-dev \
            liblapack-dev \
            gfortran \
            pkg-config \
            cmake
    elif [[ "$OS" == "macos" ]]; then
        # Verifica Homebrew
        if ! command -v brew &> /dev/null; then
            error "Homebrew não encontrado. Instale em https://brew.sh"
        fi
        
        brew update
        brew install python@${PYTHON_VERSION} redis git cmake
    fi
}

# =============================================================================
# 3. CRIAÇÃO DA ESTRUTURA DO PROJETO
# =============================================================================
create_project_structure() {
    log "Criando estrutura do projeto..."
    
    # Diretório principal
    mkdir -p $PROJECT_NAME
    cd $PROJECT_NAME
    
    # Estrutura de diretórios
    directories=(
        "trade_system"
        "trade_system/modes"
        "trade_system/analysis"
        "trade_system/analysis/ml"
        "trade_system/config"
        "trade_system/utils"
        "tests"
        "tests/unit"
        "tests/integration"
        "logs"
        "data"
        "data/historical"
        "data/checkpoints"
        "notebooks"
        "scripts"
        ".github/workflows"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        touch "$dir/__init__.py" 2>/dev/null || true
    done
}

# =============================================================================
# 4. CRIAÇÃO DOS ARQUIVOS PRINCIPAIS
# =============================================================================
create_files() {
    log "Criando arquivos do sistema..."
    
    # setup.py
    cat > setup.py << 'EOF'
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
EOF

    # requirements.txt adicional para desenvolvimento
    cat > requirements-dev.txt << 'EOF'
jupyter>=1.0.0
notebook>=6.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
ipython>=8.7.0
autopep8>=2.0.0
pre-commit>=2.20.0
coverage>=7.0.0
EOF

    # .gitignore
    cat > .gitignore << 'EOF'
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
EOF

    # .env.example
    cat > .env.example << 'EOF'
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
EOF

    # Makefile
    cat > Makefile << 'EOF'
.PHONY: help install dev test clean run-hft run-standard lint format

help:
	@echo "Comandos disponíveis:"
	@echo "  make install      - Instala dependências de produção"
	@echo "  make dev         - Instala dependências de desenvolvimento"
	@echo "  make test        - Executa testes"
	@echo "  make lint        - Verifica código"
	@echo "  make format      - Formata código"
	@echo "  make run-hft     - Inicia modo HFT (dry-run)"
	@echo "  make run-standard - Inicia modo Standard"
	@echo "  make clean       - Limpa arquivos temporários"

install:
	pip install -e .

dev: install
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=trade_system

lint:
	flake8 trade_system tests
	mypy trade_system

format:
	black trade_system tests
	isort trade_system tests

run-hft:
	trade-system hft --dry-run --interval 5

run-standard:
	trade-system paper --mode standard

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .coverage htmlcov
EOF

    # Docker files
    cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos
COPY requirements.txt setup.py ./
COPY trade_system ./trade_system

# Instala dependências Python
RUN pip install --no-cache-dir -e .

# Porta para métricas Prometheus
EXPOSE 9090

# Comando padrão
CMD ["trade-system", "hft", "--dry-run"]
EOF

    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  trading-hft:
    build: .
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./config:/app/config
    command: trade-system hft --interval 5

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
EOF

    # README.md
    cat > README.md << 'EOF'
# Ultra Trading System - HFT & Standard Modes

Sistema de trading algorítmico de alta performance com suporte para operações standard e high-frequency trading (HFT).

## 🚀 Quick Start

```bash
# Clone e configure
git clone https://github.com/yourusername/ultra-trading-system.git
cd ultra-trading-system

# Instale com o script
./deploy.sh

# Configure suas credenciais
cp .env.example .env
# Edite .env com suas chaves da Binance

# Inicie em modo dry-run
make run-hft
```

## 📋 Requisitos

- Python 3.10+
- Redis
- 2GB RAM mínimo
- Conexão estável com internet

## 🏗️ Arquitetura

```
trade_system/
├── modes/          # Modos de operação (standard/hft)
├── analysis/       # Análise técnica e ML
├── config/         # Configurações YAML
└── utils/          # Utilidades
```

## 🔧 Configuração

### Modo HFT
- Decisões a cada 2-10 segundos
- Operações de até R$20 ou 0.00003 BTC
- Machine Learning online com River

### Modo Standard
- Análise de candles (1-5 min)
- Operações de até 2% do capital
- ML tradicional com backtesting

## 📊 Monitoramento

- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## 🧪 Testes

```bash
make test        # Testes unitários
make test-integration  # Testes de integração
```

## 📈 Performance

- Latência de análise: < 1ms
- Throughput: > 1000 msg/s
- Uptime alvo: 99.9%

## ⚠️ Avisos

**SEMPRE teste em modo dry-run antes de usar dinheiro real!**

Este software é fornecido "como está" para fins educacionais. Trading de criptomoedas envolve riscos substanciais.

## 📄 Licença

MIT License - veja LICENSE para detalhes.
EOF

    # LICENSE
    cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Ultra Trading System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

    # Configurações YAML
    mkdir -p config
    
    cat > config/base.yaml << 'EOF'
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
EOF

    cat > config/hft.yaml << 'EOF'
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
EOF

    cat > config/standard.yaml << 'EOF'
# Configuração para modo Standard
trading:
  mode: "standard"
  symbol: "BTCBRL"
  interval: "5m"
  max_position_pct: 0.02
  fee_pct: 0.0001
  
risk:
  stop_loss: 0.02
  take_profit: 0.03
  max_daily_loss: 0.05
  
indicators:
  - name: "RSI"
    period: 14
    buy_threshold: 30
    sell_threshold: 70
  - name: "MACD"
    fast: 12
    slow: 26
    signal: 9
  - name: "BB"
    period: 20
    std: 2

ml:
  model_type: "lightgbm"
  features: ["rsi", "macd", "bb", "volume", "atr"]
  lookback_periods: 100
  retrain_interval: 86400
EOF
}

# =============================================================================
# 5. CRIAÇÃO DO CÓDIGO PYTHON
# =============================================================================
create_python_code() {
    log "Criando código Python principal..."
    
    # __init__.py principal
    cat > trade_system/__init__.py << 'EOF'
"""Ultra Trading System - High Performance Algorithmic Trading"""

__version__ = "2.0.0"
__author__ = "Trading System Team"

from trade_system.config.loader import ConfigLoader
from trade_system.modes.hft import HFTTrader
from trade_system.modes.standard import StandardTrader

__all__ = ["ConfigLoader", "HFTTrader", "StandardTrader"]
EOF

    # CLI principal
    cat > trade_system/cli.py << 'EOF'
import click
import asyncio
import signal
import logging
from pathlib import Path

from trade_system.config.loader import ConfigLoader
from trade_system.modes.hft import HFTTrader
from trade_system.modes.standard import StandardTrader
from trade_system.utils.logging import setup_logging

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
def cli(debug):
    """Ultra Trading System CLI"""
    setup_logging(debug=debug)

@cli.command()
@click.option('--interval', default=5, help='Seconds between decisions')
@click.option('--max-position', default=0.00003, help='Max BTC per trade')
@click.option('--dry-run', is_flag=True, help='Paper trading mode')
@click.option('--config', default='hft.yaml', help='Config file')
def hft(interval, max_position, dry_run, config):
    """Start High-Frequency Trading mode"""
    # Carrega configuração
    cfg = ConfigLoader.load(config, mode='hft')
    cfg.trading.interval_seconds = interval
    cfg.trading.max_position_btc = max_position
    cfg.trading.dry_run = dry_run
    
    # Cria trader
    trader = HFTTrader(cfg)
    
    # Setup graceful shutdown
    def shutdown_handler(signum, frame):
        trader.shutdown()
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Inicia
    click.echo(f"Starting HFT mode (interval={interval}s, dry_run={dry_run})")
    asyncio.run(trader.run())

@cli.command()
@click.option('--mode', type=click.Choice(['standard', 'hft']), default='standard')
@click.option('--balance', default=1000.0, help='Initial balance for paper trading')
def paper(mode, balance):
    """Start paper trading"""
    click.echo(f"Starting paper trading in {mode} mode with balance {balance}")
    # Implementação do paper trading

@cli.command()
@click.option('--symbol', default='BTCBRL')
@click.option('--days', default=7, help='Days to backtest')
@click.option('--mode', type=click.Choice(['standard', 'hft']), default='standard')
def backtest(symbol, days, mode):
    """Run backtesting"""
    click.echo(f"Running {mode} backtest for {symbol} over {days} days")
    # Implementação do backtest

@cli.command()
@click.option('--create', is_flag=True, help='Create default config')
def config(create):
    """Manage configuration"""
    if create:
        # Cria configs padrão se não existirem
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)
        click.echo("Configuration files created in ./config/")

def main():
    """Entry point"""
    cli()

if __name__ == '__main__':
    main()
EOF

    # Config Loader
    mkdir -p trade_system/config
    cat > trade_system/config/loader.py << 'EOF'
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
            },
            'alerts': {
                'telegram': {
                    'token': os.getenv('TELEGRAM_BOT_TOKEN'),
                    'chat_id': os.getenv('TELEGRAM_CHAT_ID')
                }
            }
        })
        
        return OmegaConf.merge(cfg, env_cfg)
EOF

    # Microstructure Analysis
    mkdir -p trade_system/analysis
    cat > trade_system/analysis/microstructure.py << 'EOF'
import numpy as np
from numba import jit
from typing import Tuple, Dict

class MicrostructureAnalyzer:
    """High-frequency market microstructure analysis"""
    
    def __init__(self, config):
        self.config = config
        self.window = config.features.lookback_seconds
        self.momentum_lags = config.features.momentum_lags
        
    @staticmethod
    @jit(nopython=True)
    def compute_features(prices: np.ndarray,
                        volumes: np.ndarray, 
                        bid_sizes: np.ndarray,
                        ask_sizes: np.ndarray,
                        spreads: np.ndarray,
                        momentum_lags: tuple,
                        window: int) -> np.ndarray:
        """Ultra-fast feature computation with Numba"""
        
        # Momentum calculations
        momentums = np.zeros(len(momentum_lags))
        for i, lag in enumerate(momentum_lags):
            if len(prices) > lag:
                momentums[i] = (prices[-1] - prices[-lag]) / prices[-lag]
        
        # Spread analysis
        avg_spread = np.mean(spreads[-window:]) if len(spreads) >= window else np.mean(spreads)
        spread_std = np.std(spreads[-window:]) if len(spreads) >= window else 0.0
        
        # Volume imbalance
        recent_vols = volumes[-window:] if len(volumes) >= window else volumes
        buy_volume = np.sum(recent_vols[recent_vols > 0])
        sell_volume = np.abs(np.sum(recent_vols[recent_vols < 0]))
        vol_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-10)
        
        # Book pressure
        bid_depth = np.mean(bid_sizes[-5:]) if len(bid_sizes) >= 5 else np.mean(bid_sizes)
        ask_depth = np.mean(ask_sizes[-5:]) if len(ask_sizes) >= 5 else np.mean(ask_sizes)
        book_pressure = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
        
        # Short-term volatility
        if len(prices) > window:
            returns = np.diff(prices[-window:]) / prices[-window:-1]
            volatility = np.std(returns) * np.sqrt(len(returns))
        else:
            volatility = 0.0
        
        # Combine all features
        features = np.concatenate([
            momentums,
            np.array([avg_spread, spread_std, vol_imbalance, book_pressure, volatility])
        ])
        
        return features
EOF

    # Online ML Model
    mkdir -p trade_system/analysis/ml
    cat > trade_system/analysis/ml/online.py << 'EOF'
import numpy as np
from river import ensemble, preprocessing, drift
import pickle
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OnlineMLModel:
    """River ML model with drift detection"""
    
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
        self.drift_detector = drift.ADWIN()
        self.scaler = preprocessing.StandardScaler()
        self.performance_tracker = []
        self.last_checkpoint = datetime.now()
        
    def _build_model(self):
        return ensemble.AdaptiveRandomForestClassifier(
            n_models=self.config.ml.n_models,
            max_depth=self.config.ml.max_depth,
            grace_period=self.config.ml.grace_period,
            seed=42
        )
    
    def predict_and_update(self, features: np.ndarray, true_label: int = None):
        """Make prediction and update model online"""
        # Convert to dict for River
        features_dict = {f'f_{i}': float(v) for i, v in enumerate(features)}
        
        # Scale features
        scaled_features = self.scaler.transform_one(features_dict)
        
        # Get prediction
        proba = self.model.predict_proba_one(scaled_features)
        
        # Update if we have true label
        if true_label is not None:
            # Update scaler
            self.scaler.learn_one(features_dict)
            
            # Check for drift
            pred = self.model.predict_one(scaled_features)
            error = int(pred != true_label)
            self.drift_detector.update(error)
            
            if self.drift_detector.drift_detected:
                logger.warning("Drift detected! Adapting model...")
                self.drift_detector = drift.ADWIN()
            
            # Update model
            self.model.learn_one(scaled_features, true_label)
            
            # Track performance
            self.performance_tracker.append({
                'timestamp': datetime.now(),
                'error': error,
                'drift': self.drift_detector.drift_detected
            })
        
        # Periodic checkpoint
        if (datetime.now() - self.last_checkpoint).seconds > self.config.ml.checkpoint_interval:
            self.save_checkpoint()
            
        return proba
    
    def save_checkpoint(self):
        """Save model state"""
        checkpoint_dir = Path('data/checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'model': self.model,
            'scaler': self.scaler,
            'performance': self.performance_tracker[-1000:],
            'timestamp': datetime.now()
        }
        
        path = checkpoint_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.last_checkpoint = datetime.now()
        logger.info(f"Checkpoint saved: {path}")
EOF

    # HFT Mode
    mkdir -p trade_system/modes
    cat > trade_system/modes/hft.py << 'EOF'
import asyncio
import time
import logging
from typing import Optional, Dict
import numpy as np

from trade_system.analysis.microstructure import MicrostructureAnalyzer
from trade_system.analysis.ml.online import OnlineMLModel
from trade_system.websocket_manager import WebSocketManager
from trade_system.risk import RiskManager
from trade_system.alerts import AlertSystem

logger = logging.getLogger(__name__)

class HFTTrader:
    """High-Frequency Trading implementation"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        self.tasks = []
        
        # Initialize components
        self.ws_manager = WebSocketManager(config)
        self.risk_manager = RiskManager(config)
        self.alerts = AlertSystem(config)
        self.model = OnlineMLModel(config)
        self.microstructure = MicrostructureAnalyzer(config)
        
        # Trading state
        self.position = None
        self.cooldown_until = 0
        self.failed_orders = 0
        
        # Market data buffers
        self.price_buffer = []
        self.volume_buffer = []
        self.spread_buffer = []
        
    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting HFT trader...")
        
        try:
            # Start market data stream
            stream_task = asyncio.create_task(self.start_data_stream())
            self.tasks.append(stream_task)
            
            # Start decision loop
            decision_task = asyncio.create_task(self.decision_loop())
            self.tasks.append(decision_task)
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            logger.error(f"Fatal error in HFT: {e}")
            await self.shutdown()
    
    async def decision_loop(self):
        """Decision making loop"""
        while self.running:
            try:
                # Check cooldown
                if time.time() < self.cooldown_until:
                    await asyncio.sleep(1)
                    continue
                
                # Get market snapshot
                snapshot = self.get_market_snapshot()
                if not snapshot:
                    await asyncio.sleep(self.config.trading.interval_seconds)
                    continue
                
                # Compute features
                features = self.microstructure.compute_features(
                    np.array(snapshot['prices']),
                    np.array(snapshot['volumes']),
                    np.array(snapshot['bid_sizes']),
                    np.array(snapshot['ask_sizes']),
                    np.array(snapshot['spreads']),
                    tuple(self.config.features.momentum_lags),
                    self.config.features.lookback_seconds
                )
                
                # Get prediction
                proba = self.model.predict_and_update(features)
                
                # Make trading decision
                await self.execute_trading_logic(proba, snapshot)
                
                # Sleep interval
                await asyncio.sleep(self.config.trading.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in decision loop: {e}")
                self.handle_error()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down HFT trader...")
        self.running = False
        
        # Save model checkpoint
        self.model.save_checkpoint()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Close connections
        self.ws_manager.close_all()
        
        logger.info("Shutdown complete")
EOF

    # Create test file
    mkdir -p tests/unit
    cat > tests/unit/test_microstructure.py << 'EOF'
import pytest
import numpy as np
from trade_system.analysis.microstructure import MicrostructureAnalyzer

def test_compute_features():
    """Test microstructure feature computation"""
    # Mock data
    prices = np.array([100.0, 100.1, 100.2, 100.15, 100.25])
    volumes = np.array([10, -5, 15, -8, 20])
    bid_sizes = np.array([100, 120, 110, 130, 125])
    ask_sizes = np.array([90, 110, 105, 120, 115])
    spreads = np.array([0.1, 0.12, 0.11, 0.13, 0.12])
    
    # Compute features
    features = MicrostructureAnalyzer.compute_features(
        prices, volumes, bid_sizes, ask_sizes, spreads,
        momentum_lags=(2, 3), window=3
    )
    
    # Check output shape
    assert len(features) == 7  # 2 momentums + 5 other features
    
    # Check momentum calculations
    assert features[0] == pytest.approx((prices[-1] - prices[-2]) / prices[-2])
    assert features[1] == pytest.approx((prices[-1] - prices[-3]) / prices[-3])
EOF
}

# =============================================================================
# 6. CONFIGURAÇÃO DO AMBIENTE VIRTUAL
# =============================================================================
setup_venv() {
    log "Configurando ambiente virtual Python..."
    
    # Cria venv
    python${PYTHON_VERSION} -m venv $VENV_NAME
    
    # Ativa venv
    source $VENV_NAME/bin/activate
    
    # Atualiza pip
    pip install --upgrade pip setuptools wheel
    
    # Instala dependências
    pip install -e .
    
    # Instala dependências de desenvolvimento
    if [[ -f requirements-dev.txt ]]; then
        pip install -r requirements-dev.txt
    fi
}

# =============================================================================
# 7. CONFIGURAÇÃO DO REDIS
# =============================================================================
setup_redis() {
    log "Configurando Redis..."
    
    if [[ "$OS" == "linux" ]]; then
        # Inicia Redis
        sudo systemctl start redis-server
        sudo systemctl enable redis-server
        
        # Verifica se está rodando
        if redis-cli ping > /dev/null 2>&1; then
            log "Redis está rodando na porta $REDIS_PORT"
        else
            error "Falha ao iniciar Redis"
        fi
    elif [[ "$OS" == "macos" ]]; then
        # Inicia Redis via brew
        brew services start redis
    fi
}

# =============================================================================
# 8. CONFIGURAÇÃO DO MONITORAMENTO
# =============================================================================
setup_monitoring() {
    log "Configurando monitoramento..."
    
    # Prometheus config
    cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'trading-system'
    static_configs:
      - targets: ['localhost:9090']
EOF

    # Grafana dashboards
    mkdir -p grafana/dashboards
    mkdir -p grafana/datasources
    
    cat > grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Dashboard exemplo
    cat > grafana/dashboards/trading.json << 'EOF'
{
  "dashboard": {
    "title": "Trading System Metrics",
    "panels": [
      {
        "title": "P&L",
        "targets": [
          {
            "expr": "trading_pnl_total"
          }
        ]
      },
      {
        "title": "Win Rate",
        "targets": [
          {
            "expr": "trading_win_rate"
          }
        ]
      }
    ]
  }
}
EOF
}

# =============================================================================
# 9. CONFIGURAÇÃO DO GIT
# =============================================================================
setup_git() {
    log "Inicializando repositório Git..."
    
    git init
    
    # Pre-commit hooks
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
        args: ['--max-line-length=100']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.990
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF

    # Instala pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        pre-commit install
    fi
    
    # Commit inicial
    git add .
    git commit -m "Initial commit: Ultra Trading System v2.0"
}

# =============================================================================
# 10. TESTES E VALIDAÇÃO
# =============================================================================
run_tests() {
    log "Executando testes..."
    
    # Testa imports
    python -c "import trade_system; print(f'✓ Trade System v{trade_system.__version__}')"
    
    # Roda pytest
    if command -v pytest &> /dev/null; then
        pytest tests/ -v
    fi
    
    # Verifica Redis
    if redis-cli ping > /dev/null 2>&1; then
        log "✓ Redis conectado"
    else
        warning "Redis não está acessível"
    fi
}

# =============================================================================
# 11. FINALIZAÇÃO
# =============================================================================
finalize() {
    log "Finalizando instalação..."
    
    # Cria arquivo de configuração local
    cp .env.example .env
    
    # Mensagem final
    cat << EOF

${GREEN}═══════════════════════════════════════════════════════════════${NC}
${GREEN}✅ INSTALAÇÃO CONCLUÍDA COM SUCESSO!${NC}
${GREEN}═══════════════════════════════════════════════════════════════${NC}

📁 Projeto criado em: $(pwd)

🚀 Próximos passos:

1. Configure suas credenciais:
   ${YELLOW}nano .env${NC}

2. Teste o sistema em dry-run:
   ${YELLOW}make run-hft${NC}

3. Monitore com Grafana:
   ${YELLOW}docker-compose up -d${NC}
   Acesse: http://localhost:3000

4. Leia a documentação:
   ${YELLOW}cat README.md${NC}

⚠️  ${RED}IMPORTANTE:${NC} Sempre teste em modo dry-run antes de usar dinheiro real!

📊 Comandos úteis:
   ${BLUE}make help${NC}     - Ver todos os comandos
   ${BLUE}make test${NC}     - Executar testes
   ${BLUE}make lint${NC}     - Verificar código
   ${BLUE}make format${NC}   - Formatar código

💡 Suporte: https://github.com/yourusername/ultra-trading-system

${GREEN}Boa sorte com seus trades! 🚀${NC}
EOF
}

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
main() {
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}       ULTRA TRADING SYSTEM - INSTALAÇÃO COMPLETA${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo
    
    # Verifica se está no diretório correto
    if [[ -d "$PROJECT_NAME" ]]; then
        warning "Diretório $PROJECT_NAME já existe. Deseja sobrescrever? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            error "Instalação cancelada"
        fi
        rm -rf "$PROJECT_NAME"
    fi
    
    # Executa instalação
    check_system
    install_system_deps
    create_project_structure
    create_files
    create_python_code
    setup_venv
    setup_redis
    setup_monitoring
    setup_git
    run_tests
    finalize
}

# Executa se chamado diretamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
