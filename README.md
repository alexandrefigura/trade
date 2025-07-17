# Sistema de Trading Ultra-Otimizado v5.2

Sistema de trading algorítmico de alta performance com análise técnica ultra-rápida, machine learning e paper trading.

## 🚀 Características

- **Análise Técnica Ultra-Rápida**: Indicadores otimizados com Numba para latência < 1ms
- **WebSocket em Tempo Real**: Dados de mercado com buffer otimizado
- **Machine Learning**: Predições adaptativas baseadas em padrões
- **Paper Trading**: Teste estratégias com dados reais sem risco
- **Sistema de Alertas**: Notificações via Telegram e email
- **Gestão de Risco**: Proteções avançadas e stop loss dinâmico
- **Cache Redis**: Performance máxima com fallback local
- **Rate Limiting**: Proteção contra limites de API

## 📦 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/yourusername/ultra-trading-system.git
cd ultra-trading-system
```

### 2. Crie ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale o pacote
```bash
pip install -e .
```

### 4. Configure as credenciais
```bash
export BINANCE_API_KEY="sua_api_key"
export BINANCE_API_SECRET="sua_api_secret"

# Opcional - para alertas
export TELEGRAM_BOT_TOKEN="seu_token"
export TELEGRAM_CHAT_ID="seu_chat_id"
```

### 5. Crie arquivo de configuração
```bash
trade-system config --create
```

## 🎮 Uso Rápido

### Paper Trading (Recomendado)
```bash
# Iniciar paper trading com backtest de validação
trade-system paper

# Modo debug com parâmetros agressivos
trade-system paper --debug

# Definir balance inicial
trade-system paper --balance 5000
```

### Backtest
```bash
# Backtest padrão (7 dias)
trade-system backtest

# Backtest com mais dados
trade-system backtest --days 30

# Backtest de outro par
trade-system backtest --symbol ETHUSDT
```

## ⚙️ Configuração

Edite `config.yaml` para personalizar:

```yaml
trading:
  symbol: "BTCUSDT"
  min_confidence: 0.75  # Confiança mínima para trades
  max_position_pct: 0.02  # Máximo 2% do balance por posição

risk:
  max_volatility: 0.05  # Máxima volatilidade aceita
  max_spread_bps: 20  # Máximo spread em basis points
  max_daily_loss: 0.02  # Stop loss diário de 2%

ta:
  rsi_buy_threshold: 30
  rsi_sell_threshold: 70
  # ... mais parâmetros
```

## 📊 Arquitetura

```
trade_system/
├── config.py             # Configurações e parâmetros
├── logging_config.py     # Sistema de logging centralizado
├── alerts.py             # Sistema de alertas multi-canal
├── cache.py              # Cache ultra-rápido com Redis
├── rate_limiter.py       # Controle de rate limiting
├── websocket_manager.py  # WebSocket para dados em tempo real
├── analysis/
│   ├── technical.py      # Análise técnica com Numba
│   ├── orderbook.py      # Análise de orderbook
│   └── ml.py             # Machine Learning simplificado
├── risk.py               # Gestão de risco e validações
├── backtester.py         # Sistema de backtesting
├── signals.py            # Consolidação de sinais
├── checkpoint.py         # Sistema de checkpoints
└── cli.py                # Interface de linha de comando
```

## 🔒 Segurança

- **Paper Trading por Padrão**: Sempre teste em modo simulado primeiro
- **Validações de Mercado**: Proteção contra condições extremas
- **Stop Loss Diário**: Limite de perda configurável
- **Rate Limiting**: Proteção contra banimento de API
- **Checkpoints**: Recuperação automática em caso de falha

## 📈 Performance

- Latência de análise técnica: < 1ms
- Throughput WebSocket: > 1000 msg/s
- Cache Redis: < 0.1ms de latência
- Consumo de memória: < 500MB típico

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/amazing`)
3. Commit suas mudanças (`git commit -m 'Add amazing feature'`)
4. Push para a branch (`git push origin feature/amazing`)
5. Abra um Pull Request

## ⚠️ Disclaimer

Este software é fornecido "como está" para fins educacionais. Trading de criptomoedas envolve riscos substanciais. Sempre teste em paper trading antes de usar dinheiro real.

## 📝 Licença

MIT License - veja LICENSE para detalhes.
