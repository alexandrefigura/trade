import yaml

# Ajustar config.yaml para ser MUITO agressivo
config = {
    'trading': {
        'symbol': 'BTCUSDT',
        'min_confidence': 0.20,  # Apenas 20% de confiança
        'max_position_pct': 0.10,  # 10% por trade
        'take_profit_pct': 0.01,  # 1% de lucro
        'stop_loss_pct': 0.02,  # 2% de perda
    },
    'risk': {
        'max_daily_loss': 0.10,  # 10% perda máxima
        'min_balance_usd': 50.0,
        'max_pct_per_trade': 0.20,  # 20% por trade
        'min_trade_usd': 20.0,
    },
    'technical': {
        'rsi_buy_threshold': 55,  # Comprar mesmo com RSI alto
        'rsi_sell_threshold': 45,  # Vender mesmo com RSI baixo
        'rsi_period': 7,
        'sma_short_period': 3,
        'sma_long_period': 10,
        'buy_threshold': 0.01,  # Threshold mínimo
        'sell_threshold': 0.01,
    },
    'alerts': {
        'enable_alerts': False,  # Desabilitar até configurar
    },
    'redis': {
        'use_redis': False,
    }
}

with open('config.yaml', 'w') as f:
    yaml.dump(config, f)

print("✅ Sistema configurado para MÁXIMA agressividade!")
print("⚠️  ATENÇÃO: Isso é apenas para PAPER TRADING!")
print("Reinicie o sistema: trade-system paper")
