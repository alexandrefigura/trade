#!/usr/bin/env python3
"""
Logger simples para registrar trades e enviar notificações
"""

import json
import os
from datetime import datetime
from typing import Dict

class TradeLogger:
    """Registra todos os trades para análise"""
    
    def __init__(self, alerts_manager=None):
        self.trades_file = "logs/trades_paper.json"
        self.alerts = alerts_manager
        self.ensure_log_dir()
        self.load_trades()
    
    def ensure_log_dir(self):
        """Garante que o diretório existe"""
        os.makedirs(os.path.dirname(self.trades_file), exist_ok=True)
    
    def load_trades(self):
        """Carrega histórico de trades"""
        if os.path.exists(self.trades_file):
            with open(self.trades_file, 'r') as f:
                self.trades = json.load(f)
        else:
            self.trades = []
    
    def save_trades(self):
        """Salva trades"""
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def log_trade(self, trade_data: Dict):
        """Registra um trade e notifica via Telegram"""
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade_data)
        self.save_trades()
        
        # Notificar via Telegram
        if self.alerts and 'exit_price' in trade_data:
            self.notify_closed_trade(trade_data)
    
    def notify_closed_trade(self, trade: Dict):
        """Notifica trade fechado via Telegram"""
        profit_loss = trade.get('profit_loss', 0)
        emoji = "✅" if profit_loss > 0 else "❌"
        
        message = f"""
{emoji} **Trade Fechado**

Tipo: {trade.get('type', 'N/A')}
Entrada: ${trade.get('entry_price', 0):,.2f}
Saída: ${trade.get('exit_price', 0):,.2f}
Resultado: ${profit_loss:,.2f}
Saldo: ${trade.get('balance', 0):,.2f}

#PaperTrading
"""
        
        if self.alerts:
            self.alerts.send_trade_alert(
                action=trade.get('type', 'TRADE'),
                price=trade.get('exit_price', 0),
                confidence=trade.get('confidence', 0),
                reason=f"P/L: ${profit_loss:,.2f}"
            )

# Instância global
trade_logger = None

def init_trade_logger(alerts_manager=None):
    """Inicializa o logger"""
    global trade_logger
    trade_logger = TradeLogger(alerts_manager)
    return trade_logger
