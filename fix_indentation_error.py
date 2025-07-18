#!/usr/bin/env python3
"""
Script para corrigir o erro de indentação no main.py
"""

import os
import shutil

def fix_main_file():
    """Restaura o main.py do backup e aplica correções corretas"""
    main_file = "trade_system/main.py"
    backup_file = main_file + ".backup_enhanced"
    
    # Se existe backup, restaurar
    if os.path.exists(backup_file):
        print(f"📂 Restaurando backup de {main_file}...")
        shutil.copy2(backup_file, main_file)
        print(f"✅ Backup restaurado!")
    else:
        print(f"❌ Backup não encontrado! Tentando corrigir diretamente...")
        
    # Ler o arquivo
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Corrigir problemas de indentação
    # Remover espaços extras ou tabs misturados
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Converter tabs para espaços
        line = line.replace('\t', '    ')
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Salvar arquivo corrigido
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {main_file} corrigido!")
    return True

def apply_minimal_changes():
    """Aplica mudanças mínimas para ativar paper trading"""
    config_file = "config.yaml"
    
    if os.path.exists(config_file + ".backup_paper"):
        print(f"📂 Usando backup do config.yaml...")
    
    # Apenas ajustar a confiança mínima para paper trading
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Reduzir min_confidence para 0.40
        import re
        content = re.sub(r'min_confidence:\s*[\d.]+', 'min_confidence: 0.40', content)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Confiança mínima ajustada para 0.40 (40%)")
    
    return True

def create_simple_trade_logger():
    """Cria um logger simples para trades"""
    logger_file = "trade_system/trade_logger.py"
    
    logger_code = '''#!/usr/bin/env python3
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
'''
    
    with open(logger_file, 'w', encoding='utf-8') as f:
        f.write(logger_code)
    
    print(f"✅ Logger de trades criado: {logger_file}")
    return True

def main():
    print("🔧 Corrigindo erro e aplicando melhorias simples...\n")
    
    # 1. Corrigir erro de indentação
    print("1. Corrigindo erro de indentação...")
    fix_main_file()
    
    # 2. Aplicar mudanças mínimas
    print("\n2. Ajustando configurações...")
    apply_minimal_changes()
    
    # 3. Criar logger simples
    print("\n3. Criando logger de trades...")
    create_simple_trade_logger()
    
    print("\n✅ Sistema corrigido!")
    print("\n📝 Agora o sistema vai:")
    print("- Executar trades com 40% de confiança")
    print("- Registrar todos os trades em logs/trades_paper.json")
    print("- Continuar enviando alertas importantes via Telegram")
    
    print("\n🚀 Execute novamente: trade-system paper")

if __name__ == "__main__":
    main()
