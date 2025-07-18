#!/usr/bin/env python3
"""
Script para forçar a execução de trades em Paper Trading
"""

import os
import re

def modify_risk_manager():
    """Modifica o gerenciador de risco para ser menos restritivo em paper trading"""
    risk_file = "trade_system/risk.py"
    
    if not os.path.exists(risk_file):
        print(f"❌ Arquivo {risk_file} não encontrado!")
        return False
    
    with open(risk_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup
    with open(risk_file + '.backup_paper', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Adicionar lógica especial para paper trading
    paper_logic = '''
        # Em paper trading, ser menos restritivo
        if hasattr(self, 'mode') and str(getattr(self, 'mode', '')).upper() == 'PAPER':
            # Aceitar trades com menor confiança
            if confidence < 0.35:  # Mínimo de 35% em paper
                return False, "Confiança muito baixa para paper trading"
            
            # Permitir mais trades para aprendizado
            return True, "Paper trading - trade permitido para aprendizado"
'''
    
    # Inserir após as verificações de risco básicas
    # Procurar por "def can_trade"
    pattern = r'(def can_trade.*?:\s*\n)(.*?)(return True)'
    
    def replacer(match):
        method_def = match.group(1)
        method_body = match.group(2)
        
        # Inserir a lógica de paper trading antes do return True
        return method_def + method_body + paper_logic + '\n        ' + match.group(3)
    
    content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    # Salvar
    with open(risk_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {risk_file} modificado para paper trading!")
    return True

def add_trade_notification():
    """Adiciona notificação específica para cada trade executado"""
    main_file = "trade_system/main.py"
    
    if not os.path.exists(main_file):
        print(f"❌ Arquivo {main_file} não encontrado!")
        return False
    
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Adicionar import do trade_logger se não existir
    if 'from trade_system.trade_logger import' not in content:
        imports_section = "from trade_system.checkpoint import CheckpointManager"
        content = content.replace(
            imports_section,
            imports_section + "\nfrom trade_system.trade_logger import init_trade_logger, trade_logger"
        )
    
    # Inicializar trade_logger no __init__
    init_pattern = r'(self\.checkpoint = CheckpointManager.*?\n)'
    init_replacement = r'\1        \n        # Inicializar logger de trades\n        init_trade_logger(self.alerts)\n'
    content = re.sub(init_pattern, init_replacement, content)
    
    # Adicionar notificação após executar trade
    # Procurar onde o trade é executado
    trade_notification = '''
            # Notificar trade em paper trading
            if self.mode == TradingMode.PAPER and trade_logger:
                trade_data = {
                    'type': action,
                    'entry_price': price,
                    'amount': amount,
                    'confidence': confidence,
                    'balance': self.risk_manager.balance,
                    'indicators': {
                        'rsi': data.get('indicators', {}).get('rsi', 0),
                        'volume_ratio': data.get('indicators', {}).get('volume_ratio', 0)
                    }
                }
                trade_logger.log_trade(trade_data)
                
                # Notificação simplificada
                self.alerts.send_trade_alert(
                    action=action,
                    price=price,
                    confidence=confidence,
                    reason=f"Paper Trade - Balance: ${self.risk_manager.balance:,.2f}"
                )
'''
    
    # Inserir após "await self._execute_trade"
    pattern = r'(await self\._execute_trade\(action, amount, price\))'
    replacement = r'\1' + trade_notification
    content = re.sub(pattern, replacement, content)
    
    # Salvar
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {main_file} modificado para notificações!")
    return True

def update_config_aggressive():
    """Torna a configuração mais agressiva para paper trading"""
    config_file = "config.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ Arquivo {config_file} não encontrado!")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ajustes para paper trading mais ativo
    replacements = [
        (r'min_confidence:\s*[\d.]+', 'min_confidence: 0.35'),  # 35% mínimo
        (r'max_position_pct:\s*[\d.]+', 'max_position_pct: 0.05'),  # 5% por trade
        (r'rsi_buy_threshold:\s*[\d.]+', 'rsi_buy_threshold: 35'),  # Mais oportunidades
        (r'rsi_sell_threshold:\s*[\d.]+', 'rsi_sell_threshold: 65'),
        (r'stop_loss_pct:\s*[\d.]+', 'stop_loss_pct: 0.03'),  # Stop loss 3%
        (r'take_profit_pct:\s*[\d.]+', 'take_profit_pct: 0.02'),  # Take profit 2%
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Salvar
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {config_file} ajustado para paper trading agressivo!")
    return True

def main():
    print("🚀 Configurando sistema para Paper Trading ativo...\n")
    
    # 1. Modificar gerenciador de risco
    print("1. Ajustando gerenciador de risco...")
    modify_risk_manager()
    
    # 2. Adicionar notificações
    print("\n2. Configurando notificações de trades...")
    add_trade_notification()
    
    # 3. Tornar configuração mais agressiva
    print("\n3. Ajustando parâmetros para mais trades...")
    update_config_aggressive()
    
    print("\n✅ Sistema configurado para Paper Trading ativo!")
    print("\n📋 Mudanças aplicadas:")
    print("- Confiança mínima: 35%")
    print("- Tamanho da posição: até 5% do saldo")
    print("- RSI mais sensível (compra < 35, venda > 65)")
    print("- Stop loss: 3% / Take profit: 2%")
    print("- Notificações via Telegram para cada trade")
    
    print("\n⚠️ IMPORTANTE: Reinicie o sistema!")
    print("1. Pressione Ctrl+C para parar o sistema atual")
    print("2. Execute novamente: trade-system paper")
    
    print("\n💡 Agora o sistema vai:")
    print("- Executar mais trades para aprender")
    print("- Notificar cada operação no Telegram")
    print("- Mostrar saldo atualizado após cada trade")

if __name__ == "__main__":
    main()
