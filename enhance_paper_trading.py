#!/usr/bin/env python3
"""
Script para melhorar o sistema de Paper Trading
- Forçar execução de trades simulados
- Notificar cada trade via Telegram
- Mostrar lucro/prejuízo
- Aprender com os resultados
"""

import os
import re

def enhance_main_file():
    """Melhora o arquivo main.py para executar trades simulados"""
    main_file = "trade_system/main.py"
    
    if not os.path.exists(main_file):
        print(f"❌ Arquivo {main_file} não encontrado!")
        return False
    
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup
    with open(main_file + '.backup_enhanced', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Encontrar onde os trades são executados
    # Vamos modificar a lógica de decisão para sempre executar em paper trading
    
    # Adicionar método para forçar trades em paper mode
    force_trade_code = '''
    async def _should_execute_trade(self, action: str, confidence: float) -> bool:
        """Decide se deve executar o trade"""
        if self.mode == TradingMode.PAPER:
            # Em paper trading, executar trades com confiança > 40%
            # para ter mais dados de aprendizado
            return confidence > 0.40
        else:
            # Em modo real, ser mais conservador
            return confidence > self.config['trading']['min_confidence']
'''
    
    # Adicionar antes do método run
    if "_should_execute_trade" not in content:
        pattern = r'(async def run\(self\):)'
        replacement = force_trade_code + '\n\n' + r'\1'
        content = re.sub(pattern, replacement, content)
    
    # Modificar a lógica de execução
    # Procurar onde verifica a confiança
    pattern = r'if confidence >= self\.config\[\'trading\'\]\[\'min_confidence\'\]:'
    replacement = 'if await self._should_execute_trade(action, confidence):'
    content = re.sub(pattern, replacement, content)
    
    # Salvar
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {main_file} melhorado!")
    return True

def enhance_alerts():
    """Melhora o sistema de alertas para notificar cada trade"""
    alerts_file = "trade_system/alerts.py"
    
    if not os.path.exists(alerts_file):
        print(f"❌ Arquivo {alerts_file} não encontrado!")
        return False
    
    with open(alerts_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup
    with open(alerts_file + '.backup_enhanced', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Adicionar método para notificar trades
    trade_notification = '''
    def notify_trade_result(self, trade_type: str, entry_price: float, exit_price: float, 
                           amount: float, profit_loss: float, balance: float):
        """Notifica resultado de um trade via Telegram"""
        
        # Emoji baseado no resultado
        emoji = "✅" if profit_loss > 0 else "❌"
        profit_emoji = "📈" if profit_loss > 0 else "📉"
        
        # Calcular percentual
        percent = (profit_loss / (amount * entry_price)) * 100 if entry_price > 0 else 0
        
        message = f"""
{emoji} **TRADE FECHADO** {emoji}

**Tipo:** {trade_type}
**Entrada:** ${entry_price:,.2f}
**Saída:** ${exit_price:,.2f}
**Quantidade:** {amount:.8f} BTC

{profit_emoji} **Resultado:** ${profit_loss:,.2f} ({percent:+.2f}%)

💰 **Saldo Atual:** ${balance:,.2f}

#PaperTrading #{'Lucro' if profit_loss > 0 else 'Prejuízo'}
"""
        
        self._send_telegram(message)
        
        # Log para aprendizado
        self.logger.info(f"Trade Result - Type: {trade_type}, P/L: ${profit_loss:.2f}, Balance: ${balance:.2f}")
    
    def notify_daily_summary(self, total_trades: int, winning_trades: int, 
                           total_profit: float, balance: float, initial_balance: float):
        """Envia resumo diário"""
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
        
        message = f"""
📊 **RESUMO DIÁRIO** 📊

**Total de Trades:** {total_trades}
**Trades Vencedores:** {winning_trades} ({win_rate:.1f}%)
**Lucro/Prejuízo do Dia:** ${total_profit:,.2f}

**Saldo Inicial:** ${initial_balance:,.2f}
**Saldo Final:** ${balance:,.2f}
**Retorno:** {total_return:+.2f}%

#PaperTrading #ResumoDiário
"""
        
        self._send_telegram(message)
'''
    
    # Adicionar os métodos antes do final da classe
    if "notify_trade_result" not in content:
        # Encontrar o final da classe AlertManager
        pattern = r'(class AlertManager:.*?)((?=class)|$)'
        
        def replacer(match):
            class_content = match.group(1)
            # Adicionar antes do final da classe
            return class_content.rstrip() + '\n' + trade_notification + '\n\n' + match.group(2)
        
        content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    # Salvar
    with open(alerts_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {alerts_file} melhorado!")
    return True

def create_learning_module():
    """Cria módulo de aprendizado para analisar trades"""
    learning_file = "trade_system/learning.py"
    
    learning_code = '''#!/usr/bin/env python3
"""
Módulo de aprendizado - Analisa trades e melhora estratégias
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np

class TradeLearning:
    """Sistema de aprendizado com trades"""
    
    def __init__(self):
        self.trades_file = "data/trades_history.json"
        self.patterns_file = "data/winning_patterns.json"
        self.ensure_data_dir()
        self.load_data()
    
    def ensure_data_dir(self):
        """Garante que o diretório de dados existe"""
        os.makedirs("data", exist_ok=True)
    
    def load_data(self):
        """Carrega histórico de trades"""
        if os.path.exists(self.trades_file):
            with open(self.trades_file, 'r') as f:
                self.trades = json.load(f)
        else:
            self.trades = []
        
        if os.path.exists(self.patterns_file):
            with open(self.patterns_file, 'r') as f:
                self.patterns = json.load(f)
        else:
            self.patterns = {}
    
    def save_data(self):
        """Salva dados"""
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2)
        
        with open(self.patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def record_trade(self, trade_data: Dict):
        """Registra um trade para análise"""
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade_data)
        self.save_data()
        
        # Analisar padrões
        self.analyze_patterns()
    
    def analyze_patterns(self):
        """Analisa padrões vencedores e perdedores"""
        if len(self.trades) < 10:
            return
        
        winning_trades = [t for t in self.trades if t['profit_loss'] > 0]
        losing_trades = [t for t in self.trades if t['profit_loss'] < 0]
        
        # Padrões de trades vencedores
        if winning_trades:
            self.patterns['winning'] = {
                'avg_rsi': np.mean([t['indicators']['rsi'] for t in winning_trades]),
                'avg_volume_ratio': np.mean([t['indicators']['volume_ratio'] for t in winning_trades]),
                'avg_confidence': np.mean([t['confidence'] for t in winning_trades]),
                'common_action': max(set([t['action'] for t in winning_trades]), 
                                   key=[t['action'] for t in winning_trades].count),
                'best_hours': self.get_best_trading_hours(winning_trades)
            }
        
        # Padrões de trades perdedores
        if losing_trades:
            self.patterns['losing'] = {
                'avg_rsi': np.mean([t['indicators']['rsi'] for t in losing_trades]),
                'avg_volume_ratio': np.mean([t['indicators']['volume_ratio'] for t in losing_trades]),
                'avg_confidence': np.mean([t['confidence'] for t in losing_trades]),
                'common_action': max(set([t['action'] for t in losing_trades]), 
                                   key=[t['action'] for t in losing_trades].count),
                'worst_hours': self.get_best_trading_hours(losing_trades)
            }
        
        self.save_data()
    
    def get_best_trading_hours(self, trades: List[Dict]) -> List[int]:
        """Identifica melhores horários para trading"""
        hours = [datetime.fromisoformat(t['timestamp']).hour for t in trades]
        hour_counts = {}
        for h in hours:
            hour_counts[h] = hour_counts.get(h, 0) + 1
        
        # Top 3 horários
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [h[0] for h in sorted_hours[:3]]
    
    def get_recommendations(self, current_indicators: Dict) -> Dict:
        """Gera recomendações baseadas no aprendizado"""
        if 'winning' not in self.patterns:
            return {'action': 'WAIT', 'reason': 'Dados insuficientes para análise'}
        
        winning = self.patterns['winning']
        current_hour = datetime.now().hour
        
        # Verificar se é um bom horário
        good_hour = current_hour in winning.get('best_hours', [])
        
        # Comparar indicadores
        rsi_match = abs(current_indicators.get('rsi', 50) - winning['avg_rsi']) < 10
        volume_match = current_indicators.get('volume_ratio', 1) > winning['avg_volume_ratio'] * 0.8
        
        if good_hour and rsi_match and volume_match:
            return {
                'action': winning['common_action'],
                'confidence_boost': 0.1,
                'reason': f"Padrão vencedor detectado (RSI: {winning['avg_rsi']:.1f}, Volume: {winning['avg_volume_ratio']:.2f})"
            }
        
        return {'action': 'WAIT', 'reason': 'Condições não correspondem a padrões vencedores'}
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas gerais"""
        if not self.trades:
            return {}
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['profit_loss'] > 0])
        total_profit = sum(t['profit_loss'] for t in self.trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_profit': total_profit,
            'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0,
            'best_trade': max(self.trades, key=lambda t: t['profit_loss'])['profit_loss'] if self.trades else 0,
            'worst_trade': min(self.trades, key=lambda t: t['profit_loss'])['profit_loss'] if self.trades else 0
        }

# Instância global
learning_system = TradeLearning()
'''
    
    with open(learning_file, 'w', encoding='utf-8') as f:
        f.write(learning_code)
    
    print(f"✅ Módulo de aprendizado criado: {learning_file}")
    return True

def update_config_for_paper():
    """Atualiza configuração para paper trading mais ativo"""
    config_file = "config.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ Arquivo {config_file} não encontrado!")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup
    with open(config_file + '.backup_paper', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Ajustar configurações para paper trading
    # Reduzir min_confidence para 0.40 para mais trades de aprendizado
    content = re.sub(r'min_confidence:\s*[\d.]+', 'min_confidence: 0.40', content)
    
    # Adicionar configuração de paper trading se não existir
    if 'paper_trading:' not in content:
        paper_config = '''
# Configurações específicas para Paper Trading
paper_trading:
  force_trade_threshold: 0.40  # Executar trades com 40% de confiança
  max_trades_per_day: 20       # Máximo de trades por dia
  learning_mode: true          # Modo de aprendizado ativo
  notify_all_trades: true      # Notificar todos os trades via Telegram
'''
        content += '\n' + paper_config
    
    # Salvar
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {config_file} atualizado para paper trading!")
    return True

def main():
    print("🎓 Melhorando sistema de Paper Trading para aprendizado...\n")
    
    # 1. Melhorar main.py
    print("1. Atualizando lógica de execução de trades...")
    enhance_main_file()
    
    # 2. Melhorar alertas
    print("\n2. Melhorando sistema de notificações...")
    enhance_alerts()
    
    # 3. Criar módulo de aprendizado
    print("\n3. Criando módulo de aprendizado...")
    create_learning_module()
    
    # 4. Atualizar configurações
    print("\n4. Atualizando configurações...")
    update_config_for_paper()
    
    print("\n✅ Sistema melhorado com sucesso!")
    print("\n📋 Novas funcionalidades:")
    print("- Execução automática de trades com 40% de confiança")
    print("- Notificação via Telegram de cada trade fechado")
    print("- Análise de padrões vencedores e perdedores")
    print("- Aprendizado contínuo com os resultados")
    print("- Resumo diário de performance")
    
    print("\n🚀 Execute novamente: trade-system paper")
    print("\n💡 O sistema agora vai:")
    print("1. Executar mais trades para aprender")
    print("2. Notificar cada operação no Telegram")
    print("3. Mostrar lucro/prejuízo e saldo")
    print("4. Aprender com os padrões de sucesso e falha")

if __name__ == "__main__":
    main()
