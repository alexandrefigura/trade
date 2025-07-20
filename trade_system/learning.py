#!/usr/bin/env python3
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
