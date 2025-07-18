"""
Módulo de gestão de risco com proteções aprimoradas
"""
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
import time

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Representa uma posição aberta"""
    side: str
    entry_price: float
    quantity: float
    size: float
    entry_time: float
    stop_loss: float = None
    take_profit: float = None


class RiskManager:
    """Gerenciador de risco com stop loss e take profit automáticos"""
    
    def __init__(self, config):
        self.config = config.get('risk', config)
        
        # Balanço
        self.initial_balance = config.get('initial_balance', 10000)
        self.current_balance = self.initial_balance
        
        # Parâmetros de risco
        self.max_position_pct = self.config.get('max_position_pct', 0.02)
        self.max_volatility = self.config.get('max_volatility', 0.03)
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)
        self.tp_multiplier = self.config.get('tp_multiplier', 1.5)
        self.sl_multiplier = self.config.get('sl_multiplier', 1.0)
        
        # Novos parâmetros
        self.positions: List[Position] = []
        self.max_positions = self.config.get('max_positions', 1)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 2.0)
        self.take_profit_pct = self.config.get('take_profit_pct', 3.0)
        
        # Estatísticas
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.daily_trades = 0
        self.max_daily_trades = self.config.get('max_daily_trades', 50)
        
        # Estado da posição (compatibilidade)
        self.position = None
        
        logger.info(f"💰 Risk Manager inicializado - Balance: ${self.current_balance:,.2f}")
        logger.info(f"📊 Stop Loss: {self.stop_loss_pct}% | Take Profit: {self.take_profit_pct}%")
    
    def has_open_position(self) -> bool:
        """Verifica se há posição aberta"""
        return len(self.positions) > 0
    
    def get_open_position(self) -> Optional[Position]:
        """Retorna a primeira posição aberta"""
        return self.positions[0] if self.positions else None
    
    def can_open_position(self, side: str) -> bool:
        """Verifica se pode abrir nova posição"""
        # Verifica limite de posições
        if len(self.positions) >= self.max_positions:
            logger.warning(f"⚠️ Limite de posições atingido: {len(self.positions)}/{self.max_positions}")
            return False
        
        # Verifica se já tem posição no mesmo lado
        for pos in self.positions:
            if pos.side == side:
                logger.warning(f"⚠️ Já existe posição {side} aberta")
                return False
        
        # Verifica limite diário de trades
        if self.daily_trades >= self.max_daily_trades:
            logger.warning(f"⚠️ Limite diário de trades atingido: {self.daily_trades}")
            return False
        
        # Verifica stop loss diário
        if abs(self.daily_pnl) >= self.initial_balance * self.max_daily_loss:
            logger.warning(f"⚠️ Stop loss diário atingido: ${self.daily_pnl:.2f}")
            return False
        
        return True
    
    def calculate_position_size(self, confidence: float, volatility: float, price: float) -> float:
        """Calcula tamanho da posição baseado no risco"""
        # Tamanho base
        base_size = self.current_balance * self.max_position_pct
        
        # Ajusta por volatilidade
        if volatility > self.max_volatility:
            vol_factor = self.max_volatility / volatility
            base_size *= vol_factor
            logger.debug(f"Volatilidade alta ({volatility:.2%}), reduzindo posição em {(1-vol_factor):.1%}")
        
        # Ajusta por confiança
        confidence_factor = min(confidence, 1.0)
        position_size = base_size * confidence_factor
        
        # Limites
        min_size = 10.0  # Mínimo $10
        max_size = self.current_balance * 0.1  # Máximo 10% do balanço
        
        position_size = max(min_size, min(position_size, max_size))
        
        logger.debug(f"[DEBUG] Pos size: ${position_size:.2f} | conf: {confidence:.2f} | vol: {volatility:.4f} | qty: {position_size/price:.6f}")
        
        return position_size
    
    def open_position(self, side: str, price: float, quantity: float, size: float, current_time: float) -> Optional[Position]:
        """Abre nova posição com stop loss e take profit"""
        if not self.can_open_position(side):
            return None
        
        # Calcula stop loss e take profit
        if side == 'BUY':
            stop_loss = price * (1 - self.stop_loss_pct / 100)
            take_profit = price * (1 + self.take_profit_pct / 100)
        else:  # SELL
            stop_loss = price * (1 + self.stop_loss_pct / 100)
            take_profit = price * (1 - self.take_profit_pct / 100)
        
        position = Position(
            side=side,
            entry_price=price,
            quantity=quantity,
            size=size,
            entry_time=current_time,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions.append(position)
        self.position = position  # Compatibilidade
        self.daily_trades += 1
        
        logger.info(f"   Stop Loss: ${stop_loss:,.2f} | Take Profit: ${take_profit:,.2f}")
        
        return position
    
    def check_exit_conditions(self, position: Position, current_price: float) -> Optional[str]:
        """Verifica condições de saída"""
        if position.side == 'BUY':
            if current_price <= position.stop_loss:
                return 'STOP_LOSS'
            elif current_price >= position.take_profit:
                return 'TAKE_PROFIT'
        else:  # SELL
            if current_price >= position.stop_loss:
                return 'STOP_LOSS'
            elif current_price <= position.take_profit:
                return 'TAKE_PROFIT'
        
        return None
    
    def close_position(self, position: Position, current_price: float, reason: str) -> dict:
        """Fecha posição e calcula resultado"""
        # Calcula P&L
        if position.side == 'BUY':
            pnl = (current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - current_price) * position.quantity
        
        pnl_pct = (pnl / position.size) * 100
        
        # Atualiza estatísticas
        self.daily_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            logger.info(f"✅ Posição fechada com LUCRO: ${pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            self.losing_trades += 1
            logger.info(f"❌ Posição fechada com PREJUÍZO: ${pnl:.2f} ({pnl_pct:.2f}%)")
        
        # Remove posição
        self.positions.remove(position)
        self.position = None  # Compatibilidade
        
        # Atualiza balanço
        self.current_balance += pnl
        
        return {
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'new_balance': self.current_balance
        }
    
    # Métodos de compatibilidade com código existente
    def set_position(self, position_dict: dict):
        """Compatibilidade com código antigo"""
        self.position = position_dict
    
    def clear_position(self):
        """Compatibilidade com código antigo"""
        self.position = None
        self.positions.clear()
    
    def update_after_trade(self, pnl: float, fee: float):
        """Compatibilidade com código antigo"""
        # Já atualizado em close_position, mas mantém compatibilidade
        pass
    
    def should_close_position(self, current_price: float, entry_price: float, side: str) -> tuple:
        """Compatibilidade com código antigo"""
        # Verifica stop loss e take profit manualmente
        if side == 'BUY':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            if pnl_pct <= -self.stop_loss_pct:
                return True, 'STOP_LOSS'
            elif pnl_pct >= self.take_profit_pct:
                return True, 'TAKE_PROFIT'
        else:  # SELL
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            if pnl_pct <= -self.stop_loss_pct:
                return True, 'STOP_LOSS'
            elif pnl_pct >= self.take_profit_pct:
                return True, 'TAKE_PROFIT'
        
        return False, None
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas de risco"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'daily_trades': self.daily_trades,
            'open_positions': len(self.positions),
            'return_pct': ((self.current_balance - self.initial_balance) / self.initial_balance * 100)
        }
    
    def reset_daily_stats(self):
        """Reseta estatísticas diárias"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        logger.info("📊 Estatísticas diárias resetadas")


# Alias para compatibilidade
UltraFastRiskManager = RiskManager
