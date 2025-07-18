# Adicione estas modificações ao risk.py existente

# No início do arquivo, adicione:
from dataclasses import dataclass
from typing import Optional, List

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

# Na classe RiskManager existente, adicione estes novos atributos no __init__:
def __init__(self, config):
    # ... código existente ...
    
    # Novos atributos
    self.positions: List[Position] = []
    self.max_positions = config.get('max_positions', 1)
    self.stop_loss_pct = config.get('stop_loss_pct', 2.0)
    self.take_profit_pct = config.get('take_profit_pct', 3.0)
    self.daily_pnl = 0.0
    self.total_trades = 0
    self.winning_trades = 0
    self.losing_trades = 0

# Adicione estes novos métodos à classe RiskManager:

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
    
    return True

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
    
    # Atualiza balanço
    self.balance += pnl
    
    return {
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'reason': reason,
        'new_balance': self.balance
    }
