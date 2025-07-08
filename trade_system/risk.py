"""
Gestão de risco avançada com Kelly Criterion otimizado e proteções de drawdown
"""
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class TradingState(Enum):
    """Estados possíveis do sistema de trading"""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    RESTRICTED = "RESTRICTED"
    KILL_SWITCH = "KILL_SWITCH"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class KellyParameters:
    """Parâmetros para cálculo do Kelly Criterion"""
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 1.5
    kelly_fraction: float = 0.0
    kelly_multiplier: float = 0.25  # Fator de segurança (Kelly fracionário)
    min_kelly: float = 0.01  # Mínimo 1%
    max_kelly: float = 0.25  # Máximo 25%
    
    def calculate(self):
        """Calcula o Kelly Criterion otimizado"""
        if self.win_rate <= 0 or self.win_loss_ratio <= 0:
            self.kelly_fraction = self.min_kelly
            return
        
        # Fórmula clássica do Kelly
        # f = p - q/b
        # onde: p = probabilidade de ganho, q = probabilidade de perda, b = ratio ganho/perda
        q = 1 - self.win_rate
        kelly = self.win_rate - (q / self.win_loss_ratio)
        
        # Aplicar fator de segurança (Kelly fracionário)
        kelly *= self.kelly_multiplier
        
        # Aplicar limites
        self.kelly_fraction = max(self.min_kelly, min(kelly, self.max_kelly))
        
        return self.kelly_fraction


@dataclass
class DrawdownProtection:
    """Sistema de proteção contra drawdown"""
    max_daily_drawdown: float = 0.02  # 2%
    max_weekly_drawdown: float = 0.05  # 5%
    max_monthly_drawdown: float = 0.10  # 10%
    max_absolute_drawdown: float = 0.15  # 15%
    
    # Kill switches
    daily_loss_kill_switch: float = 0.03  # 3% perda diária = kill switch
    consecutive_losses_kill_switch: int = 5  # 5 perdas consecutivas
    rapid_drawdown_threshold: float = 0.05  # 5% em 1 hora
    
    # Estados
    current_state: TradingState = field(default=TradingState.NORMAL)
    kill_switch_activated: bool = field(default=False)
    kill_switch_timestamp: Optional[datetime] = field(default=None)
    kill_switch_reason: str = field(default="")
    
    # Cooldown
    kill_switch_cooldown_hours: int = 24
    state_transition_cooldown_minutes: int = 30


class AdvancedRiskManager:
    """Sistema avançado de gestão de risco com Kelly Criterion e Kill Switch"""
    
    def __init__(self, config):
        self.config = config
        self.current_balance = 10000.0
        self.initial_balance = 10000.0
        self.daily_starting_balance = 10000.0
        self.weekly_starting_balance = 10000.0
        self.monthly_starting_balance = 10000.0
        
        # Kelly Criterion
        self.kelly_params = KellyParameters()
        
        # Proteção de Drawdown
        self.drawdown_protection = DrawdownProtection(
            max_daily_drawdown=getattr(config, 'max_daily_drawdown', 0.02),
            max_weekly_drawdown=getattr(config, 'max_weekly_drawdown', 0.05),
            max_monthly_drawdown=getattr(config, 'max_monthly_drawdown', 0.10),
            daily_loss_kill_switch=getattr(config, 'daily_loss_kill_switch', 0.03),
            consecutive_losses_kill_switch=getattr(config, 'max_consecutive_losses', 5)
        )
        
        # Históricos
        self.trade_history = deque(maxlen=1000)
        self.balance_history = deque(maxlen=10000)
        self.pnl_history = deque(maxlen=1000)
        self.kelly_history = deque(maxlen=100)
        self.state_history = deque(maxlen=100)
        
        # Métricas em tempo real
        self.position_info = None
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.consecutive_losses = 0
        self.peak_balance = 10000.0
        self.total_fees_paid = 0.0
        
        # Controle de tempo
        self.last_daily_reset = datetime.now()
        self.last_weekly_reset = datetime.now()
        self.last_monthly_reset = datetime.now()
        self.last_state_change = datetime.now()
        
        # Risk scoring
        self.risk_score = 0
        self.risk_factors = {}
        
        # Performance tracking para Kelly
        self.winning_trades = deque(maxlen=100)
        self.losing_trades = deque(maxlen=100)
        
        logger.info(f"💰 Advanced Risk Manager inicializado - Balance: ${self.current_balance:,.2f}")
        logger.info(f"🛡️ Proteções: Daily DD {self.drawdown_protection.max_daily_drawdown*100:.1f}%, "
                   f"Kill Switch {self.drawdown_protection.daily_loss_kill_switch*100:.1f}%")
    
    def calculate_position_size(
        self,
        confidence: float,
        volatility: float,
        current_price: Optional[float] = None,
        market_conditions: Optional[Dict] = None
    ) -> float:
        """
        Cálculo avançado de tamanho de posição usando Kelly Criterion
        """
        # Verificar kill switch primeiro
        if self.drawdown_protection.kill_switch_activated:
            logger.warning("🚨 Kill switch ativo - trading bloqueado")
            return 0.0
        
        # Reset periódico
        self._check_periodic_resets()
        
        # Verificar estado do sistema
        current_state = self._evaluate_trading_state()
        if current_state == TradingState.KILL_SWITCH:
            return 0.0
        
        # Atualizar Kelly Criterion
        self._update_kelly_parameters()
        kelly_fraction = self.kelly_params.calculate()
        
        # Base position size com Kelly
        base_position_pct = kelly_fraction
        
        # Ajustes por estado do sistema
        state_multiplier = self._get_state_multiplier(current_state)
        base_position_pct *= state_multiplier
        
        # Ajuste por confiança (não-linear)
        confidence_multiplier = self._calculate_confidence_multiplier(confidence)
        position_pct = base_position_pct * confidence_multiplier
        
        # Ajuste por volatilidade
        volatility_multiplier = self._calculate_volatility_multiplier(volatility)
        position_pct *= volatility_multiplier
        
        # Ajuste por condições de mercado
        if market_conditions:
            market_multiplier = self._calculate_market_multiplier(market_conditions)
            position_pct *= market_multiplier
        
        # Ajuste por drawdown atual
        drawdown_multiplier = self._calculate_drawdown_multiplier()
        position_pct *= drawdown_multiplier
        
        # Aplicar limites absolutos
        position_pct = self._apply_position_limits(position_pct, current_state)
        
        # Calcular valor da posição
        position_value = self.current_balance * position_pct
        
        # Verificar limites mínimos
        min_position = getattr(self.config, 'min_position_value', 50.0)
        if position_value < min_position:
            return 0.0
        
        # Registrar no histórico
        self.kelly_history.append({
            'timestamp': datetime.now(),
            'kelly_fraction': kelly_fraction,
            'final_position_pct': position_pct,
            'confidence': confidence,
            'volatility': volatility,
            'state': current_state.value
        })
        
        # Log detalhado
        if current_price and position_value > 0:
            quantity = position_value / current_price
            logger.info(f"""
📊 Posição calculada (Kelly Criterion):
- Kelly: {kelly_fraction*100:.2f}% (Win rate: {self.kelly_params.win_rate*100:.1f}%, W/L: {self.kelly_params.win_loss_ratio:.2f})
- Estado: {current_state.value} (mult: {state_multiplier:.2f})
- Posição final: ${position_value:.2f} ({position_pct*100:.2f}%)
- Quantidade: {quantity:.6f} @ ${current_price:.2f}
- Risk Score: {self.risk_score}/100
            """)
        
        return position_value
    
    def _update_kelly_parameters(self):
        """Atualiza parâmetros do Kelly Criterion baseado no histórico"""
        if len(self.winning_trades) < 5 or len(self.losing_trades) < 3:
            # Poucos dados, usar valores conservadores
            self.kelly_params.win_rate = 0.5
            self.kelly_params.win_loss_ratio = 1.5
            return
        
        # Calcular win rate
        total_trades = len(self.winning_trades) + len(self.losing_trades)
        self.kelly_params.win_rate = len(self.winning_trades) / total_trades
        
        # Calcular médias
        avg_win = np.mean(list(self.winning_trades)) if self.winning_trades else 0
        avg_loss = abs(np.mean(list(self.losing_trades))) if self.losing_trades else 1
        
        self.kelly_params.avg_win = avg_win
        self.kelly_params.avg_loss = avg_loss
        self.kelly_params.win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5
        
        # Ajustar multiplicador baseado em volatilidade dos resultados
        if len(self.trade_history) > 20:
            returns = [t['pnl_pct'] for t in list(self.trade_history)[-20:]]
            return_std = np.std(returns)
            
            # Maior volatilidade = menor multiplicador Kelly
            if return_std > 0.03:  # Alta volatilidade
                self.kelly_params.kelly_multiplier = 0.15
            elif return_std > 0.02:
                self.kelly_params.kelly_multiplier = 0.20
            else:
                self.kelly_params.kelly_multiplier = 0.25
    
    def _evaluate_trading_state(self) -> TradingState:
        """Avalia e atualiza o estado do sistema de trading"""
        # Calcular métricas atuais
        daily_drawdown = (self.daily_starting_balance - self.current_balance) / self.daily_starting_balance
        weekly_drawdown = (self.weekly_starting_balance - self.current_balance) / self.weekly_starting_balance
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        # Verificar kill switches
        if daily_drawdown > self.drawdown_protection.daily_loss_kill_switch:
            self._activate_kill_switch(f"Perda diária excedeu limite: {daily_drawdown*100:.1f}%")
            return TradingState.KILL_SWITCH
        
        if self.consecutive_losses >= self.drawdown_protection.consecutive_losses_kill_switch:
            self._activate_kill_switch(f"Perdas consecutivas: {self.consecutive_losses}")
            return TradingState.KILL_SWITCH
        
        # Verificar drawdown rápido (1 hora)
        if self._check_rapid_drawdown():
            self._activate_kill_switch("Drawdown rápido detectado")
            return TradingState.KILL_SWITCH
        
        # Determinar estado baseado em drawdown
        if current_drawdown > self.drawdown_protection.max_absolute_drawdown:
            return TradingState.EMERGENCY_STOP
        elif daily_drawdown > self.drawdown_protection.max_daily_drawdown * 0.8:
            return TradingState.RESTRICTED
        elif weekly_drawdown > self.drawdown_protection.max_weekly_drawdown * 0.7:
            return TradingState.CAUTION
        else:
            return TradingState.NORMAL
    
    def _activate_kill_switch(self, reason: str):
        """Ativa o kill switch do sistema"""
        if not self.drawdown_protection.kill_switch_activated:
            self.drawdown_protection.kill_switch_activated = True
            self.drawdown_protection.kill_switch_timestamp = datetime.now()
            self.drawdown_protection.kill_switch_reason = reason
            self.drawdown_protection.current_state = TradingState.KILL_SWITCH
            
            logger.critical(f"""
🚨🚨🚨 KILL SWITCH ATIVADO 🚨🚨🚨
Razão: {reason}
Balance: ${self.current_balance:.2f}
P&L Diário: ${self.daily_pnl:.2f}
Cooldown: {self.drawdown_protection.kill_switch_cooldown_hours}h
            """)
            
            # Registrar no histórico
            self.state_history.append({
                'timestamp': datetime.now(),
                'state': TradingState.KILL_SWITCH,
                'reason': reason,
                'balance': self.current_balance,
                'daily_pnl': self.daily_pnl
            })
    
    def _check_rapid_drawdown(self) -> bool:
        """Verifica se houve drawdown rápido (1 hora)"""
        if len(self.balance_history) < 60:  # Menos de 1 hora de dados
            return False
        
        balance_1h_ago = list(self.balance_history)[-60]
        rapid_drawdown = (balance_1h_ago - self.current_balance) / balance_1h_ago
        
        return rapid_drawdown > self.drawdown_protection.rapid_drawdown_threshold
    
    def _get_state_multiplier(self, state: TradingState) -> float:
        """Retorna multiplicador baseado no estado do sistema"""
        multipliers = {
            TradingState.NORMAL: 1.0,
            TradingState.CAUTION: 0.7,
            TradingState.RESTRICTED: 0.3,
            TradingState.KILL_SWITCH: 0.0,
            TradingState.EMERGENCY_STOP: 0.0
        }
        return multipliers.get(state, 0.5)
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Multiplicador não-linear baseado em confiança"""
        if confidence < 0.5:
            return 0.3
        elif confidence < 0.6:
            return 0.5
        elif confidence < 0.7:
            return 0.7
        elif confidence < 0.8:
            return 0.9
        elif confidence < 0.9:
            return 1.1
        else:
            return 1.2
    
    def _calculate_volatility_multiplier(self, volatility: float) -> float:
        """Ajusta por volatilidade usando limites dinâmicos"""
        # Volatilidade adaptativa baseada em histórico
        if hasattr(self, 'volatility_history') and len(self.volatility_history) > 20:
            avg_vol = np.mean(list(self.volatility_history)[-20:])
            vol_std = np.std(list(self.volatility_history)[-20:])
            
            # Z-score da volatilidade atual
            z_score = (volatility - avg_vol) / vol_std if vol_std > 0 else 0
            
            if z_score > 2:  # Muito acima da média
                return 0.3
            elif z_score > 1:
                return 0.6
            elif z_score < -1:  # Muito abaixo da média
                return 1.3
            else:
                return 1.0
        else:
            # Fallback para limites fixos
            if volatility > 0.04:
                return 0.3
            elif volatility > 0.03:
                return 0.5
            elif volatility > 0.02:
                return 0.7
            elif volatility > 0.01:
                return 1.0
            else:
                return 1.2
    
    def _calculate_market_multiplier(self, market_conditions: Dict) -> float:
        """Ajusta baseado em condições de mercado"""
        score = market_conditions.get('score', 100)
        
        if score >= 90:
            return 1.1
        elif score >= 80:
            return 1.0
        elif score >= 70:
            return 0.8
        elif score >= 60:
            return 0.6
        else:
            return 0.3
    
    def _calculate_drawdown_multiplier(self) -> float:
        """Reduz exposição progressivamente com drawdown"""
        current_dd = (self.peak_balance - self.current_balance) / self.peak_balance
        
        if current_dd < 0.02:
            return 1.0
        elif current_dd < 0.05:
            return 0.8
        elif current_dd < 0.08:
            return 0.6
        elif current_dd < 0.10:
            return 0.4
        else:
            return 0.2
    
    def _apply_position_limits(self, position_pct: float, state: TradingState) -> float:
        """Aplica limites baseados no estado"""
        # Limites por estado
        state_limits = {
            TradingState.NORMAL: 0.10,  # 10% máximo
            TradingState.CAUTION: 0.05,  # 5% máximo
            TradingState.RESTRICTED: 0.02,  # 2% máximo
            TradingState.KILL_SWITCH: 0.0,
            TradingState.EMERGENCY_STOP: 0.0
        }
        
        max_allowed = state_limits.get(state, 0.05)
        return min(position_pct, max_allowed)
    
    def should_close_position(
        self,
        current_price: float,
        entry_price: float,
        side: str = 'BUY'
    ) -> Tuple[bool, str]:
        """
        Sistema avançado de stops com proteções adicionais
        """
        if self.position_info is None:
            return False, ""
        
        # Kill switch fecha todas as posições
        if self.drawdown_protection.kill_switch_activated:
            return True, "KILL SWITCH ATIVO"
        
        # Estado de emergência fecha posições perdedoras
        current_state = self._evaluate_trading_state()
        if current_state == TradingState.EMERGENCY_STOP:
            pnl_pct = self._calculate_pnl_pct(current_price, entry_price, side)
            if pnl_pct < 0:
                return True, "EMERGENCY STOP - Fechando posições perdedoras"
        
        # Sistema de stops padrão (melhorado)
        return self._evaluate_standard_stops(current_price, entry_price, side)
    
    def _evaluate_standard_stops(
        self,
        current_price: float,
        entry_price: float,
        side: str
    ) -> Tuple[bool, str]:
        """Avalia stops padrão com melhorias"""
        # Calcular P&L
        pnl_pct = self._calculate_pnl_pct(current_price, entry_price, side)
        
        # Atualizar máximos
        if pnl_pct > self.position_info.get('highest_pnl', 0):
            self.position_info['highest_pnl'] = pnl_pct
        
        # Stop loss dinâmico baseado em volatilidade e Kelly
        volatility = self.position_info.get('volatility', 0.01)
        kelly_fraction = self.kelly_params.kelly_fraction
        
        # Stop mais apertado quando Kelly baixo (menos edge)
        stop_multiplier = 2.0 - (kelly_fraction / self.kelly_params.max_kelly)
        base_stop = getattr(self.config, 'stop_loss_pct', 0.015)
        dynamic_stop = base_stop * stop_multiplier * (1 + volatility * 10)
        
        if pnl_pct < -dynamic_stop:
            return True, f"Stop Loss Dinâmico ({dynamic_stop*100:.1f}%)"
        
        # Take profit dinâmico baseado em Kelly
        base_tp = getattr(self.config, 'take_profit_pct', 0.025)
        
        # TP maior quando Kelly alto (mais edge)
        tp_multiplier = 0.5 + (kelly_fraction / self.kelly_params.max_kelly) * 1.5
        dynamic_tp = base_tp * tp_multiplier * (1 + volatility * 5)
        
        if pnl_pct > dynamic_tp:
            return True, f"Take Profit Dinâmico ({dynamic_tp*100:.1f}%)"
        
        # Trailing stop progressivo
        if pnl_pct > 0.01:  # Ativação em 1%
            highest_pnl = self.position_info.get('highest_pnl', pnl_pct)
            
            # Trailing mais agressivo baseado no lucro
            if highest_pnl > 0.05:
                keep_ratio = 0.85
            elif highest_pnl > 0.03:
                keep_ratio = 0.75
            elif highest_pnl > 0.02:
                keep_ratio = 0.65
            else:
                keep_ratio = 0.50
            
            trailing_level = highest_pnl * keep_ratio
            
            if pnl_pct < trailing_level:
                return True, f"Trailing Stop ({trailing_level*100:.1f}%)"
        
        # Time-based stops
        position_duration = time.time() - self.position_info.get('entry_timestamp', time.time())
        
        # Breakeven mais agressivo em estados restritivos
        current_state = self._evaluate_trading_state()
        if current_state in [TradingState.CAUTION, TradingState.RESTRICTED]:
            if position_duration > 300 and pnl_pct < 0.002:  # 5 min
                return True, "Breakeven (Estado Restritivo)"
        
        # Stop por tempo máximo
        max_duration = getattr(self.config, 'max_position_duration', 14400)  # 4h
        if position_duration > max_duration:
            return True, f"Tempo máximo ({max_duration/3600:.1f}h)"
        
        return False, ""
    
    def _calculate_pnl_pct(self, current_price: float, entry_price: float, side: str) -> float:
        """Calcula P&L percentual"""
        if side == 'BUY':
            return (current_price - entry_price) / entry_price
        else:
            return (entry_price - current_price) / entry_price
    
    def update_pnl(self, pnl: float, fees: float = 0):
        """Atualiza P&L e métricas com tracking avançado"""
        # Atualizar balanços
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
        self.current_balance += pnl
        self.total_fees_paid += fees
        
        # Atualizar históricos
        self.balance_history.append(self.current_balance)
        self.pnl_history.append(pnl)
        
        # Trade tracking
        if self.position_info:
            pnl_pct = pnl / (self.position_info.get('position_value', 1))
            
            trade_record = {
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'fees': fees,
                'balance': self.current_balance,
                'timestamp': datetime.now(),
                'state': self.drawdown_protection.current_state.value
            }
            self.trade_history.append(trade_record)
            
            # Atualizar win/loss tracking
            if pnl > 0:
                self.winning_trades.append(pnl_pct)
                self.consecutive_losses = 0
            else:
                self.losing_trades.append(pnl_pct)
                self.consecutive_losses += 1
        
        # Atualizar peak e drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Verificar estado após atualização
        new_state = self._evaluate_trading_state()
        if new_state != self.drawdown_protection.current_state:
            old_state = self.drawdown_protection.current_state
            self.drawdown_protection.current_state = new_state
            logger.warning(f"🔄 Mudança de estado: {old_state.value} → {new_state.value}")
    
    def _check_periodic_resets(self):
        """Reseta métricas periodicamente"""
        now = datetime.now()
        
        # Reset diário
        if now.date() > self.last_daily_reset.date():
            logger.info(f"📅 Reset diário - P&L anterior: ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.daily_starting_balance = self.current_balance
            self.last_daily_reset = now
            self.consecutive_losses = 0
            
            # Verificar se pode sair do kill switch
            if self.drawdown_protection.kill_switch_activated:
                hours_since = (now - self.drawdown_protection.kill_switch_timestamp).total_seconds() / 3600
                if hours_since >= self.drawdown_protection.kill_switch_cooldown_hours:
                    self.drawdown_protection.kill_switch_activated = False
                    self.drawdown_protection.current_state = TradingState.CAUTION
                    logger.warning("✅ Kill switch desativado após cooldown")
        
        # Reset semanal
        if now - self.last_weekly_reset > timedelta(days=7):
            logger.info(f"📅 Reset semanal - P&L: ${self.weekly_pnl:.2f}")
            self.weekly_pnl = 0.0
            self.weekly_starting_balance = self.current_balance
            self.last_weekly_reset = now
        
        # Reset mensal
        if now.month != self.last_monthly_reset.month:
            logger.info(f"📅 Reset mensal - P&L: ${self.monthly_pnl:.2f}")
            self.monthly_pnl = 0.0
            self.monthly_starting_balance = self.current_balance
            self.last_monthly_reset = now
    
    def get_risk_metrics(self) -> Dict:
        """Retorna métricas detalhadas de risco"""
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        daily_drawdown = (self.daily_starting_balance - self.current_balance) / self.daily_starting_balance
        
        # Calcular risk score
        self._calculate_risk_score()
        
        return {
            'current_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'monthly_pnl': self.monthly_pnl,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance,
            'max_drawdown': max(self.max_drawdown, current_drawdown),
            'current_drawdown': current_drawdown,
            'daily_drawdown': daily_drawdown,
            'total_fees': self.total_fees_paid,
            'consecutive_losses': self.consecutive_losses,
            'trading_state': self.drawdown_protection.current_state.value,
            'kill_switch_active': self.drawdown_protection.kill_switch_activated,
            'can_trade': self._can_trade(),
            'risk_score': self.risk_score,
            'risk_factors': self.risk_factors,
            'kelly_fraction': self.kelly_params.kelly_fraction,
            'win_rate': self.kelly_params.win_rate,
            'win_loss_ratio': self.kelly_params.win_loss_ratio,
            'position_sizing': {
                'kelly': self.kelly_params.kelly_fraction,
                'state_multiplier': self._get_state_multiplier(self.drawdown_protection.current_state),
                'max_position_pct': self._apply_position_limits(1.0, self.drawdown_protection.current_state)
            }
        }
    
    def _calculate_risk_score(self):
        """Calcula score de risco detalhado (0-100, onde 100 é máximo risco)"""
        score = 0
        factors = {}
        
        # Drawdown (0-30 pontos)
        current_dd = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_dd > 0.10:
            dd_score = 30
        elif current_dd > 0.05:
            dd_score = 20
        elif current_dd > 0.02:
            dd_score = 10
        else:
            dd_score = 0
        score += dd_score
        factors['drawdown'] = dd_score
        
        # Perdas consecutivas (0-20 pontos)
        if self.consecutive_losses >= 5:
            loss_score = 20
        elif self.consecutive_losses >= 3:
            loss_score = 10
        elif self.consecutive_losses >= 2:
            loss_score = 5
        else:
            loss_score = 0
        score += loss_score
        factors['consecutive_losses'] = loss_score
        
        # Daily loss (0-25 pontos)
        daily_loss_pct = abs(self.daily_pnl / self.daily_starting_balance) if self.daily_starting_balance > 0 else 0
        if daily_loss_pct > 0.02:
            daily_score = 25
        elif daily_loss_pct > 0.015:
            daily_score = 15
        elif daily_loss_pct > 0.01:
            daily_score = 10
        else:
            daily_score = 0
        score += daily_score
        factors['daily_loss'] = daily_score
        
        # Win rate baixo (0-15 pontos)
        if self.kelly_params.win_rate < 0.3:
            wr_score = 15
        elif self.kelly_params.win_rate < 0.4:
            wr_score = 10
        elif self.kelly_params.win_rate < 0.45:
            wr_score = 5
        else:
            wr_score = 0
        score += wr_score
        factors['low_win_rate'] = wr_score
        
        # Estado do sistema (0-10 pontos)
        state_scores = {
            TradingState.NORMAL: 0,
            TradingState.CAUTION: 3,
            TradingState.RESTRICTED: 6,
            TradingState.KILL_SWITCH: 10,
            TradingState.EMERGENCY_STOP: 10
        }
        state_score = state_scores.get(self.drawdown_protection.current_state, 5)
        score += state_score
        factors['system_state'] = state_score
        
        self.risk_score = min(score, 100)
        self.risk_factors = factors
    
    def _can_trade(self) -> bool:
        """Verifica se pode abrir novas posições"""
        if self.drawdown_protection.kill_switch_activated:
            return False
        
        if self.drawdown_protection.current_state in [TradingState.KILL_SWITCH, TradingState.EMERGENCY_STOP]:
            return False
        
        if self.current_balance < 100:
            return False
        
        return True
    
    def set_position_info(self, position: Dict):
        """Define informações da posição com tracking avançado"""
        if position:
            position.update({
                'entry_timestamp': time.time(),
                'highest_pnl': 0,
                'lowest_pnl': 0,
                'position_value': position.get('size', 0) * position.get('entry_price', 0),
                'entry_balance': self.current_balance,
                'entry_state': self.drawdown_protection.current_state.value,
                'kelly_fraction': self.kelly_params.kelly_fraction
            })
        
        self.position_info = position
    
    def emergency_close_all_positions(self) -> List[str]:
        """Fecha todas as posições em emergência"""
        logger.critical("🚨 FECHAMENTO DE EMERGÊNCIA DE TODAS AS POSIÇÕES")
        
        positions_to_close = []
        if self.position_info:
            positions_to_close.append(self.position_info.get('symbol', 'UNKNOWN'))
        
        # Ativar kill switch
        self._activate_kill_switch("Fechamento de emergência executado")
        
        return positions_to_close


# Aliases para compatibilidade
class UltraFastRiskManager(AdvancedRiskManager):
    """Mantém compatibilidade com código existente"""
    
    def __init__(self, config):
        super().__init__(config)
        # Configurações específicas para compatibilidade
        if not hasattr(config, 'stop_loss_base'):
            config.stop_loss_base = getattr(config, 'stop_loss_pct', 0.015)
        if not hasattr(config, 'take_profit_base'):
            config.take_profit_base = getattr(config, 'take_profit_pct', 0.025)
        if not hasattr(config, 'enable_trailing'):
            config.enable_trailing = True
        if not hasattr(config, 'trailing_activation'):
            config.trailing_activation = 0.01
        if not hasattr(config, 'breakeven_time'):
            config.breakeven_time = 600  # 10 minutos
        if not hasattr(config, 'max_position_duration'):
            config.max_position_duration = 14400  # 4 horas
        
        self.max_drawdown = getattr(config, 'max_drawdown', 0.15)


class MarketConditionValidator:
    """Valida condições de mercado com score detalhado"""
    
    def __init__(self, config):
        self.config = config
        self.last_validation = time.time()
        self.validation_interval = 60
        self.market_score = 100
        self.unsafe_reasons = []
        
        # Históricos
        self.volatility_history = deque(maxlen=100)
        self.spread_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.score_history = deque(maxlen=50)
        
        logger.info("🛡️ Market Validator inicializado")
    
    async def validate_market_conditions(
        self,
        market_data: Dict,
        client = None
    ) -> Tuple[bool, List[str]]:
        """
        Validação completa de condições de mercado
        """
        reasons = []
        score = 100
        
        # Debug mode sempre seguro
        if self.config.debug_mode:
            return True, []
        
        # 1. Volatilidade
        volatility = self._check_volatility(market_data)
        if volatility is not None:
            self.volatility_history.append(volatility)
            vol_score, vol_reason = self._score_volatility(volatility)
            score -= vol_score
            if vol_reason:
                reasons.append(vol_reason)
        
        # 2. Spread
        spread_bps = self._check_spread(market_data)
        if spread_bps is not None:
            self.spread_history.append(spread_bps)
            spread_score, spread_reason = self._score_spread(spread_bps)
            score -= spread_score
            if spread_reason:
                reasons.append(spread_reason)
        
        # 3. Liquidez
        liquidity_score, liquidity_reason = self._check_liquidity(market_data)
        score -= liquidity_score
        if liquidity_reason:
            reasons.append(liquidity_reason)
        
        # 4. Padrões anormais
        anomaly_score, anomaly_reason = self._check_anomalies(market_data)
        score -= anomaly_score
        if anomaly_reason:
            reasons.append(anomaly_reason)
        
        # 5. Horário
        time_score, time_reason = self._check_trading_time()
        score -= time_score
        if time_reason:
            reasons.append(time_reason)
        
        # Score final
        self.market_score = max(0, score)
        self.score_history.append(self.market_score)
        self.unsafe_reasons = reasons
        
        # Decisão baseada em score configurável
        min_score = getattr(self.config, 'min_market_score', 60)
        is_safe = self.market_score >= min_score
        
        return is_safe, reasons
    
    def _check_volatility(self, market_data: Dict) -> Optional[float]:
        """Calcula volatilidade atual"""
        if 'prices' not in market_data or len(market_data['prices']) < 100:
            return None
        
        prices = market_data['prices'][-100:]
        return np.std(prices) / np.mean(prices)
    
    def _score_volatility(self, volatility: float) -> Tuple[int, str]:
        """Pontua volatilidade"""
        if volatility > self.config.max_volatility:
            return 30, f"Volatilidade extrema: {volatility*100:.2f}%"
        elif volatility > self.config.max_volatility * 0.8:
            return 15, f"Volatilidade alta: {volatility*100:.2f}%"
        elif volatility > self.config.max_volatility * 0.6:
            return 5, None
        return 0, None
    
    def _check_spread(self, market_data: Dict) -> Optional[float]:
        """Calcula spread"""
        if 'orderbook_asks' not in market_data or 'orderbook_bids' not in market_data:
            return None
        
        asks = market_data['orderbook_asks']
        bids = market_data['orderbook_bids']
        
        if len(asks) > 0 and len(bids) > 0:
            if asks[0, 0] > 0 and bids[0, 0] > 0:
                spread = asks[0, 0] - bids[0, 0]
                return (spread / bids[0, 0]) * 10000
        
        return None
    
    def _score_spread(self, spread_bps: float) -> Tuple[int, str]:
        """Pontua spread"""
        if spread_bps > self.config.max_spread_bps:
            return 25, f"Spread alto: {spread_bps:.1f} bps"
        elif spread_bps > self.config.max_spread_bps * 0.8:
            return 10, f"Spread elevado: {spread_bps:.1f} bps"
        return 0, None
    
    def _check_liquidity(self, market_data: Dict) -> Tuple[int, str]:
        """Verifica liquidez"""
        if 'orderbook_bids' not in market_data or 'orderbook_asks' not in market_data:
            return 0, None
        
        bids = market_data['orderbook_bids']
        asks = market_data['orderbook_asks']
        
        # Profundidade
        if len(bids) < 10 or len(asks) < 10:
            return 20, "Orderbook raso"
        
        # Volume nos primeiros níveis
        bid_volume = np.sum(bids[:5, 1]) * bids[0, 0] if len(bids) >= 5 else 0
        ask_volume = np.sum(asks[:5, 1]) * asks[0, 0] if len(asks) >= 5 else 0
        
        min_liquidity = getattr(self.config, 'min_liquidity_depth', 50000)
        
        if bid_volume < min_liquidity or ask_volume < min_liquidity:
            return 15, f"Baixa liquidez: ${min(bid_volume, ask_volume):,.0f}"
        
        return 0, None
    
    def _check_anomalies(self, market_data: Dict) -> Tuple[int, str]:
        """Detecta anomalias"""
        if 'prices' not in market_data or len(market_data['prices']) < 50:
            return 0, None
        
        prices = market_data['prices']
        
        # Flash crash/pump
        recent_change = (prices[-1] - prices[-10]) / prices[-10]
        if abs(recent_change) > 0.03:  # 3% em 10 períodos
            return 50, f"Movimento anormal: {recent_change*100:+.1f}%"
        
        # Gap detection
        if len(prices) > 1:
            gap = abs(prices[-1] - prices[-2]) / prices[-2]
            if gap > 0.01:  # Gap > 1%
                return 20, f"Gap detectado: {gap*100:.1f}%"
        
        return 0, None
    
    def _check_trading_time(self) -> Tuple[int, str]:
        """Verifica horário"""
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # Fins de semana
        if current_day in [5, 6]:  # Sábado, Domingo
            return 5, "Final de semana"
        
        # Horários de baixa liquidez (UTC)
        if 2 <= current_hour <= 6:
            return 10, "Horário de baixa liquidez"
        
        return 0, None
    
    def get_market_health(self) -> Dict:
        """Retorna saúde detalhada do mercado"""
        avg_volatility = np.mean(list(self.volatility_history)[-20:]) if self.volatility_history else 0
        avg_spread = np.mean(list(self.spread_history)[-20:]) if self.spread_history else 0
        score_trend = "IMPROVING" if len(self.score_history) > 10 and self.score_history[-1] > self.score_history[-10] else "DECLINING"
        
        return {
            'score': self.market_score,
            'status': self._get_market_status(),
            'is_safe': self.market_score >= 60,
            'reasons': self.unsafe_reasons,
            'metrics': {
                'avg_volatility': avg_volatility,
                'avg_spread': avg_spread,
                'score_trend': score_trend
            },
            'history': {
                'scores': list(self.score_history)[-10:],
                'volatility': list(self.volatility_history)[-10:],
                'spread': list(self.spread_history)[-10:]
            }
        }
    
    def _get_market_status(self) -> str:
        """Status textual do mercado"""
        if self.market_score >= 90:
            return "EXCELENTE"
        elif self.market_score >= 80:
            return "MUITO BOM"
        elif self.market_score >= 70:
            return "BOM"
        elif self.market_score >= 60:
            return "REGULAR"
        elif self.market_score >= 40:
            return "RUIM"
        else:
            return "CRÍTICO"