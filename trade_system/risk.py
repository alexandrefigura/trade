"""
Gestão de risco ultra-rápida com stops dinâmicos e proteções avançadas
"""
import time
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from collections import deque
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class UltraFastRiskManager:
    """Gestão de risco com cálculos otimizados e stops dinâmicos"""
    
    def __init__(self, config):
        self.config = config
        self.current_balance = 10000.0
        self.initial_balance = 10000.0
        self.daily_pnl = 0.0
        self.position_info = None
        
        # Histórico
        self.pnl_history = np.zeros(1000, dtype=np.float32)
        self.pnl_index = 0
        self.trade_history = deque(maxlen=100)
        
        # Métricas
        self.total_fees_paid = 0.0
        self.peak_balance = 10000.0
        self.drawdown_start = None
        self.max_drawdown = 0.0
        self.daily_trades = 0
        self.last_reset_day = datetime.now().date()
        
        # Consecutive losses tracking
        self.consecutive_losses = 0
        self.max_consecutive_losses = getattr(config, 'max_consecutive_losses', 3)
        
        # Position management
        self.max_position_value = None
        self.max_positions = 1
        self.current_positions = 0
        
        # Dynamic risk parameters
        self.base_risk_pct = config.max_position_pct
        self.current_risk_pct = self.base_risk_pct
        
        logger.info(f"💰 Risk Manager inicializado - Balance: ${self.current_balance:,.2f}")
    
    def calculate_position_size(
        self,
        confidence: float,
        volatility: float,
        current_price: Optional[float] = None
    ) -> float:
        """
        Cálculo dinâmico do tamanho da posição com Kelly Criterion
        """
        # Reset diário
        self._check_daily_reset()
        
        # Verificar limites
        if not self._check_risk_limits():
            return 0.0
        
        # Ajustar risco por performance
        self._adjust_risk_by_performance()
        
        # Kelly Criterion modificado
        win_rate = self._calculate_win_rate()
        avg_win_loss_ratio = self._calculate_win_loss_ratio()
        
        if win_rate > 0 and avg_win_loss_ratio > 0:
            kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap em 25%
        else:
            kelly_fraction = 0.1  # Default 10%
        
        # Ajustar por confiança
        confidence_factor = self._calculate_confidence_factor(confidence)
        position_pct = kelly_fraction * confidence_factor * self.current_risk_pct
        
        # Ajustar por volatilidade
        volatility_factor = self._calculate_volatility_factor(volatility)
        position_pct *= volatility_factor
        
        # Ajustar por drawdown
        drawdown_factor = self._calculate_drawdown_factor()
        position_pct *= drawdown_factor
        
        # Calcular valor
        position_value = self.current_balance * position_pct
        
        # Aplicar limites
        position_value = self._apply_position_limits(position_value)
        
        # Log
        if current_price and position_value > 0:
            quantity = position_value / current_price
            logger.info(f"""
📊 Posição calculada:
- Valor: ${position_value:.2f} ({position_pct*100:.1f}%)
- Quantidade: {quantity:.6f} @ ${current_price:.2f}
- Win rate: {win_rate*100:.1f}%
- Kelly: {kelly_fraction*100:.1f}%
- Volatilidade: {volatility*100:.2f}%
            """)
        
        return position_value
    
    def _calculate_win_rate(self) -> float:
        """Calcula taxa de acerto histórica"""
        if len(self.trade_history) < 5:
            return 0.5  # Default 50%
        
        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        return wins / len(self.trade_history)
    
    def _calculate_win_loss_ratio(self) -> float:
        """Calcula ratio médio ganho/perda"""
        if len(self.trade_history) < 5:
            return 1.5  # Default 1.5:1
        
        wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]
        
        if wins and losses:
            return np.mean(wins) / np.mean(losses)
        return 1.5
    
    def _calculate_confidence_factor(self, confidence: float) -> float:
        """Ajusta fator baseado na confiança"""
        # Escala não-linear: penaliza confiança baixa
        if confidence < 0.6:
            return confidence * 0.5
        elif confidence < 0.7:
            return confidence * 0.8
        elif confidence < 0.8:
            return confidence
        else:
            return confidence * 1.1  # Boost para alta confiança
    
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """Fator de ajuste por volatilidade"""
        if volatility < 0.005:  # Muito baixa
            return 1.3
        elif volatility < 0.01:  # Baixa
            return 1.2
        elif volatility < 0.02:  # Normal
            return 1.0
        elif volatility < 0.03:  # Alta
            return 0.7
        elif volatility < 0.04:  # Muito alta
            return 0.5
        else:  # Extrema
            return 0.3
    
    def _calculate_drawdown_factor(self) -> float:
        """Reduz tamanho em drawdown"""
        current_dd = self._get_current_drawdown()
        
        if current_dd < 0.02:  # < 2%
            return 1.0
        elif current_dd < 0.05:  # < 5%
            return 0.8
        elif current_dd < 0.08:  # < 8%
            return 0.6
        else:  # > 8%
            return 0.4
    
    def _adjust_risk_by_performance(self):
        """Ajusta risco baseado em performance recente"""
        if len(self.trade_history) < 10:
            return
        
        recent_trades = list(self.trade_history)[-10:]
        recent_pnl = sum(t['pnl'] for t in recent_trades)
        
        if recent_pnl > 0:
            # Performance positiva: aumentar risco gradualmente
            self.current_risk_pct = min(
                self.base_risk_pct * 1.2,
                self.current_risk_pct * 1.05
            )
        else:
            # Performance negativa: reduzir risco
            self.current_risk_pct = max(
                self.base_risk_pct * 0.5,
                self.current_risk_pct * 0.95
            )
    
    def should_close_position(
        self,
        current_price: float,
        entry_price: float,
        side: str = 'BUY'
    ) -> Tuple[bool, str]:
        """
        Sistema avançado de stops com múltiplas condições
        """
        if self.position_info is None:
            return False, ""
        
        # Converter timestamps
        entry_time = self.position_info.get('entry_timestamp', time.time())
        if isinstance(entry_time, datetime):
            entry_time = entry_time.timestamp()
        
        # Calcular P&L
        if side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Atualizar máximos
        if pnl_pct > self.position_info.get('highest_pnl', 0):
            self.position_info['highest_pnl'] = pnl_pct
        
        # 1. Stop Loss Dinâmico
        volatility = self.position_info.get('volatility', 0.01)
        stop_loss = self._calculate_dynamic_stop_loss(volatility)
        
        if pnl_pct < -stop_loss:
            return True, f"Stop Loss ({stop_loss*100:.1f}%)"
        
        # 2. Take Profit Dinâmico
        take_profit = self._calculate_dynamic_take_profit(volatility)
        
        if pnl_pct > take_profit:
            return True, f"Take Profit ({take_profit*100:.1f}%)"
        
        # 3. Trailing Stop Avançado
        if self.config.enable_trailing and pnl_pct > self.config.trailing_activation:
            trailing_stop = self._calculate_trailing_stop(pnl_pct)
            
            if pnl_pct < trailing_stop:
                return True, f"Trailing Stop ({trailing_stop*100:.1f}%)"
        
        # 4. Time-based stops
        position_duration = time.time() - entry_time
        
        # Breakeven após X tempo
        if position_duration > self.config.breakeven_time and pnl_pct < 0.001:
            return True, f"Breakeven ({self.config.breakeven_time/60:.0f}min)"
        
        # Stop por tempo máximo
        if position_duration > self.config.max_position_duration:
            return True, f"Time limit ({self.config.max_position_duration/3600:.1f}h)"
        
        # 5. Volatility spike stop
        current_volatility = self._estimate_current_volatility(current_price)
        if current_volatility > volatility * 2:
            return True, "Volatility spike"
        
        # 6. Drawdown protection
        if self._get_current_drawdown() > self.config.max_drawdown * 0.5:
            if pnl_pct < 0:
                return True, "Drawdown protection"
        
        return False, ""
    
    def _calculate_dynamic_stop_loss(self, volatility: float) -> float:
        """Stop loss baseado em volatilidade"""
        base_stop = self.config.stop_loss_base
        
        # Ajustar por volatilidade
        if volatility < 0.01:
            return base_stop * 0.8  # Stop mais apertado
        elif volatility < 0.02:
            return base_stop
        elif volatility < 0.03:
            return base_stop * 1.5
        else:
            return base_stop * 2.0  # Stop mais largo
    
    def _calculate_dynamic_take_profit(self, volatility: float) -> float:
        """Take profit baseado em volatilidade"""
        base_tp = self.config.take_profit_base
        
        # Ajustar por volatilidade
        if volatility < 0.01:
            return base_tp * 0.8
        elif volatility < 0.02:
            return base_tp
        elif volatility < 0.03:
            return base_tp * 1.5
        else:
            return base_tp * 2.0
    
    def _calculate_trailing_stop(self, current_pnl: float) -> float:
        """Trailing stop progressivo"""
        highest_pnl = self.position_info.get('highest_pnl', current_pnl)
        
        # Quanto maior o lucro, mais apertado o trailing
        if highest_pnl > 0.05:  # > 5%
            keep_ratio = 0.8  # Manter 80%
        elif highest_pnl > 0.03:  # > 3%
            keep_ratio = 0.7  # Manter 70%
        elif highest_pnl > 0.02:  # > 2%
            keep_ratio = 0.6  # Manter 60%
        else:
            keep_ratio = 0.5  # Manter 50%
        
        return highest_pnl * keep_ratio
    
    def _estimate_current_volatility(self, current_price: float) -> float:
        """Estima volatilidade atual"""
        # Implementação simplificada
        if hasattr(self, '_price_history'):
            recent_prices = list(self._price_history)[-20:]
            if len(recent_prices) > 5:
                return np.std(recent_prices) / np.mean(recent_prices)
        return 0.01  # Default
    
    def _check_risk_limits(self) -> bool:
        """Verifica todos os limites de risco"""
        # Stop loss diário
        if self.daily_pnl < -self.config.max_daily_loss * self.current_balance:
            logger.warning(f"🛑 Stop loss diário: ${self.daily_pnl:.2f}")
            return False
        
        # Perdas consecutivas
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"🛑 Máximo de perdas consecutivas: {self.consecutive_losses}")
            return False
        
        # Drawdown máximo
        if self._get_current_drawdown() > self.config.max_drawdown:
            logger.warning(f"🛑 Drawdown máximo: {self._get_current_drawdown()*100:.1f}%")
            return False
        
        # Balance mínimo
        if self.current_balance < 100:
            logger.warning(f"🛑 Balance insuficiente: ${self.current_balance:.2f}")
            return False
        
        return True
    
    def _get_current_drawdown(self) -> float:
        """Calcula drawdown atual"""
        if self.peak_balance > 0:
            return (self.peak_balance - self.current_balance) / self.peak_balance
        return 0.0
    
    def _apply_position_limits(self, position_value: float) -> float:
        """Aplica limites ao tamanho da posição"""
        # Mínimo
        min_position = 50.0
        if position_value < min_position:
            return 0.0
        
        # Máximo absoluto
        max_allowed = self.current_balance * 0.1  # 10% máximo
        position_value = min(position_value, max_allowed)
        
        # Máximo por volatilidade
        if hasattr(self, 'current_volatility'):
            if self.current_volatility > 0.03:
                position_value = min(position_value, self.current_balance * 0.05)
        
        return position_value
    
    def update_pnl(self, pnl: float, fees: float = 0):
        """Atualiza P&L e métricas"""
        self.daily_pnl += pnl
        self.current_balance += pnl
        self.total_fees_paid += fees
        
        # Histórico
        idx = self.pnl_index % 1000
        self.pnl_history[idx] = pnl
        self.pnl_index += 1
        
        # Trade history
        if self.position_info:
            trade_record = {
                'pnl': pnl,
                'fees': fees,
                'balance': self.current_balance,
                'timestamp': time.time()
            }
            self.trade_history.append(trade_record)
            
            # Consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        
        # Peak e drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.drawdown_start = None
        else:
            if self.drawdown_start is None:
                self.drawdown_start = datetime.now()
            current_drawdown = self._get_current_drawdown()
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _check_daily_reset(self):
        """Reset diário de métricas"""
        current_day = datetime.now().date()
        if current_day > self.last_reset_day:
            logger.info(f"📅 Novo dia - Reset de métricas")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_day = current_day
            self.consecutive_losses = 0  # Reset no novo dia
    
    def get_risk_metrics(self) -> Dict:
        """Retorna métricas detalhadas de risco"""
        current_drawdown = self._get_current_drawdown()
        win_rate = self._calculate_win_rate()
        
        return {
            'current_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': current_drawdown,
            'total_fees': self.total_fees_paid,
            'daily_trades': self.daily_trades,
            'can_trade': self._check_risk_limits(),
            'risk_level': self._calculate_risk_level(),
            'win_rate': win_rate,
            'consecutive_losses': self.consecutive_losses,
            'current_risk_pct': self.current_risk_pct,
            'trades_today': self.daily_trades
        }
    
    def _calculate_risk_level(self) -> str:
        """Calcula nível de risco atual"""
        score = 0
        
        # Drawdown
        dd = self._get_current_drawdown()
        if dd > 0.08:
            score += 3
        elif dd > 0.05:
            score += 2
        elif dd > 0.02:
            score += 1
        
        # Daily loss
        daily_loss_pct = abs(self.daily_pnl / self.current_balance) if self.current_balance > 0 else 0
        if daily_loss_pct > 0.015:
            score += 2
        elif daily_loss_pct > 0.01:
            score += 1
        
        # Consecutive losses
        if self.consecutive_losses >= 3:
            score += 2
        elif self.consecutive_losses >= 2:
            score += 1
        
        # Classificar
        if score >= 5:
            return "CRÍTICO"
        elif score >= 3:
            return "ALTO"
        elif score >= 1:
            return "MÉDIO"
        else:
            return "BAIXO"
    
    def set_position_info(self, position: Dict):
        """Define informações da posição"""
        if position:
            # Garantir timestamp
            if 'entry_timestamp' not in position:
                position['entry_timestamp'] = time.time()
            
            # Adicionar campos de controle
            position['highest_pnl'] = 0
            position['lowest_pnl'] = 0
            
            # Salvar volatilidade atual
            if 'volatility' not in position:
                position['volatility'] = 0.01
        
        self.position_info = position
        self.current_positions = 1 if position else 0
        self.daily_trades += 1
    
    def clear_position(self):
        """Limpa posição"""
        self.position_info = None
        self.current_positions = 0


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