"""
Gestão de risco ultra-rápida e validação de condições de mercado
"""
import time
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class UltraFastRiskManager:
    """Gestão de risco com cálculos otimizados"""

    def __init__(self, config):
        self.config = config
        self.initial_balance = config.initial_balance
        self.current_balance = self.initial_balance
        self.daily_pnl = 0.0
        self.max_daily_loss = config.max_daily_loss * self.initial_balance
        self.position_info: Optional[Dict] = None

        # Histórico circular de PnL
        self.pnl_history = np.zeros(self.config.pnl_history_size, dtype=np.float32)
        self.pnl_idx = 0

        # Métricas de drawdown
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0

        # Contadores
        self.daily_trades = 0
        self.last_reset_day = datetime.now().date()

        logger.info(f"💰 Risk Manager inicializado – Balance: ${self.current_balance:,.2f}")

    def calculate_position_size(
        self,
        confidence: float,
        volatility: float,
        current_price: Optional[float] = None
    ) -> float:
        """
        Retorna o valor da posição (USD) baseado em Kelly simplificado,
        ajustado por confiança e volatilidade, com checagem de limites.
        """
        # Reset diário se mudou o dia
        self._reset_daily_if_needed()

        # Verifica se ainda pode abrir posições
        if not self._within_risk_limits():
            return 0.0

        # Critério de Kelly simplificado
        win_rate = self.config.assumed_win_rate
        avg_win_loss = self.config.assumed_win_loss_ratio
        k = (win_rate * avg_win_loss - (1 - win_rate)) / avg_win_loss
        k = np.clip(k, 0.0, self.config.kelly_max)

        # Base pelo pct de posição do config e confiança
        pct = k * confidence * self.config.max_position_pct

        # Ajuste por volatilidade
        vol_adj = self._volatility_factor(volatility)
        pct *= vol_adj

        # Valor bruto de posição
        pos_value = self.current_balance * pct

        # Limitações mín./máx.
        pos_value = self._apply_limits(pos_value)

        # Debug log
        if current_price and pos_value > 0:
            qty = pos_value / current_price
            logger.debug(
                f"📊 Pos size: ${pos_value:.2f} ({pct*100:.2f}%) => "
                f"{qty:.6f} @ ${current_price:.2f}"
            )

        return pos_value

    def _within_risk_limits(self) -> bool:
        """Checa stop-loss diário, número de posições e balance mínimo."""
        # Stop-loss diário
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning(f"🛑 Stop-loss diário atingido (${self.daily_pnl:.2f})")
            return False
        # Máx. posições simultâneas
        if self.position_info is not None:
            logger.debug("Já há posição aberta")
            return False
        # Balance mínimo
        if self.current_balance < self.config.min_balance:
            logger.warning(f"Saldo insuficiente (${self.current_balance:.2f})")
            return False
        return True

    def _volatility_factor(self, vol: float) -> float:
        """Ajusta pct de posição conforme volatilidade."""
        c = self.config
        if vol < c.vol_low:
            return c.vol_factor_low
        if vol < c.vol_med:
            return 1.0
        if vol < c.vol_high:
            return c.vol_factor_high
        return c.vol_factor_extreme

    def _apply_limits(self, value: float) -> float:
        """Aplica limites de posição mínima e máxima."""
        # Mínimo
        if value < self.config.min_trade_usd:
            return 0.0
        # Máx. absoluto
        value = min(value, self.config.max_trade_usd)
        # Máx. percentual do balance
        value = min(value, self.current_balance * self.config.max_position_pct)
        return value

    def set_position_info(self, position: Dict):
        """Armazena dados da posição aberta e conta trade."""
        self.position_info = position
        self.daily_trades += 1

    def clear_position(self):
        """Reseta estado de posição aberta."""
        self.position_info = None

    def should_close_position(
        self,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Verifica TP, SL, trailing e time stop.
        Retorna (deve_fechar, motivo).
        """
        if not self.position_info:
            return False, ""

        side = self.position_info['side']
        entry = self.position_info['entry_price']
        pnl_pct = ((current_price - entry) / entry) if side == 'BUY' \
                  else ((entry - current_price) / entry)

        # Atualiza highest_pnl
        hp = self.position_info.get('highest_pnl', 0.0)
        self.position_info['highest_pnl'] = max(hp, pnl_pct)

        # Stop Loss / Take Profit
        sl = self.position_info.get('stop_loss_pct', self.config.default_sl)
        tp = self.position_info.get('take_profit_pct', self.config.default_tp)
        if pnl_pct <= -sl:
            return True, "Stop Loss"
        if pnl_pct >= tp:
            return True, "Take Profit"

        # Trailing Stop
        thresh = self.config.trailing_pct
        if self.position_info['highest_pnl'] > thresh:
            if pnl_pct < self.position_info['highest_pnl'] * self.config.trailing_retention:
                return True, "Trailing Stop"

        # Time Stop
        elapsed = time.time() - self.position_info.get('entry_time_ts', time.time())
        if elapsed > self.config.max_duration and abs(pnl_pct) < self.config.time_stop_pct:
            return True, "Time Stop"

        return False, ""

    def update_pnl(self, pnl: float, fees: float = 0.0):
        """Atualiza balance, PnL diário, fees e métricas de drawdown."""
        self.daily_pnl += pnl
        self.current_balance += pnl - fees
        self.config.total_fees += fees

        # Histórico circular
        idx = self.pnl_idx % self.config.pnl_history_size
        self.pnl_history[idx] = pnl
        self.pnl_idx += 1

        # Drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        else:
            dd = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, dd)

    def _reset_daily_if_needed(self):
        """Reseta PnL e trades diários ao mudar o dia."""
        today = datetime.now().date()
        if today > self.last_reset_day:
            logger.info("📅 Reset diário do Risk Manager")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_day = today

    def get_risk_metrics(self) -> Dict:
        """Retorna estado atual de risco e métricas principais."""
        current_dd = (self.peak_balance - self.current_balance) / self.peak_balance \
                     if self.peak_balance > 0 else 0.0
        return {
            'balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': current_dd,
            'can_trade': self._within_risk_limits()
        }


class MarketConditionValidator:
    """Valida condições de mercado para decidir se está seguro operar."""

    def __init__(self, config):
        self.config = config
        self._last_check = 0.0
        self._interval = config.market_check_interval
        self.is_safe = True
        self.reasons: List[str] = []

        logger.info("🛡️ Market Validator inicializado")

    async def validate(self, market_data: Dict, client=None) -> Tuple[bool, List[str]]:
        """Valida volatilidade, spread, volume e outros sinais de risco."""
        now = time.time()
        if self.config.debug_mode or now - self._last_check < self._interval:
            return self.is_safe, []

        self.reasons.clear()
        score = 100

        # 1) Volatilidade
        vol = self._get_volatility(market_data)
        if vol is not None:
            if vol > self.config.max_volatility:
                self.reasons.append(f"Vol alta: {vol*100:.2f}%")
                score -= 30
            elif vol > self.config.max_volatility * 0.8:
                score -= 15

        # 2) Spread
        sp = self._get_spread(market_data)
        if sp is not None:
            if sp > self.config.max_spread_bps:
                self.reasons.append(f"Spread alto: {sp:.1f}bps")
                score -= 25
            elif sp > self.config.max_spread_bps * 0.8:
                score -= 10

        # 3) Async volume 24h
        if client:
            ok, reason = await self._check_volume_async(client)
            if not ok:
                self.reasons.append(reason)
                score -= 20

        # 4) Liquidez
        ok, reason = self._check_liquidity(market_data)
        if not ok:
            self.reasons.append(reason)
            score -= 15

        # 5) Horário
        ok, reason = self._check_time()
        if not ok:
            self.reasons.append(reason)
            score -= 10

        # 6) Flash crash
        if self._detect_flash_crash(market_data):
            self.reasons.append("⚠️ FLASH CRASH")
            score = 0

        self.is_safe = (score >= self.config.market_safe_score)
        self._last_check = now
        return self.is_safe, self.reasons

    # Métodos auxiliares abaixo (_get_volatility, _get_spread, _check_volume_async,
    # _check_liquidity, _check_time, _detect_flash_crash) mantêm a mesma lógica,
    # mas com checagens explícitas e retornos claros.

