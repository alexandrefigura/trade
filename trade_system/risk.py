"""
Gestão de risco ultra-rápida e validação de condições de mercado
"""
import time
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class UltraFastRiskManager:
    """Gestão de risco com cálculos otimizados"""

    def __init__(self, config):
        self.config = config
        self.initial_balance = getattr(config, "initial_balance", 10000.0)
        self.current_balance = self.initial_balance
        self.daily_pnl = 0.0
        self.max_daily_loss = config.max_daily_loss
        self.position_info: Optional[Dict] = None

        self.pnl_history = np.zeros(1000, dtype=np.float32)
        self.pnl_index = 0

        self.peak_balance = self.current_balance
        self.drawdown_start: Optional[datetime] = None
        self.max_drawdown = 0.0

        self.daily_trades = 0
        self.last_reset_day = datetime.now().date()

        self.max_positions = getattr(config, "max_positions", 1)
        self.current_positions = 0

        logger.info(f"💰 Risk Manager inicializado - Balance: ${self.current_balance:,.2f}")

    def calculate_position_size(
        self,
        confidence: float,
        volatility: float,
        current_price: Optional[float] = None
    ) -> float:
        self._check_daily_reset()
        if not self._check_risk_limits():
            return 0.0

        win_rate = getattr(self.config, "assumed_win_rate", 0.55)
        reward_risk = getattr(self.config, "assumed_rr_ratio", 1.5)
        kelly = (win_rate * reward_risk - (1 - win_rate)) / reward_risk
        kelly = max(0.0, min(kelly, getattr(self.config, "kelly_cap", 0.25)))

        base_pct = kelly * confidence * self.config.max_position_pct

        if volatility < 0.01:
            vol_factor = 1.2
        elif volatility < 0.02:
            vol_factor = 1.0
        elif volatility < 0.03:
            vol_factor = 0.5
        else:
            vol_factor = 0.3
        base_pct *= vol_factor

        value = self.current_balance * base_pct
        min_usd = getattr(self.config, "min_trade_usd", 50.0)
        max_pct_per_trade = getattr(self.config, "max_pct_per_trade", 0.10)
        value = max(min_usd, min(value, self.current_balance * max_pct_per_trade))

        if current_price and value > 0:
            qty = value / current_price
            logger.debug(
                f"📊 Position sizing: ${value:.2f} ({base_pct*100:.2f}%) = {qty:.6f} units @ ${current_price:.2f}"
            )

        return value

    def _check_risk_limits(self) -> bool:
        if self.daily_pnl < -self.max_daily_loss * self.initial_balance:
            logger.warning(f"🛑 Stop daily loss atingido: ${self.daily_pnl:.2f}")
            return False
        if self.current_positions >= self.max_positions:
            logger.debug("Máximo de posições simultâneas atingido")
            return False
        if self.current_balance < getattr(self.config, "min_balance_usd", 100.0):
            logger.warning(f"Saldo insuficiente: ${self.current_balance:.2f}")
            return False
        return True

    def should_close_position(
        self,
        current_price: float,
        entry_price: float,
        side: str = 'BUY'
    ) -> Tuple[bool, str]:
        if self.position_info is None:
            return False, ""

        pnl_pct = ((current_price - entry_price) / entry_price) if side == 'BUY' \
            else ((entry_price - current_price) / entry_price)
        highest = self.position_info.get('highest_pnl', 0.0)
        if pnl_pct > highest:
            self.position_info['highest_pnl'] = pnl_pct

        tp = self.position_info.get('take_profit_pct', self.config.take_profit_pct)
        sl = self.position_info.get('stop_loss_pct', self.config.stop_loss_pct)
        if pnl_pct >= tp:
            return True, "Take Profit"
        if pnl_pct <= -sl:
            return True, "Stop Loss"

        if highest > getattr(self.config, "trailing_start_pct", 0.005):
            trail_factor = getattr(self.config, "trailing_pct", 0.7)
            if pnl_pct < highest * trail_factor:
                return True, "Trailing Stop"

        elapsed = time.time() - self.position_info.get('entry_time', time.time())
        max_dur = self.position_info.get('max_duration', getattr(self.config, "max_position_duration", 3600))
        if elapsed > max_dur and abs(pnl_pct) < getattr(self.config, "time_stop_pct", 0.002):
            return True, "Time Stop"

        return False, ""

    def update_after_trade(self, pnl: float, fees: float = 0.0):
        self.daily_pnl += pnl
        self.current_balance += pnl - fees
        self.pnl_history[self.pnl_index % 1000] = pnl
        self.pnl_index += 1
        self.daily_trades += 1

        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.drawdown_start = None
        else:
            if self.drawdown_start is None:
                self.drawdown_start = datetime.now()
            dd = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, dd)

    def _check_daily_reset(self):
        today = datetime.now().date()
        if today > self.last_reset_day:
            logger.info("📅 Reset diário de métricas")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_day = today

    def set_position(self, info: Dict):
        self.position_info = info
        self.current_positions = 1

    def clear_position(self):
        self.position_info = None
        self.current_positions = 0

    def get_risk_metrics(self) -> Dict:
        curr_dd = ((self.peak_balance - self.current_balance) / self.peak_balance) if self.peak_balance > 0 else 0.0
        return {
            'current_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': curr_dd,
            'daily_trades': self.daily_trades,
            'can_trade': self._check_risk_limits()
        }


class MarketConditionValidator:
    """Valida se as condições de mercado são seguras"""

    def __init__(self, config):
        self.config = config
        self.last_check = 0.0
        self.check_interval = getattr(config, "market_check_interval_s", 60)
        self.is_safe = True
        self.reasons: List[str] = []
        self.score = 100.0

        self.vol_history: List[float] = []
        self.spread_history: List[float] = []
        self.vol24h_history: List[float] = []

        logger.info("🛡️ Market Validator inicializado")

    async def validate(
        self,
        market_data: Dict,
        client=None
    ) -> Tuple[bool, List[str]]:
        now = time.time()
        if getattr(self.config, "debug_mode", False):
            return True, []

        prices = market_data.get('prices')
        if prices is not None and len(prices) >= 100:
            arr = np.asarray(prices[-100:], dtype=np.float64)
            vol = float(np.std(arr) / np.mean(arr))
            self.vol_history.append(vol)
            if vol > self.config.max_volatility:
                self.reasons.append(f"Volatilidade muito alta: {vol*100:.2f}%")
                self.score -= 30
            elif vol > 0.8 * self.config.max_volatility:
                self.reasons.append(f"Volatilidade elevada: {vol*100:.2f}%")
                self.score -= 15

        asks = market_data.get('orderbook_asks', [])
        bids = market_data.get('orderbook_bids', [])
        if (asks is not None and bids is not None
                and len(asks) > 0 and len(bids) > 0
                and asks[0][0] > 0 and bids[0][0] > 0):
            sp = (asks[0][0] - bids[0][0]) / bids[0][0] * 10000
            self.spread_history.append(sp)
            if sp > self.config.max_spread_bps:
                self.reasons.append(f"Spread muito alto: {sp:.1f} bps")
                self.score -= 25
            elif sp > 0.8 * self.config.max_spread_bps:
                self.reasons.append(f"Spread elevado: {sp:.1f} bps")
                self.score -= 10

        if client and now - self.last_check > self.check_interval:
            try:
                ticker = await client.get_ticker(symbol=self.config.symbol)
                vol24 = float(ticker.get('quoteVolume', 0.0))
                self.vol24h_history.append(vol24)
                if vol24 < self.config.min_volume_24h:
                    self.reasons.append(f"Volume 24h baixo: ${vol24:,.0f}")
                    self.score -= 20
            except Exception:
                pass
            self.last_check = now

        if len(asks) < 5 or len(bids) < 5:
            self.reasons.append("Orderbook raso")
            self.score -= 15
        else:
            bid_vol = sum(lvl[1] for lvl in bids[:5])
            ask_vol = sum(lvl[1] for lvl in asks[:5])
            if bid_vol < 10 or ask_vol < 10:
                self.reasons.append("Baixa liquidez")
                self.score -= 15

        hour = datetime.utcnow().hour
        if 2 <= hour <= 6:
            self.reasons.append("Baixa liquidez (2-6 UTC)")
            self.score -= 10
        if datetime.utcnow().weekday() == 6:
            self.reasons.append("Domingo - liquidez reduzida")
            self.score -= 10

        if prices is not None and len(prices) >= 50:
            recent = np.mean(prices[-10:])
            older = np.mean(prices[-50:-40])
            if older > 0 and abs(recent - older)/older > 0.05:
                self.reasons.append("⚠️ FLASH CRASH DETECTADO")
                self.score = 0

        self.score = max(0.0, self.score)
        self.is_safe = (self.score >= getattr(self.config, "min_market_score", 50.0))
        return self.is_safe, self.reasons

    async def validate_market_conditions(
        self,
        market_data: Dict,
        client=None
    ) -> Tuple[bool, List[str]]:
        return await self.validate(market_data, client)

    def get_market_health(self) -> Dict:
        return {
            'score': self.score,
            'is_safe': self.is_safe,
            'reasons': self.reasons,
            'avg_volatility': np.mean(self.vol_history[-10:]) if self.vol_history else 0.0,
            'avg_spread':    np.mean(self.spread_history[-10:]) if self.spread_history else 0.0,
            'last_volume_24h': self.vol24h_history[-1] if self.vol24h_history else 0.0
        }
