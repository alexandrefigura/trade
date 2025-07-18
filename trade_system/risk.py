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
        # Se config não contiver initial_balance, usa 10000 por default
        self.initial_balance = getattr(config, "initial_balance", 10000.0)
        self.current_balance = self.initial_balance
        self.daily_pnl = 0.0
        self.max_daily_loss = config.max_daily_loss
        self.position_info = None

        # Histórico de PnL em buffer circular
        self.pnl_history = np.zeros(1000, dtype=np.float32)
        self.pnl_index = 0

        # Estatísticas de drawdown
        self.peak_balance = self.current_balance
        self.drawdown_start = None
        self.max_drawdown = 0.0

        # Contador de trades diários
        self.daily_trades = 0
        self.last_reset_day = datetime.now().date()

        # Limites de posições simultâneas
        self.max_positions = getattr(config, "max_positions", 1)
        self.current_positions = 0

        logger.info(f"💰 Risk Manager inicializado - Balance: ${self.current_balance:,.2f}")

    def calculate_position_size(
        self,
        confidence: float,
        volatility: float,
        current_price: Optional[float] = None
    ) -> float:
        """
        Calcula rapidamente o valor em USD para a próxima posição
        """
        self._check_daily_reset()
        if not self._check_risk_limits():
            return 0.0

        # Critério de Kelly simplificado
        win_rate = getattr(self.config, "assumed_win_rate", 0.55)
        reward_risk = getattr(self.config, "assumed_rr_ratio", 1.5)
        kelly = (win_rate * reward_risk - (1 - win_rate)) / reward_risk
        kelly = max(0.0, min(kelly, getattr(self.config, "kelly_cap", 0.25)))

        # Pct de balance alocado
        base_pct = kelly * confidence * self.config.max_position_pct

        # Ajuste por volatilidade
        if volatility < 0.01:
            vol_factor = 1.2
        elif volatility < 0.02:
            vol_factor = 1.0
        elif volatility < 0.03:
            vol_factor = 0.5
        else:
            vol_factor = 0.3
        base_pct *= vol_factor

        # Valor bruto da posição
        value = self.current_balance * base_pct

        # Limites de valor
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
        """Verifica se ainda pode abrir nova posição hoje"""
        # Limite de perda diária
        if self.daily_pnl < -self.max_daily_loss * self.initial_balance:
            logger.warning(f"🛑 Stop daily loss atingido: ${self.daily_pnl:.2f}")
            return False
        # Limite de posições simultâneas
        if self.current_positions >= self.max_positions:
            logger.debug("Máximo de posições simultâneas atingido")
            return False
        # Saldo mínimo remanescente
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
        """
        Verifica condições de saída: TP, SL, trailing, time stop
        """
        if self.position_info is None:
            return False, ""

        # PnL percentual
        pnl_pct = ((current_price - entry_price) / entry_price) if side == 'BUY' \
            else ((entry_price - current_price) / entry_price)
        # Atualiza maior PnL alcançado
        highest = self.position_info.get('highest_pnl', 0.0)
        if pnl_pct > highest:
            self.position_info['highest_pnl'] = pnl_pct

        tp = self.position_info.get('take_profit_pct', self.config.take_profit_pct)
        sl = self.position_info.get('stop_loss_pct', self.config.stop_loss_pct)
        # TP / SL
        if pnl_pct >= tp:
            return True, "Take Profit"
        if pnl_pct <= -sl:
            return True, "Stop Loss"
        # Trailing stop
        if highest > getattr(self.config, "trailing_start_pct", 0.005):
            trail_factor = getattr(self.config, "trailing_pct", 0.7)
            if pnl_pct < highest * trail_factor:
                return True, "Trailing Stop"
        # Time stop
        elapsed = time.time() - self.position_info.get('entry_time', time.time())
        max_dur = self.position_info.get('max_duration', getattr(self.config, "max_position_duration", 3600))
        if elapsed > max_dur and abs(pnl_pct) < getattr(self.config, "time_stop_pct", 0.002):
            return True, "Time Stop"

        return False, ""

    def update_after_trade(self, pnl: float, fees: float = 0.0):
        """
        Deve ser chamado após cada trade fechado para atualizar PnL e estatísticas
        """
        self.daily_pnl += pnl
        self.current_balance += pnl - fees
        self.pnl_history[self.pnl_index % 1000] = pnl
        self.pnl_index += 1
        self.daily_trades += 1

        # Atualiza peak & drawdown
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
        """Define informações da posição aberta"""
        self.position_info = info
        self.current_positions = 1

    def clear_position(self):
        """Limpa o estado de posição atual"""
        self.position_info = None
        self.current_positions = 0

    def get_risk_metrics(self) -> Dict:
        """Retorna as métricas de risco atuais"""
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

        # Históricos
        self.vol_history: List[float] = []
        self.spread_history: List[float] = []
        self.vol24h_history: List[float] = []

        logger.info("🛡️ Market Validator inicializado")

    async def validate(
        self,
        market_data: Dict,
        client=None
    ) -> Tuple[bool, List[str]]:
        """Executa todas as checagens e retorna (is_safe, reasons)"""
        now = time.time()
        if getattr(self.config, "debug_mode", False):
            return True, []

        # 1. Volatilidade
        vol = self._get_volatility(market_data)
        if vol is not None:
            self.vol_history.append(vol)
            if vol > self.config.max_volatility:
                self.reasons.append(f"Volatilidade muito alta: {vol*100:.2f}%")
                self.score -= 30
            elif vol > 0.8 * self.config.max_volatility:
                self.reasons.append(f"Volatilidade elevada: {vol*100:.2f}%")
                self.score -= 15

        # 2. Spread
        sp = self._get_spread(market_data)
        if sp is not None:
            self.spread_history.append(sp)
            if sp > self.config.max_spread_bps:
                self.reasons.append(f"Spread muito alto: {sp:.1f} bps")
                self.score -= 25
            elif sp > 0.8 * self.config.max_spread_bps:
                self.reasons.append(f"Spread elevado: {sp:.1f} bps")
                self.score -= 10

        # 3. Volume 24h (assíncrono)
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

        # 4. Liquidez
        ok_liq, reason_liq = self._check_liquidity(market_data)
        if not ok_liq:
            self.reasons.append(reason_liq)
            self.score -= 15

        # 5. Horário de trading
        ok_time, reason_time = self._check_trading_time()
        if not ok_time:
            self.reasons.append(reason_time)
            self.score -= 10

        # 6. Flash crash
        if self._detect_flash_crash(market_data):
            self.reasons.append("⚠️ FLASH CRASH DETECTADO")
            self.score = 0

        self.score = max(0.0, self.score)
        self.is_safe = (self.score >= getattr(self.config, "min_market_score", 50.0))
        return self.is_safe, self.reasons

    # Alias adicionado para compatibilidade com main.py
    async def validate_market_conditions(
        self,
        market_data: Dict,
        client=None
    ) -> Tuple[bool, List[str]]:
        """
        Deprecated: use `validate()`. Mantido para compatibilidade com main.py.
        """
        return await self.validate(market_data, client)

    def _get_volatility(self, data: Dict) -> Optional[float]:
        prices = data.get('prices')
        if prices and len(prices) >= 100:
            arr = np.array(prices[-100:], dtype=np.float64)
            return float(np.std(arr) / np.mean(arr))
        return None

    def _get_spread(self, data: Dict) -> Optional[float]:
        asks = data.get('orderbook_asks')
        bids = data.get('orderbook_bids')
        if asks and bids and asks[0][0] > 0 and bids[0][0] > 0:
            sp = (asks[0][0] - bids[0][0]) / bids[0][0] * 10000
            return float(sp)
        return None

    def _check_liquidity(self, data: Dict) -> Tuple[bool, str]:
        asks = data.get('orderbook_asks', [])
        bids = data.get('orderbook_bids', [])
        if len(asks) < 5 or len(bids) < 5:
            return False, "Orderbook raso"
        bid_vol = sum(lvl[1] for lvl in bids[:5])
        ask_vol = sum(lvl[1] for lvl in asks[:5])
        if bid_vol < 10 or ask_vol < 10:
            return False, "Baixa liquidez"
        return True, ""

    def _check_trading_time(self) -> Tuple[bool, str]:
        hour = datetime.utcnow().hour
        if 2 <= hour <= 6:
            return False, "Baixa liquidez (2-6 UTC)"
        if datetime.utcnow().weekday() == 6:
            return False, "Domingo - liquidez reduzida"
        return True, ""

    def _detect_flash_crash(self, data: Dict) -> bool:
        prices = data.get('prices')
        if prices and len(prices) >= 50:
            recent = np.mean(prices[-10:])
            older = np.mean(prices[-50:-40])
            if older > 0 and abs(recent - older) / older > 0.05:
                return True
        return False

    def get_market_health(self) -> Dict:
        return {
            'score': self.score,
            'is_safe': self.is_safe,
            'reasons': self.reasons,
            'avg_volatility': np.mean(self.vol_history[-10:]) if self.vol_history else 0.0,
            'avg_spread': np.mean(self.spread_history[-10:]) if self.spread_history else 0.0,
            'last_volume_24h': self.vol24h_history[-1] if self.vol24h_history else 0.0
        }
