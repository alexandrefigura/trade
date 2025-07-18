"""
Validação das condições de mercado com base em volatilidade, spread, volume e horário
"""
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class MarketConditionValidator:
    def __init__(self, config):
        self.config = config
        self.score = 100.0
        self.reasons: List[str] = []
        self.last_check = 0.0
        self.check_interval = getattr(config, "market_check_interval_s", 60)
        self.vol_history: List[float] = []
        self.spread_history: List[float] = []

    async def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        if self.config.debug_mode:
            return True, []

        self.score = 100.0
        self.reasons.clear()

        # Volatilidade
        prices = data.get("prices", [])
        if len(prices) >= 100:
            vol = float(np.std(prices[-100:]) / np.mean(prices[-100:]))
            self.vol_history.append(vol)
            if vol > self.config.max_volatility:
                self.reasons.append(f"Volatilidade alta: {vol:.2%}")
                self.score -= 25
            elif vol > 0.8 * self.config.max_volatility:
                self.reasons.append(f"Volatilidade elevada: {vol:.2%}")
                self.score -= 15

        # Spread
        asks = data.get("orderbook_asks", [])
        bids = data.get("orderbook_bids", [])
        if asks and bids and asks[0][0] > 0 and bids[0][0] > 0:
            spread_bps = (asks[0][0] - bids[0][0]) / bids[0][0] * 10000
            self.spread_history.append(spread_bps)
            if spread_bps > self.config.max_spread_bps:
                self.reasons.append(f"Spread alto: {spread_bps:.1f} bps")
                self.score -= 25
            elif spread_bps > 0.8 * self.config.max_spread_bps:
                self.reasons.append(f"Spread elevado: {spread_bps:.1f} bps")
                self.score -= 10

        # Horário
        hour = datetime.utcnow().hour
        if 2 <= hour <= 6:
            self.reasons.append("Horário de baixa liquidez (2h-6h UTC)")
            self.score -= 10
        if datetime.utcnow().weekday() == 6:
            self.reasons.append("Domingo - mercado fraco")
            self.score -= 10

        is_safe = self.score >= getattr(self.config, "min_market_score", 50)
        return is_safe, self.reasons
