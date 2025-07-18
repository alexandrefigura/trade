'''Sistema de backtesting integrado para validação de estratégias'''  
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from trade_system.logging_config import get_logger
from trade_system.analysis.technical import UltraFastTechnicalAnalysis
from trade_system.analysis.ml import SimplifiedMLPredictor
from trade_system.signals import OptimizedSignalConsolidator
from trade_system.utils import calculate_atr

logger = get_logger(__name__)

class IntegratedBacktester:
    """Sistema de backtesting para validação de estratégias"""

    def __init__(self, config):
        self.config = config

    async def backtest_strategy(
        self,
        historical_data: pd.DataFrame,
        initial_balance: float = 10000
    ) -> Dict:
        """
        Executa backtest com dados históricos
        Returns:
            Dicionário com métricas do backtest
        """
        logger.info("🔄 Iniciando backtest...")

        # Validar dados
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in historical_data.columns:
                logger.error(f"Coluna {col} ausente nos dados")
                return {}

        balance = initial_balance
        position: Optional[Dict] = None
        trades: List[Dict] = []
        equity_curve: List[Dict] = []

        # Extrair arrays
        o = historical_data['open'].astype(float).values
        h = historical_data['high'].astype(float).values
        l = historical_data['low'].astype(float).values
        c = historical_data['close'].astype(float).values
        v = historical_data['volume'].astype(float).values

        # Calcular ATR
        atr_series = calculate_atr(h, l, c, period=self.config.atr_period)

        tech = UltraFastTechnicalAnalysis(self.config)
        ml = SimplifiedMLPredictor()
        consolidator = OptimizedSignalConsolidator()

        for i in range(self.config.atr_period, len(c)):
            price_slice = c[:i+1]
            vol_slice = v[:i+1]

            # Sinais
            t_action, t_conf, t_det = tech.analyze(
                price_slice[-self.config.tech_window:],
                vol_slice[-self.config.tech_window:]
            )
            features = self._extract_features(c, v, i, t_det)
            m_action, m_conf = ml.predict(features)
            action, conf = consolidator.consolidate([
                ('tech', t_action, t_conf),
                ('ml', m_action, m_conf)
            ])

            current = c[i]
            atr = atr_series[i] if not np.isnan(atr_series[i]) else None

            # Lógica de entrada
            if position is None and action == 'BUY' and conf >= self.config.min_confidence:
                size_usd = self._calc_size(balance, conf, features['volatility'])
                qty = size_usd / current
                tp, sl = self._calc_stops(current, atr)
                position = {'entry_price': current, 'qty': qty, 'tp': tp, 'sl': sl}

            # Lógica de saída
            elif position:
                close, reason = self._should_exit(position, current)
                if close:
                    res = self._close(position, current, reason)
                    trades.append(res)
                    balance += res['pnl_net']
                    position = None

            equity_curve.append({'idx': i, 'balance': balance})

        # Fechar posição remanescente
        if position:
            res = self._close(position, c[-1], 'end')
            trades.append(res)
            balance += res['pnl_net']

        return self._metrics(trades, initial_balance, balance, equity_curve)

    def _extract_features(self, prices, volumes, idx, tech_det):
        # ... implementação conforme anterior
        return {
            'volatility': np.std(prices[max(0, idx-50):idx]) / np.mean(prices[max(0, idx-50):idx])
        }

    def _calc_size(self, balance, conf, vol):
        base = balance * self.config.max_position_pct * conf
        if vol > 0.03: base *= 0.5
        elif vol > 0.02: base *= 0.7
        return max(self.config.min_trade_usd, min(base, balance * 0.1))

    def _calc_stops(self, price, atr):
        if atr and atr > 0:
            return price + atr * self.config.tp_multiplier, price - atr * self.config.sl_multiplier
        return price * (1 + self.config.tp_pct), price * (1 - self.config.sl_pct)

    def _should_exit(self, pos, current):
        if current >= pos['tp']: return True, 'tp'
        if current <= pos['sl']: return True, 'sl'
        return False, ''

    def _close(self, pos, current, reason):
        pnl = (current - pos['entry_price']) * pos['qty']
        fees = 0.001 * (pos['entry_price'] + current) * pos['qty']
        return {'pnl_net': pnl - fees, 'reason': reason}

    def _metrics(self, trades, init, final, curve):
        # ... calcular retorno, sharpe etc
        return {'num_trades': len(trades), 'final_balance': final}
