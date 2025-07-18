"""
Sistema de backtesting integrado para validação de estratégias
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
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
        initial_balance: float = 10000.0
    ) -> Dict:
        logger.info("🔄 Iniciando backtest...")
        # 1) validar colunas
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(historical_data.columns):
            missing = required - set(historical_data.columns)
            logger.error(f"Colunas ausentes: {missing}")
            return {}

        balance = initial_balance
        position: Optional[Dict] = None
        trades: List[Dict] = []
        equity_curve: List[Dict] = []

        # 2) extrair arrays numpy
        o = historical_data['open'].values.astype(float)
        h = historical_data['high'].values.astype(float)
        l = historical_data['low'].values.astype(float)
        c = historical_data['close'].values.astype(float)
        v = historical_data['volume'].values.astype(float)

        # 3) indicadores auxiliares
        atr_series = calculate_atr(h, l, c, period=self.config.atr_period)
        tech = UltraFastTechnicalAnalysis(self.config)
        ml = SimplifiedMLPredictor()
        consolidator = OptimizedSignalConsolidator()

        # 4) loop de backtest
        start_idx = self.config.atr_period
        for i in range(start_idx, len(c)):
            slice_prices = c[: i + 1]
            slice_vols   = v[: i + 1]

            # 4.1) sinais técnico + ML
            t_act, t_conf, t_det = tech.analyze(
                slice_prices[-self.config.tech_window :],
                slice_vols[-self.config.tech_window :]
            )
            features = self._extract_features(c, v, i, t_det)
            m_act, m_conf = ml.predict(features)
            action, conf = consolidator.consolidate(
                [('tech', t_act, t_conf), ('ml', m_act, m_conf)]
            )

            price = c[i]
            current_atr = atr_series[i] if not np.isnan(atr_series[i]) else None

            # 4.2) lógica de entrada
            if position is None and action == 'BUY' and conf >= self.config.min_confidence:
                size_usd = self._calc_size(balance, conf, features['volatility'])
                if size_usd > 0:
                    qty = size_usd / price
                    tp, sl = self._calc_stops(price, current_atr)
                    position = {
                        'entry_price': price,
                        'qty': qty,
                        'tp': tp,
                        'sl': sl,
                        'entry_time': datetime.now()
                    }

            # 4.3) lógica de saída
            elif position:
                should_close, reason = self._should_exit(position, price)
                if should_close:
                    res = self._close(position, price, reason)
                    trades.append(res)
                    balance += res['pnl_net']
                    position = None

            # 4.4) registrar equity curve
            equity_curve.append({'idx': i, 'balance': balance})

        # 5) fechar posição remanescente
        if position:
            res = self._close(position, c[-1], 'end')
            trades.append(res)
            balance += res['pnl_net']

        # 6) calcular métricas finais
        metrics = self._metrics(trades, initial_balance, balance, equity_curve)
        logger.info(f"✅ Backtest concluído — Trades: {len(trades)}, Balance final: ${balance:.2f}")
        return metrics

    def _extract_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        idx: int,
        tech_details: Dict
    ) -> Dict:
        """Retorna dicionário de features para ML (inclui volatilidade e detalhes técnicos)."""
        window = prices[max(0, idx - 50) : idx]
        vol = np.std(window) / np.mean(window) if window.size else 0.0
        return {**tech_details, 'volatility': vol}

    def _calc_size(self, balance: float, conf: float, vol: float) -> float:
        """Calcula valor em USD da posição."""
        base = balance * self.config.max_position_pct * conf
        if vol > 0.03:
            base *= 0.5
        elif vol > 0.02:
            base *= 0.7
        # aplicar limites
        return max(self.config.min_trade_usd, min(base, balance * 0.1))

    def _calc_stops(
        self,
        price: float,
        atr: Optional[float]
    ) -> Tuple[float, float]:
        """Retorna (take_profit_price, stop_loss_price)."""
        if atr and atr > 0:
            return price + atr * self.config.tp_multiplier, price - atr * self.config.sl_multiplier
        return price * (1 + self.config.tp_pct), price * (1 - self.config.sl_pct)

    def _should_exit(
        self,
        pos: Dict,
        price: float
    ) -> Tuple[bool, str]:
        """Verifica TP/SL e retorna (deve_fechar, motivo)."""
        if price >= pos['tp']:
            return True, 'tp'
        if price <= pos['sl']:
            return True, 'sl'
        return False, ''

    def _close(
        self,
        pos: Dict,
        price: float,
        reason: str
    ) -> Dict:
        """Calcula PnL líquido e retorna resultado do trade."""
        entry = pos['entry_price']
        qty = pos['qty']
        pnl = (price - entry) * qty
        fee = 0.001 * (entry + price) * qty
        return {'pnl_net': pnl - fee, 'reason': reason}

    def _metrics(
        self,
        trades: List[Dict],
        init_balance: float,
        final_balance: float,
        equity_curve: List[Dict]
    ) -> Dict:
        """Calcula métricas básicas e retorna dict completo."""
        num = len(trades)
        net_profit = sum(t['pnl_net'] for t in trades)
        total_return = (final_balance - init_balance) / init_balance if init_balance else 0.0

        # Sharpe ratio simplificado
        returns = np.array([t['pnl_net'] / init_balance for t in trades])
        sharpe = (
            np.sqrt(252) * returns.mean() / returns.std()
            if returns.size > 1 and returns.std() > 0 else 0.0
        )

        return {
            'num_trades': num,
            'net_profit': net_profit,
            'return_pct': total_return,
            'sharpe_ratio': sharpe,
            'final_balance': final_balance,
            'equity_curve': equity_curve,
            'trades': trades
        }
