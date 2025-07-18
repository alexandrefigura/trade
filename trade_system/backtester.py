"""
Sistema de backtesting integrado para validação de estratégias
"""
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
        logger.info("🔄 Iniciando backtest...")
        # Validação de dados
        required = ['open', 'high', 'low', 'close', 'volume']
        if not set(required).issubset(historical_data.columns):
            logger.error(f"Dados faltando colunas OHLCV: {required}")
            return {}

        balance = initial_balance
        position: Optional[Dict] = None
        trades: List[Dict] = []
        equity_curve: List[Dict] = []

        # Extrair arrays
        o = historical_data['open'].values.astype(float)
        h = historical_data['high'].values.astype(float)
        l = historical_data['low'].values.astype(float)
        c = historical_data['close'].values.astype(float)
        v = historical_data['volume'].values.astype(float)

        # Indicadores auxiliares
        atr = calculate_atr(h, l, c, period=self.config.atr_period)
        tech = UltraFastTechnicalAnalysis(self.config)
        ml = SimplifiedMLPredictor()
        consolidator = OptimizedSignalConsolidator()

        # Loop principal
        for i in range(self.config.atr_period, len(c)):
            price_slice = c[: i + 1]
            vol_slice = v[: i + 1]

            # 1) Gerar sinais técnico e ML
            t_act, t_conf, t_det = tech.analyze(
                price_slice[-self.config.tech_window :],
                vol_slice[-self.config.tech_window :],
            )
            features = self._extract_features(c, v, i, t_det)
            m_act, m_conf = ml.predict(features)
            action, conf = consolidator.consolidate(
                [('tech', t_act, t_conf), ('ml', m_act, m_conf)]
            )

            current = c[i]
            current_atr = atr[i] if not np.isnan(atr[i]) else None

            # 2) Entrada (LONG only)
            if position is None and action == 'BUY' and conf >= self.config.min_confidence:
                size_usd = self._calc_size(balance, conf, features['volatility'])
                if size_usd > 0:
                    qty = size_usd / current
                    tp, sl = self._calc_stops(current, current_atr)
                    position = {
                        'entry_price': current,
                        'qty': qty,
                        'tp': tp,
                        'sl': sl,
                        'entry_time': datetime.now(),
                    }

            # 3) Saída
            elif position:
                close_flag, reason = self._should_exit(position, current)
                if close_flag:
                    result = self._close(position, current, reason)
                    trades.append(result)
                    balance += result['pnl_net']
                    position = None

            # 4) Equity curve
            equity_curve.append({'idx': i, 'balance': balance})

        # Fechar posição remanescente
        if position:
            result = self._close(position, c[-1], 'end')
            trades.append(result)
            balance += result['pnl_net']

        # 5) Métricas finais
        metrics = self._metrics(trades, initial_balance, balance, equity_curve)
        logger.info(f"✅ Backtest finalizado — Trades: {len(trades)} | Balance final: ${balance:.2f}")
        return metrics

    def _extract_features(self, prices, volumes, idx, tech_det) -> Dict:
        # Exemplo básico: volatilidade recente + herdar detalhes técnicos
        window = prices[max(0, idx - 50) : idx]
        vol = np.std(window) / np.mean(window) if window.size else 0.0
        return {**tech_det, 'volatility': vol}

    def _calc_size(self, balance: float, conf: float, vol: float) -> float:
        base = balance * self.config.max_position_pct * conf
        if vol > 0.03:
            base *= 0.5
        elif vol > 0.02:
            base *= 0.7
        # limites mínimo/máximo
        return max(self.config.min_trade_usd, min(base, balance * 0.1))

    def _calc_stops(self, price: float, atr: Optional[float]) -> Tuple[float, float]:
        if atr and atr > 0:
            tp = price + atr * self.config.tp_multiplier
            sl = price - atr * self.config.sl_multiplier
        else:
            tp = price * (1 + self.config.tp_pct)
            sl = price * (1 - self.config.sl_pct)
        return tp, sl

    def _should_exit(self, pos: Dict, current: float) -> Tuple[bool, str]:
        if current >= pos['tp']:
            return True, 'tp'
        if current <= pos['sl']:
            return True, 'sl'
        return False, ''

    def _close(self, pos: Dict, current: float, reason: str) -> Dict:
        entry = pos['entry_price']
        qty = pos['qty']
        pnl = (current - entry) * qty
        fee = 0.001 * (entry + current) * qty
        return {'pnl_net': pnl - fee, 'reason': reason}

    def _metrics(
        self,
        trades: List[Dict],
        init_balance: float,
        final_balance: float,
        equity_curve: List[Dict]
    ) -> Dict:
        num = len(trades)
        net = sum(t['pnl_net'] for t in trades)
        ret = (final_balance - init_balance) / init_balance if init_balance else 0.0
        # Sharpe simplificado
        returns = np.array([t['pnl_net'] / init_balance for t in trades])
        sharpe = (
            np.sqrt(252) * returns.mean() / returns.std()
            if returns.size > 1 and returns.std() > 0
            else 0.0
        )
        return {
            'num_trades': num,
            'net_profit': net,
            'return_pct': ret,
            'sharpe': sharpe,
            'final_balance': final_balance,
            'equity_curve': equity_curve,
            'trades': trades,
        }
