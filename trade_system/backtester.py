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
        # Validação de colunas
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(historical_data.columns):
            missing = required - set(historical_data.columns)
            logger.error(f"Colunas ausentes: {missing}")
            return {}

        balance = initial_balance
        position: Optional[Dict] = None
        trades: List[Dict] = []
        equity_curve: List[Dict] = []

        # Extrair arrays NumPy
        o = historical_data['open'].values.astype(float)
        h = historical_data['high'].values.astype(float)
        l = historical_data['low'].values.astype(float)
        c = historical_data['close'].values.astype(float)
        v = historical_data['volume'].values.astype(float)

        # Pré-cálculo de ATR e instância de analisadores
        atr_series = calculate_atr(h, l, c, period=self.config.atr_period)
        tech = UltraFastTechnicalAnalysis(self.config)
        ml = SimplifiedMLPredictor()
        consolidator = OptimizedSignalConsolidator()

        # Loop principal de candles
        for i in range(self.config.atr_period, len(c)):
            slice_p = c[: i + 1]
            slice_v = v[: i + 1]

            # 1) sinais técnica + ML
            t_act, t_conf, t_det = tech.analyze(
                slice_p[-self.config.tech_window:],
                slice_v[-self.config.tech_window:]
            )
            features = self._extract_features(c, v, i, t_det)
            m_act, m_conf = ml.predict(features)
            action, conf = consolidator.consolidate(
                [('tech', t_act, t_conf), ('ml', m_act, m_conf)]
            )

            price = c[i]
            current_atr = atr_series[i] if not np.isnan(atr_series[i]) else None

            # 2) entrada LONG
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

            # 3) saída
            elif position:
                should_close, reason = self._should_exit(position, price)
                if should_close:
                    result = self._close(position, price, reason)
                    trades.append(result)
                    balance += result['pnl_net']
                    position = None

            # 4) registrar equity curve
            equity_curve.append({'idx': i, 'balance': balance})

        # 5) fechar posição remanescente
        if position:
            result = self._close(position, c[-1], 'end')
            trades.append(result)
            balance += result['pnl_net']

        # 6) calcular e retornar métricas
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
        """Extrai features para ML (volatilidade + detalhes técnicos)."""
        window = prices[max(0, idx - 50):idx]
        vol = np.std(window) / np.mean(window) if window.size else 0.0
        return {**tech_details, 'volatility': vol}

    def _calc_size(self, balance: float, conf: float, vol: float) -> float:
        """Calcula valor (USD) da posição baseado em confiança e volatilidade."""
        base = balance * self.config.max_position_pct * conf
        if vol > 0.03:
            base *= 0.5
        elif vol > 0.02:
            base *= 0.7
        return max(self.config.min_trade_usd, min(base, balance * 0.1))

    def _calc_stops(
        self,
        price: float,
        atr: Optional[float]
    ) -> Tuple[float, float]:
        """Retorna (take_profit, stop_loss) baseados em ATR ou percentuais."""
        if atr and atr > 0:
            return price + atr * self.config.tp_multiplier, price - atr * self.config.sl_multiplier
        return price * (1 + self.config.tp_pct), price * (1 - self.config.sl_pct)

    def _should_exit(
        self,
        pos: Dict,
        price: float
    ) -> Tuple[bool, str]:
        """Verifica SL/TP e indica se deve fechar posição."""
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
        """Calcula PnL líquido e retorna dados do trade."""
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
        """Métricas básicas: número de trades, lucro, retorno e Sharpe."""
        num = len(trades)
        net_profit = sum(t['pnl_net'] for t in trades)
        total_return = (final_balance - init_balance) / init_balance if init_balance else 0.0

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


# ------------------------------------------------------------------------------
# Entry point esperado pelo CLI ("trade-system paper")
# ------------------------------------------------------------------------------
import pandas as _pd
import numpy as _np
from typing import Optional as _Opt, Dict as _Dict
from trade_system.config import get_config
from binance.client import Client

async def run_backtest_validation(
    config=None,
    days: int = 7,
    debug_mode: bool = False
) -> _Opt[_Dict]:
    """
    Chamado por `trade-system paper`. Baixa candles, monta DataFrame e executa backtest.
    """
    # 1) Config
    if config is None:
        config = get_config(debug_mode=debug_mode)

    logger.info(f"🔬 Executando backtest de validação ({days} dias)...")

    # 2) Credenciais
    if not getattr(config, "api_key", None) or not getattr(config, "api_secret", None):
        logger.error("❌ Credenciais da Binance não configuradas")
        return None

    client = Client(config.api_key, config.api_secret)

    # 3) Intervalo e limite
    if days <= 1:
        interval = Client.KLINE_INTERVAL_1MINUTE
        expected = days * 24 * 60
    elif days <= 7:
        interval = Client.KLINE_INTERVAL_5MINUTE
        expected = days * 24 * 12
    elif days <= 30:
        interval = Client.KLINE_INTERVAL_15MINUTE
        expected = days * 24 * 4
    else:
        interval = Client.KLINE_INTERVAL_1HOUR
        expected = days * 24

    limit = min(expected, 1000)
    logger.info(f"📊 Baixando {limit} candles de {config.symbol}...")

    # 4) Download e DataFrame
    klines = client.get_klines(symbol=config.symbol, interval=interval, limit=limit)
    df = _pd.DataFrame(
        klines,
        columns=[
            "timestamp","open","high","low","close","volume",
            "close_time","quote_volume","trades","taker_buy_base",
            "taker_buy_quote","ignore"
        ],
    )
    df["timestamp"] = _pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    logger.info(f"✅ Dados carregados: {len(df)} candles")
    logger.info(f"   Período: {df.index[0]} até {df.index[-1]}")
    logger.info(f"   Preço atual: ${df['close'].iloc[-1]:,.2f}")

    # 5) Executa backtest
    backtester = IntegratedBacktester(config)
    return await backtester.backtest_strategy(df)
