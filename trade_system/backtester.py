# trade_system/backtester.py

"""
Sistema de backtesting integrado para validação de estratégias
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
        initial_balance: float = 10_000.0
    ) -> Dict:
        logger.info("🔄 Iniciando backtest...")

        # Valida colunas necessárias
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in historical_data.columns:
                logger.error(f"Coluna {col} ausente nos dados")
                return {}

        balance = initial_balance
        position: Optional[Dict] = None
        trades: List[Dict] = []
        equity_curve: List[Dict] = []

        # Extrai numpy arrays
        o = historical_data['open'].values.astype(float)
        h = historical_data['high'].values.astype(float)
        l = historical_data['low'].values.astype(float)
        c = historical_data['close'].values.astype(float)
        v = historical_data['volume'].values.astype(float)

        # Série de ATR para stops dinâmicos
        atr_series = calculate_atr(h, l, c, period=self.config.atr_period)

        tech = UltraFastTechnicalAnalysis(self.config)
        ml = SimplifiedMLPredictor()
        consolidator = OptimizedSignalConsolidator()

        # Loop de iteração sobre candles
        for i in range(self.config.atr_period, len(c)):
            now_ts = historical_data.index[i] if hasattr(historical_data, 'index') else i
            price_slice = c[: i + 1]
            vol_slice = v[: i + 1]

            # 1) Análise técnica
            t_action, t_conf, t_details = tech.analyze(price_slice, vol_slice)

            # 2) Extrai features para ML
            features = self._extract_features(price_slice, i)
            features['rsi'] = t_details.get('rsi', 50.0)

            # 3) ML predictor
            m_action, m_conf = ml.predict(features)

            # 4) Consolidar sinais
            action, conf = consolidator.consolidate([
                ('tech', t_action, t_conf),
                ('ml',   m_action, m_conf),
            ])

            current_price = c[i]
            atr = atr_series[i] if not np.isnan(atr_series[i]) else None

            # Abrir posição LONG
            if position is None and action == 'BUY' and conf >= self.config.min_confidence:
                size_usd = self._calc_size(balance, conf, features['volatility'])
                if size_usd > 0:
                    qty = size_usd / current_price
                    tp, sl = self._calc_stops(current_price, atr)
                    position = {
                        'entry_price': current_price,
                        'qty': qty,
                        'tp': tp,
                        'sl': sl,
                    }
                    logger.info(f"🟢 Entrada LONG: qty={qty:.6f} @ {current_price:.2f}")

            # Verifica saída
            elif position is not None:
                should_close, reason = self._should_exit(position, current_price)
                if should_close:
                    trade = self._close(position, current_price, reason)
                    trades.append(trade)
                    balance += trade['pnl_net']
                    position = None

            # Registra equity curve
            equity_curve.append({
                'timestamp': now_ts,
                'balance': balance,
                'in_position': position is not None
            })

        # Fechar posição remanescente ao final
        if position is not None:
            final_price = c[-1]
            trade = self._close(position, final_price, 'end')
            trades.append(trade)
            balance += trade['pnl_net']

        # Compila métricas
        return self._metrics(trades, initial_balance, balance, equity_curve)

    def _extract_features(
        self,
        prices: np.ndarray,
        idx: int
    ) -> Dict:
        """Extrai features básicas (e.g., volatilidade) para ML."""
        window = min(idx, 50)
        if window > 1:
            recent = prices[idx - window: idx]
            vol = float(np.std(recent) / np.mean(recent))
        else:
            vol = 0.0
        return {'volatility': vol}

    def _calc_size(self, balance: float, conf: float, vol: float) -> float:
        """
        Calcula alocação em USD para a posição:
          - Proporcional à confiança e ao risco (volatilidade).
          - Respeita min/max definidos em config.
        """
        base = balance * self.config.max_position_pct * conf
        if vol > 0.03:
            base *= 0.5
        elif vol > 0.02:
            base *= 0.7

        min_usd = getattr(self.config, 'min_trade_usd', 50.0)
        max_usd = balance * 0.1
        return max(min_usd, min(base, max_usd))

    def _calc_stops(self, price: float, atr: Optional[float]) -> Tuple[float, float]:
        """Define TP e SL usando ATR ou percentuais fixos."""
        if atr and atr > 0:
            tp = price + atr * self.config.tp_multiplier
            sl = price - atr * self.config.sl_multiplier
        else:
            tp = price * (1 + self.config.tp_pct)
            sl = price * (1 - self.config.sl_pct)
        return tp, sl

    def _should_exit(self, pos: Dict, current: float) -> Tuple[bool, str]:
        """Verifica se atingiu TP ou SL."""
        if current >= pos['tp']:
            return True, 'tp'
        if current <= pos['sl']:
            return True, 'sl'
        return False, ''

    def _close(self, pos: Dict, current: float, reason: str) -> Dict:
        """Fecha posição e calcula P&L líquido considerando fees."""
        entry = pos['entry_price']
        qty = pos['qty']
        pnl = (current - entry) * qty
        fees = (abs(entry) + abs(current)) * qty * 0.001  # 0.1% fees
        net = pnl - fees
        logger.info(f"🔴 Saída ({reason}): P&L net {net:.2f}")
        return {
            'timestamp': datetime.utcnow(),
            'entry_price': entry,
            'exit_price': current,
            'qty': qty,
            'pnl_gross': pnl,
            'fees': fees,
            'pnl_net': net,
            'reason': reason
        }

    def _metrics(
        self,
        trades: List[Dict],
        initial_balance: float,
        final_balance: float,
        equity_curve: List[Dict]
    ) -> Dict:
        """Calcula métricas finais e retorna resultados."""
        return {
            'num_trades': len(trades),
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'net_profit': final_balance - initial_balance,
            'equity_curve': equity_curve,
            'trades': trades
        }


async def run_backtest_validation(
    config=None,
    days: int = 7,
    debug_mode: bool = False
) -> Optional[Dict]:
    """
    Função auxiliar para rodar backtest com dados da Binance.
    """
    from trade_system.config import get_config
    from binance.client import Client

    if config is None:
        config = get_config(debug_mode=debug_mode)

    logger.info(f"🔬 Executando backtest de validação ({days} dias)...")

    if not config.api_key or not config.api_secret:
        logger.error("❌ Credenciais da Binance não configuradas")
        return None

    try:
        client = Client(config.api_key, config.api_secret)

        # Define intervalo e limita obtido
        if days <= 1:
            interval = Client.KLINE_INTERVAL_1MINUTE
            expected = days * 24 * 60
        elif days <= 7:
            interval = Client.KLINE_INTERVAL_5MINUTE
            expected = days * 24 * 12
        else:
            interval = Client.KLINE_INTERVAL_15MINUTE
            expected = days * 24 * 4

        limit = min(expected, 1000)
        logger.info(f"📊 Baixando {limit} candles de {config.symbol}...")

        klines = client.get_klines(
            symbol=config.symbol,
            interval=interval,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df[['open', 'high', 'low', 'close', 'volume']] = \
            df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        logger.info(f"✅ Dados carregados: {len(df)} candles")
        logger.info(f"   Período: {df.index[0]} até {df.index[-1]}")
        logger.info(f"   Preço atual: ${df['close'].iloc[-1]:,.2f}")

        backtester = IntegratedBacktester(config)
        results = await backtester.backtest_strategy(df)

        # Pós-backtest
        if results and results.get('num_trades', 0) > 0:
            if results['net_profit'] < 0:
                logger.warning("⚠️ Backtest terminou no prejuízo")
        else:
            logger.warning("⚠️ Nenhum trade executado no backtest")

        return results

    except Exception as e:
        logger.error(f"❌ Erro no backtest: {e}")
        return None
