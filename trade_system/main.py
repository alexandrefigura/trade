"""
Módulo principal para orquestração do sistema de trading
"""
import asyncio
import signal
from datetime import datetime
import numpy as np
from typing import Optional, Dict, List, Tuple

from trade_system.config import get_config
from trade_system.logging_config import setup_logging, get_logger
from trade_system.cache import UltraFastCache
from trade_system.rate_limiter import RateLimiter
from trade_system.alerts import AlertSystem
from trade_system.websocket_manager import UltraFastWebSocketManager
from trade_system.analysis.technical import UltraFastTechnicalAnalysis
from trade_system.analysis.orderbook import ParallelOrderbookAnalyzer
from trade_system.analysis.ml import SimplifiedMLPredictor
from trade_system.risk import UltraFastRiskManager, MarketConditionValidator
from trade_system.signals import OptimizedSignalConsolidator
from trade_system.checkpoint import CheckpointManager

logger = get_logger(__name__)


class TradingSystem:
    """Sistema principal de trading"""

    def __init__(self, config=None, paper_trading: bool = True):
        self.config = config or get_config()
        self.paper_trading = paper_trading

        # Infra
        self.cache = UltraFastCache(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.alert_system = AlertSystem(self.config)
        self.ws_manager = UltraFastWebSocketManager(self.config, self.cache)

        # Análise
        self.technical_analyzer = UltraFastTechnicalAnalysis(self.config)
        self.orderbook_analyzer = ParallelOrderbookAnalyzer(self.config)
        self.ml_predictor = SimplifiedMLPredictor()
        self.signal_consolidator = OptimizedSignalConsolidator()

        # Risco e validação
        self.risk_manager = UltraFastRiskManager(self.config)
        self.market_validator = MarketConditionValidator(self.config)

        # Checkpoint
        self.checkpoint_manager = CheckpointManager()

        # Estado
        self.position: Optional[Dict] = None
        self.is_running = False
        self.performance_stats: Dict[str, float] = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'start_balance': self.risk_manager.current_balance,
            'session_start': datetime.now().timestamp()
        }

        logger.info(f"🚀 Sistema inicializado - Modo: {'PAPER TRADING' if paper_trading else 'LIVE'}")

    async def initialize(self):
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            self._restore_from_checkpoint(checkpoint)

        self.ws_manager.start_delayed()
        await self.alert_system.send_startup_alert(mode="PAPER" if self.paper_trading else "LIVE")

        logger.info("⏳ Aguardando dados do WebSocket...")
        for _ in range(50):
            await asyncio.sleep(0.1)
            if self.ws_manager.buffer_filled or self.ws_manager.buffer_index > 100:
                logger.info("✅ Dados recebidos!")
                break

    async def run(self):
        self.is_running = True
        cycle = 0

        try:
            while self.is_running:
                data = self.ws_manager.get_latest_data()
                if not data:
                    await asyncio.sleep(0.1)
                    continue

                is_safe, reasons = await self.market_validator.validate(data, client=self.ws_manager.client)
                if not is_safe and not self.config.debug_mode:
                    if cycle % (int(10_000 / self.config.main_loop_interval_ms)) == 0:
                        logger.warning(f"⚠️ Mercado inseguro: {', '.join(reasons)}")
                    await asyncio.sleep(0.1)
                    cycle += 1
                    continue

                signals = await self._analyze_market(data)
                action, conf = self.signal_consolidator.consolidate(signals)

                if self.position:
                    await self._manage_position(data, action, conf)
                else:
                    if action != 'HOLD' and conf >= self.config.min_confidence:
                        await self._open_position(data, action, conf)

                if self.checkpoint_manager.should_checkpoint():
                    await self._save_checkpoint()

                cycle += 1
                await asyncio.sleep(self.config.main_loop_interval_ms / 1000.0)

        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}", exc_info=True)
            await self.alert_system.send_alert("Erro no Sistema", str(e), level="critical")
        finally:
            await self.shutdown()

    async def _analyze_market(self, data: Dict) -> List[Tuple[str, str, float]]:
        prices = np.array(data['prices'], dtype=np.float64)
        volumes = np.array(data['volumes'], dtype=np.float64)

        ta_a, ta_c, ta_d = self.technical_analyzer.analyze(prices, volumes)
        ob_a, ob_c, ob_d = self.orderbook_analyzer.analyze(
            data['orderbook_bids'], data['orderbook_asks'], self.cache
        )
        features = {
            'rsi': ta_d.get('rsi', 50.0),
            'momentum': self._calculate_momentum(prices),
            'volume_ratio': self._calculate_volume_ratio(volumes),
            'spread_bps': ob_d.get('spread_bps', 0.0),
            'volatility': self._calculate_volatility(prices)
        }
        ml_a, ml_c = self.ml_predictor.predict(features)

        return [
            ('technical', ta_a, ta_c),
            ('orderbook', ob_a, ob_c),
            ('ml', ml_a, ml_c),
        ]

    async def _open_position(self, data: Dict, action: str, conf: float):
        if self.paper_trading:
            await self._open_paper_position(data, action, conf)
        else:
            logger.error("🔴 Trading real ainda não implementado")

    async def _open_paper_position(self, data: Dict, action: str, conf: float):
        price = float(data['prices'][-1])
        vol = self._calculate_volatility(np.array(data['prices'], float))

        size = self.risk_manager.calculate_position_size(conf, vol, price)
        if size <= 0:
            return

        fee = size * self.config.trade_fee_pct
        qty = size / price

        self.position = {
            'side': action,
            'entry_price': price,
            'quantity': qty,
            'entry_time': datetime.now(),
            'confidence': conf,
            'entry_fee': fee,
        }

        self.risk_manager.set_position(self.position)
        self.risk_manager.current_balance -= fee

        logger.info(f"🟢 POSIÇÃO ABERTA [{action}] @ ${price:.2f} x {qty:.6f} = ${size:.2f}")
        await self.alert_system.send_alert(
            f"POSIÇÃO ABERTA {action}",
            f"Preço: ${price:.2f}\nQuantidade: {qty:.6f}\nValor: ${size:.2f}",
            level="info"
        )

    async def _manage_position(self, data: Dict, action: str, conf: float):
        price = float(data['prices'][-1])
        close, reason = self.risk_manager.should_close_position(
            price, self.position['entry_price'], side=self.position['side']
        )
        if close:
            await self._close_position(price, reason)

    async def _close_position(self, exit_price: float, reason: str):
        if not self.position:
            return

        side = self.position['side']
        entry = self.position['entry_price']
        qty = self.position['quantity']

        pnl = (exit_price - entry) * qty if side == 'BUY' else (entry - exit_price) * qty
        fee = exit_price * qty * self.config.trade_fee_pct
        net = pnl - fee

        self.performance_stats['total_trades']   += 1
        self.performance_stats['winning_trades'] += 1 if net>0 else 0
        self.performance_stats['total_pnl']      += net
        self.performance_stats['total_fees']     += (self.position['entry_fee'] + fee)

        self.risk_manager.update_after_trade(net, fee)
        self.risk_manager.clear_position()

        logger.info(f"🔴 POSIÇÃO FECHADA ({reason}) P&L: ${net:.2f} Taxa: ${fee:.2f}")
        await self.alert_system.send_alert(
            f"POSIÇÃO FECHADA ({reason})",
            f"P&L: ${net:.2f}\nTaxa: ${fee:.2f}",
            level="info"
        )

        self.position = None

    async def _save_checkpoint(self):
        state = {
            'balance': self.risk_manager.current_balance,
            'position': self.position,
            'performance_stats': self.performance_stats,
            'paper_trading': self.paper_trading
        }
        self.checkpoint_manager.save_checkpoint(state)
        self.checkpoint_manager.update_checkpoint_time()

    def _restore_from_checkpoint(self, ckpt: Dict):
        self.risk_manager.current_balance = ckpt.get('balance', self.risk_manager.current_balance)
        self.position = ckpt.get('position', None)
        self.performance_stats = ckpt.get('performance_stats', self.performance_stats)
        logger.info("✅ Estado restaurado do checkpoint")

    async def shutdown(self):
        logger.info("🛑 Desligando sistema...")
        self.is_running = False
        self.ws_manager.stop()
        await self._save_checkpoint()
        await self.alert_system.send_shutdown_alert()
        logger.info("✅ Sistema desligado")

    # ——— Helpers ———

    def _calculate_momentum(self, prices: np.ndarray) -> float:
        p = prices
        n = self.config.momentum_period
        return float((p[-1] - p[-n]) / p[-n]) if p.size >= n else 0.0

    def _calculate_volume_ratio(self, vols: np.ndarray) -> float:
        n = self.config.volume_ratio_period
        return float(vols[-1] / np.mean(vols[-n:])) if vols.size >= n else 1.0

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        n = self.config.volatility_period
        return float(np.std(prices[-n:]) / np.mean(prices[-n:])) if prices.size >= n else 0.01


# Entrypoints

async def run_paper_trading(
    config=None,
    initial_balance: float = 10000.0,
    debug_mode: bool = False
):
    setup_logging()
    cfg = config or get_config(debug_mode=debug_mode)
    system = TradingSystem(cfg, paper_trading=True)
    system.risk_manager.current_balance = initial_balance
    system.risk_manager.initial_balance = initial_balance

    await system.initialize()
    try:
        await system.run()
    except KeyboardInterrupt:
        logger.info("⏹️ Interrompido pelo usuário")
    finally:
        await system.shutdown()


def handle_signals():
    def _handler(sig, frame):
        logger.info(f"Sinal {sig} recebido, finalizando...")
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    handle_signals()
    asyncio.run(run_paper_trading(debug_mode=True))
