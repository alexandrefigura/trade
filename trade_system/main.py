"""
Módulo principal para orquestração do sistema de trading
"""
import os
import asyncio
import signal
from datetime import datetime
import numpy as np              # <— import adicionado para cálculos
from typing import Optional, Dict, List, Tuple

from trade_system.config import get_config
from trade_system.logging_config import setup_logging, get_logger
from trade_system.cache import UltraFastCache
from trade_system.rate_limiter import RateLimiter
from trade_system.alerts import AlertSystem
from trade_system.websocket_manager import UltraFastWebSocketManager
from trade_system.analysis.ultrafast_technical import UltraFastTechnicalAnalysis
from trade_system.analysis.orderbook import ParallelOrderbookAnalyzer
from trade_system.analysis.ml import SimplifiedMLPredictor
from trade_system.risk import UltraFastRiskManager, MarketConditionValidator
from trade_system.signals import OptimizedSignalConsolidator
from trade_system.checkpoint import CheckpointManager
from trade_system.backtester import run_backtest_validation

logger = get_logger(__name__)


class TradingSystem:
    """Sistema principal de trading"""
    
    def __init__(self, config=None, paper_trading: bool = True):
        self.config = config or get_config()
        self.paper_trading = paper_trading

        # Componentes de infra
        self.cache = UltraFastCache(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.alert_system = AlertSystem(self.config)
        self.ws_manager = UltraFastWebSocketManager(self.config, self.cache)

        # Módulos de análise
        self.technical_analyzer = UltraFastTechnicalAnalysis(self.config)
        self.orderbook_analyzer = ParallelOrderbookAnalyzer(self.config)
        self.ml_predictor = SimplifiedMLPredictor()
        self.signal_consolidator = OptimizedSignalConsolidator()

        # Gestão de risco e validação de mercado
        self.risk_manager = UltraFastRiskManager(self.config)
        self.market_validator = MarketConditionValidator(self.config)

        # Checkpoint
        self.checkpoint_manager = CheckpointManager()

        # Estado interno
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
        """Inicializa componentes assíncronos e carrega checkpoint"""
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            self._restore_from_checkpoint(checkpoint)

        # Startup do WebSocket
        self.ws_manager.start_delayed()

        # Envia alerta de início
        await self.alert_system.send_startup_alert(
            mode="PAPER" if self.paper_trading else "LIVE"
        )

        # Aguarda buffer inicial
        logger.info("⏳ Aguardando dados do WebSocket...")
        for _ in range(50):
            await asyncio.sleep(0.1)
            if self.ws_manager.buffer_filled or self.ws_manager.buffer_index > 100:
                logger.info("✅ Dados recebidos!")
                break

    async def run(self):
        """Loop principal de trading"""
        self.is_running = True
        cycle = 0

        try:
            while self.is_running:
                market_data = self.ws_manager.get_latest_data()
                if not market_data:
                    await asyncio.sleep(0.1)
                    continue

                # Validação de condições de mercado
                is_safe, reasons = await self.market_validator.validate(
                    market_data, client=self.ws_manager.client
                )
                if not is_safe and not self.config.debug_mode:
                    if cycle % (int(10_000 / self.config.main_loop_interval_ms)) == 0:
                        logger.warning(f"⚠️ Mercado inseguro: {', '.join(reasons)}")
                    await asyncio.sleep(0.1)
                    cycle += 1
                    continue

                # Análises (TA, orderbook, ML)
                composite_signals = await self._analyze_market(market_data)

                # Consolidação de sinais
                action, confidence = self.signal_consolidator.consolidate(composite_signals)

                # Gestão de posições
                if self.position:
                    await self._manage_position(market_data, action, confidence)
                else:
                    if action != 'HOLD' and confidence >= self.config.min_confidence:
                        await self._open_position(market_data, action, confidence)

                # Checkpoint periódico
                if self.checkpoint_manager.should_checkpoint():
                    await self._save_checkpoint()

                cycle += 1
                await asyncio.sleep(self.config.main_loop_interval_ms / 1000.0)

        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}", exc_info=True)
            await self.alert_system.send_alert("Erro no Sistema", str(e), level="critical")
        finally:
            await self.shutdown()

    async def _analyze_market(self, market_data: Dict) -> List[Tuple[str, str, float]]:
        """Executa TA, análise de orderbook e ML, retornando lista de (origem, ação, confiança)"""
        prices = np.array(market_data['prices'], dtype=np.float64)
        volumes = np.array(market_data['volumes'], dtype=np.float64)

        # 1. Técnica
        ta_action, ta_conf, ta_details = self.technical_analyzer.analyze(prices, volumes)

        # 2. Orderbook
        ob_action, ob_conf, ob_details = self.orderbook_analyzer.analyze(
            market_data['orderbook_bids'],
            market_data['orderbook_asks'],
            self.cache
        )

        # 3. ML
        features = {
            'rsi': ta_details.get('rsi', 50.0),
            'momentum': self._calculate_momentum(prices),
            'volume_ratio': self._calculate_volume_ratio(volumes),
            'spread_bps': ob_details.get('spread_bps', 0.0),
            'volatility': self._calculate_volatility(prices)
        }
        ml_action, ml_conf = self.ml_predictor.predict(features)

        return [
            ('technical', ta_action, ta_conf),
            ('orderbook', ob_action, ob_conf),
            ('ml', ml_action, ml_conf),
        ]

    async def _open_position(self, market_data: Dict, action: str, confidence: float):
        """Roteia abertura de posição (paper ou live)"""
        if self.paper_trading:
            await self._open_paper_position(market_data, action, confidence)
        else:
            logger.error("🔴 Trading real ainda não implementado")

    async def _open_paper_position(self, market_data: Dict, action: str, confidence: float):
        """Abre posição simulada (paper)"""
        price = float(market_data['prices'][-1])
        volatility = self._calculate_volatility(np.array(market_data['prices'], float))

        size_usd = self.risk_manager.calculate_position_size(confidence, volatility, price)
        if size_usd <= 0:
            return

        entry_fee = size_usd * self.config.trade_fee_pct
        qty = size_usd / price

        # Monta posição
        self.position = {
            'side': action,
            'entry_price': price,
            'quantity': qty,
            'entry_time': datetime.now(),
            'confidence': confidence,
            'entry_fee': entry_fee,
        }

        # Atualiza risco
        self.risk_manager.set_position(self.position)
        self.risk_manager.current_balance -= entry_fee

        logger.info(f"🟢 POSIÇÃO ABERTA [{action}] @ ${price:.2f} x {qty:.6f} = ${size_usd:.2f}")
        await self.alert_system.send_alert(
            f"POSIÇÃO ABERTA {action}",
            f"Preço: ${price:.2f}\nQuantidade: {qty:.6f}\nValor: ${size_usd:.2f}",
            level="info"
        )

    async def _manage_position(self, market_data: Dict, action: str, confidence: float):
        """Verifica se deve fechar posição atual"""
        price = float(market_data['prices'][-1])
        should_close, reason = self.market_validator.validate  # note: validate called internally in run
        close, reason = self.risk_manager.should_close_position(
            price,
            self.position['entry_price'],
            side=self.position['side']
        )
        if close:
            await self._close_position(price, reason)

    async def _close_position(self, exit_price: float, reason: str):
        """Fecha posição (paper) e atualiza estatísticas e risco"""
        if not self.position:
            return

        side = self.position['side']
        entry = self.position['entry_price']
        qty = self.position['quantity']

        pnl = (exit_price - entry) * qty if side == 'BUY' else (entry - exit_price) * qty
        exit_fee = exit_price * qty * self.config.trade_fee_pct
        net_pnl = pnl - exit_fee

        # Stats
        self.performance_stats['total_trades'] += 1
        if net_pnl > 0:
            self.performance_stats['winning_trades'] += 1
        self.performance_stats['total_pnl'] += net_pnl
        self.performance_stats['total_fees'] += self.position['entry_fee'] + exit_fee

        # Risco
        self.risk_manager.update_after_trade(net_pnl, exit_fee)
        self.risk_manager.clear_position()

        logger.info(f"🔴 POSIÇÃO FECHADA ({reason}) P&L: ${net_pnl:.2f} Taxa: ${exit_fee:.2f}")
        await self.alert_system.send_alert(
            f"POSIÇÃO FECHADA ({reason})",
            f"P&L: ${net_pnl:.2f}\nTaxa: ${exit_fee:.2f}",
            level="info"
        )

        self.position = None

    async def _save_checkpoint(self):
        """Salva estado atual em checkpoint"""
        state = {
            'balance': self.risk_manager.current_balance,
            'position': self.position,
            'performance_stats': self.performance_stats,
            'paper_trading': self.paper_trading
        }
        self.checkpoint_manager.save_checkpoint(state)
        self.checkpoint_manager.update_checkpoint_time()

    def _restore_from_checkpoint(self, ckpt: Dict):
        """Restaura estado a partir de checkpoint"""
        self.risk_manager.current_balance = ckpt.get('balance', self.risk_manager.current_balance)
        self.position = ckpt.get('position', None)
        self.performance_stats = ckpt.get('performance_stats', self.performance_stats)
        logger.info("✅ Estado restaurado do checkpoint")

    async def shutdown(self):
        """Desliga sistema, salva checkpoint final e envia alerta"""
        logger.info("🛑 Desligando sistema...")
        self.is_running = False
        self.ws_manager.stop()
        await self._save_checkpoint()
        await self.alert_system.send_shutdown_alert()
        logger.info("✅ Sistema desligado")

    # ——— Helpers de feature engineering ———

    def _calculate_momentum(self, prices: np.ndarray) -> float:
        if prices.size < self.config.momentum_period:
            return 0.0
        return float((prices[-1] - prices[-self.config.momentum_period]) / prices[-self.config.momentum_period])

    def _calculate_volume_ratio(self, volumes: np.ndarray) -> float:
        if volumes.size < self.config.volume_ratio_period:
            return 1.0
        avg = float(np.mean(volumes[-self.config.volume_ratio_period:]))
        return float(volumes[-1] / avg) if avg > 0 else 1.0

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        if prices.size < self.config.volatility_period:
            return 0.01
        return float(np.std(prices[-self.config.volatility_period:]) / np.mean(prices[-self.config.volatility_period:]))

# ——— Entrypoints ———

async def run_paper_trading(
    config=None,
    initial_balance: float = 10000.0,
    debug_mode: bool = False
):
    """
    Executa o sistema em modo PAPER TRADING
    """
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
    """Configura handlers para SIGINT e SIGTERM"""
    def _handler(sig, frame):
        logger.info(f"Sinal {sig} recebido, finalizando...")
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    handle_signals()
    asyncio.run(run_paper_trading(debug_mode=True))
