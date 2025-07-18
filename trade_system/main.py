"""
Módulo principal para orquestração do sistema de trading
"""
import os
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

        # Infraestrutura
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

        # Gerenciamento de checkpoint
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

        self.ws_manager.start_delayed()
        await self.alert_system.send_startup_alert("PAPER" if self.paper_trading else "LIVE")

        logger.info("⏳ Aguardando dados do WebSocket...")
        for _ in range(50):
            await asyncio.sleep(0.1)
            if self.ws_manager.buffer_filled or self.ws_manager.buffer_index > 100:
                logger.info("✅ Dados recebidos!")
                break

    async def run(self):
        """Loop principal de coleta, análise e execução"""
        self.is_running = True
        cycle = 0
        interval_s = self.config.main_loop_interval_ms / 1000.0
        log_every = max(1, int(10 / interval_s))  # a cada ~10s

        try:
            while self.is_running:
                data = self.ws_manager.get_latest_data()
                if not data:
                    await asyncio.sleep(0.1)
                    continue

                # 1) Validar condições de mercado
                is_safe, reasons = await self.market_validator.validate(data)
                if not is_safe and not self.config.debug_mode:
                    if cycle % log_every == 0:
                        logger.warning(f"⚠️ Mercado inseguro: {', '.join(reasons)}")
                    await asyncio.sleep(0.1)
                    cycle += 1
                    continue

                # 2) Gerar sinais
                signals = await self._analyze_market(data)
                action, confidence = self.signal_consolidator.consolidate(signals)

                # 3) Gerenciar posições
                if self.position:
                    await self._manage_position(data, action, confidence)
                else:
                    if action != 'HOLD' and confidence >= self.config.min_confidence:
                        await self._open_position(data, action, confidence)

                # 4) Checkpoint periódico
                if self.checkpoint_manager.should_checkpoint():
                    await self._save_checkpoint()

                cycle += 1
                await asyncio.sleep(interval_s)

        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}", exc_info=True)
            await self.alert_system.send_alert("Erro no Sistema", str(e), "critical")
        finally:
            await self.shutdown()

    async def _analyze_market(self, market_data: Dict) -> List[Tuple[str, str, float]]:
        """Executa as três camadas de análise e retorna a lista de sinais"""
        # Técnica
        tech_act, tech_conf, tech_det = self.technical_analyzer.analyze(
            market_data['prices'], market_data['volumes']
        )
        # Orderbook
        ob_act, ob_conf, ob_det = self.orderbook_analyzer.analyze(
            market_data['orderbook_bids'],
            market_data['orderbook_asks'],
            self.cache
        )
        # ML
        features = {
            'rsi': tech_det.get('rsi', 50.0),
            'momentum': self._calculate_momentum(market_data['prices']),
            'volume_ratio': self._calculate_volume_ratio(market_data['volumes']),
            'spread_bps': ob_det.get('spread_bps', 0.0),
            'volatility': self._calculate_volatility(market_data['prices']),
        }
        ml_act, ml_conf = self.ml_predictor.predict(features)

        return [
            ('technical', tech_act, tech_conf),
            ('orderbook', ob_act, ob_conf),
            ('ml', ml_act, ml_conf),
        ]

    async def _open_position(self, data: Dict, action: str, confidence: float):
        """Dispara abertura de posição (paper ou live)"""
        if self.paper_trading:
            await self._open_paper_position(data, action, confidence)
        else:
            logger.error("⚠️ Trading real ainda não implementado")

    async def _open_paper_position(self, data: Dict, action: str, confidence: float):
        """Abre posição simulada e registra no RiskManager"""
        price = float(data['prices'][-1])
        vol = self._calculate_volatility(data['prices'])
        value = self.risk_manager.calculate_position_size(confidence, vol, price)
        if value <= 0:
            return

        fee = value * self.config.fee_rate  # ex: 0.001
        qty = value / price

        self.position = {
            'side': action,
            'entry_price': price,
            'quantity': qty,
            'entry_time': datetime.now(),
            'confidence': confidence,
            'entry_fee': fee,
            'paper_trade': True,
        }
        self.risk_manager.set_position(self.position)
        self.risk_manager.current_balance -= fee

        logger.info(f"🟢 POSIÇÃO ABERTA [{action}]  price=${price:.2f} size=${value:.2f} qty={qty:.6f} fee=${fee:.2f}")
        await self.alert_system.send_alert(f"Nova Posição {action}",
                                          f"Preço: ${price:.2f}\nValor: ${value:.2f}\nConf: {confidence:.1%}",
                                          "info")

    async def _manage_position(self, data: Dict, action: str, confidence: float):
        """Verifica se deve fechar a posição atual"""
        price = float(data['prices'][-1])
        should_close, reason = self.risk_manager.should_close_position(
            price,
            self.position['entry_price'],
            self.position['side']
        )
        if should_close:
            await self._close_position(price, reason)

    async def _close_position(self, exit_price: float, reason: str):
        """Fecha posição simulada, atualiza P&L e risk metrics"""
        entry_price = self.position['entry_price']
        qty = self.position['quantity']
        side = self.position['side']

        pnl = (exit_price - entry_price) * qty if side == 'BUY' else (entry_price - exit_price) * qty
        exit_fee = exit_price * qty * self.config.fee_rate
        pnl_net = pnl - exit_fee
        total_fees = self.position['entry_fee'] + exit_fee

        # Estatísticas
        self.performance_stats['total_trades'] += 1
        if pnl_net > 0:
            self.performance_stats['winning_trades'] += 1
        self.performance_stats['total_pnl'] += pnl_net
        self.performance_stats['total_fees'] += total_fees

        # Atualiza risk manager
        self.risk_manager.update_after_trade(pnl_net, exit_fee)
        self.risk_manager.clear_position()
        self.position = None

        logger.info(f"🔴 POSIÇÃO FECHADA  exit=${exit_price:.2f} P&L=${pnl_net:.2f} fees=${total_fees:.2f} reason={reason}")
        await self.alert_system.send_alert("Posição Fechada",
                                          f"P&L: ${pnl_net:.2f}\nFees: ${total_fees:.2f}\nMotivo: {reason}",
                                          "info")

    async def _save_checkpoint(self):
        state = {
            'balance': self.risk_manager.current_balance,
            'position': self.position,
            'performance_stats': self.performance_stats,
            'paper_trading': self.paper_trading
        }
        self.checkpoint_manager.save_checkpoint(state)
        self.checkpoint_manager.update_checkpoint_time()
        logger.debug("✅ Checkpoint salvo")

    def _restore_from_checkpoint(self, ckpt: Dict):
        self.risk_manager.current_balance = ckpt.get('balance', self.risk_manager.initial_balance)
        self.position = ckpt.get('position')
        self.performance_stats = ckpt.get('performance_stats', self.performance_stats)
        logger.info("✅ Estado restaurado do checkpoint")

    async def shutdown(self):
        """Encerra sistema, salva checkpoint final e envia alerta"""
        logger.info("🛑 Desligando sistema...")
        self.is_running = False
        self.ws_manager.stop()
        await self._save_checkpoint()
        await self.alert_system.send_shutdown_alert()
        logger.info("✅ Sistema desligado")

    # Métodos utilitários de cálculo de features
    def _calculate_momentum(self, prices: List[float]) -> float:
        if len(prices) < 20:
            return 0.0
        return (prices[-1] - prices[-20]) / prices[-20]

    def _calculate_volume_ratio(self, volumes: List[float]) -> float:
        if len(volumes) < 20:
            return 1.0
        avg = np.mean(volumes[-20:])
        return float(volumes[-1] / avg) if avg > 0 else 1.0

    def _calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 50:
            return 0.01
        arr = np.array(prices[-50:], dtype=np.float64)
        return float(np.std(arr) / np.mean(arr))


async def run_paper_trading(
    config=None,
    initial_balance: float = 10000.0,
    debug_mode: bool = False
):
    """Ponto de entrada para paper trading"""
    setup_logging()
    cfg = config or get_config(debug_mode=debug_mode)

    system = TradingSystem(cfg, paper_trading=True)
    # ajusta balance inicial
    system.risk_manager.current_balance = initial_balance
    system.risk_manager.initial_balance = initial_balance

    await system.initialize()
    try:
        await system.run()
    except KeyboardInterrupt:
        logger.info("⏹️ Interrompido pelo usuário")
        await system.shutdown()


def handle_signals():
    """Configura tratamento de SIGINT/SIGTERM"""
    def _handler(sig, frame):
        logger.info(f"Sinal {sig} recebido, encerrando...")
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    handle_signals()
    asyncio.run(run_paper_trading(debug_mode=True))
