"""
Módulo principal para orquestração do sistema de trading
Versão aprimorada com gestão de risco avançada
"""
import os
import time
import asyncio
import signal
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from trade_system.config import TradingConfig
from trade_system.logging_config import setup_logging
from trade_system.cache import CacheManager
from trade_system.rate_limiter import RateLimiter
from trade_system.alerts import AlertManager
from trade_system.websocket_manager import WebSocketManager
from trade_system.analysis.technical import TechnicalAnalyzer
from trade_system.analysis.orderbook import OrderbookAnalyzer
from trade_system.analysis.ml import SimplifiedMLPredictor
from trade_system.risk import RiskManager
from trade_system.validation import MarketConditionValidator
from trade_system.signals import OptimizedSignalConsolidator
from trade_system.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class TradingSystem:
    """Sistema principal de trading com proteções aprimoradas"""

    def __init__(self, config=None, paper_trading: bool = True):
        self.config = config or TradingConfig.from_env()
        self.paper_trading = paper_trading

        # Infraestrutura
        self.cache = CacheManager(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.alert_system = AlertManager(self.config)
        self.ws_manager = WebSocketManager(self.config)

        # Módulos de análise
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.orderbook_analyzer = OrderbookAnalyzer(self.config)
        self.ml_predictor = SimplifiedWebSocketManager()
        self.signal_consolidator = OptimizedSignalConsolidator()

        # Gestão de risco e validação de mercado
        self.risk_manager = RiskManager(self.config)
        self.market_validator = MarketConditionValidator(self.config)

        # Gerenciamento de checkpoint
        self.checkpoint_manager = CheckpointManager()

        # Estado interno
        self.position: Optional[Dict] = None
        self.is_running = False
        self.last_price = None
        self.last_signal_time = 0
        self.signal_cooldown = self.config.get('signal_cooldown', 60)  # 60s entre trades
        
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
        """Loop principal de coleta, análise e execução com proteções aprimoradas"""
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

                # Atualiza último preço
                self.last_price = float(data['prices'][-1])

                # 1) Verificar stop loss/take profit se tem posição aberta
                if hasattr(self.risk_manager, 'has_open_position') and self.risk_manager.has_open_position():
                    await self._check_positions_exit()

                # 2) Validar condições de mercado
                is_safe, reasons = await self.market_validator.validate(data)
                if not is_safe and not self.config.debug_mode:
                    if cycle % log_every == 0:
                        logger.warning(f"⚠️ Mercado inseguro: {', '.join(reasons)}")
                    await asyncio.sleep(0.1)
                    cycle += 1
                    continue

                # 3) Gerar sinais
                signals = await self._analyze_market(data)
                action, confidence = self.signal_consolidator.consolidate(signals)

                # 4) Gerenciar posições com proteções
                if self.position or (hasattr(self.risk_manager, 'has_open_position') and self.risk_manager.has_open_position()):
                    await self._manage_position(data, action, confidence)
                else:
                    # Verificar cooldown entre trades
                    if action != 'HOLD' and confidence >= self.config.min_confidence:
                        current_time = time.time()
                        if current_time - self.last_signal_time >= self.signal_cooldown:
                            await self._open_position(data, action, confidence)
                            self.last_signal_time = current_time
                        else:
                            remaining = self.signal_cooldown - (current_time - self.last_signal_time)
                            if cycle % log_every == 0:
                                logger.debug(f"⏳ Cooldown ativo: {remaining:.0f}s restantes")

                # 5) Checkpoint periódico
                if self.checkpoint_manager.should_checkpoint():
                    await self._save_checkpoint()

                cycle += 1
                await asyncio.sleep(interval_s)

        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}", exc_info=True)
            await self.alert_system.send_alert("Erro no Sistema", str(e), "critical")
        finally:
            await self.shutdown()

    async def _check_positions_exit(self):
        """Verifica condições de saída das posições abertas"""
        position = self.risk_manager.get_open_position()
        if not position or not self.last_price:
            return
        
        exit_reason = self.risk_manager.check_exit_conditions(position, self.last_price)
        
        if exit_reason:
            result = self.risk_manager.close_position(position, self.last_price, exit_reason)
            
            # Atualiza estatísticas
            self.performance_stats['total_trades'] += 1
            if result['pnl'] > 0:
                self.performance_stats['winning_trades'] += 1
            self.performance_stats['total_pnl'] += result['pnl']
            
            logger.info(f"📊 Trade fechado por {exit_reason}")
            logger.info(f"   P&L: ${result['pnl']:.2f} ({result['pnl_pct']:.2f}%)")
            logger.info(f"   Novo balanço: ${result['new_balance']:.2f}")
            
            # Limpa posição local
            self.position = None
            
            # Salva checkpoint
            await self._save_checkpoint()
            
            # Alerta
            await self.alert_system.send_alert(
                f"Posição Fechada - {exit_reason}",
                f"P&L: ${result['pnl']:.2f}\nBalanço: ${result['new_balance']:.2f}",
                "info"
            )

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
        """Dispara abertura de posição com verificações aprimoradas"""
        # Verificação adicional de posição aberta
        if hasattr(self.risk_manager, 'has_open_position') and self.risk_manager.has_open_position():
            logger.warning("⚠️ Tentativa de abrir posição com posição já aberta")
            return
            
        if self.paper_trading:
            await self._open_paper_position(data, action, confidence)
        else:
            logger.error("⚠️ Trading real ainda não implementado")

    async def _open_paper_position(self, data: Dict, action: str, confidence: float):
        """Abre posição simulada com proteções aprimoradas"""
        price = float(data['prices'][-1])
        vol = self._calculate_volatility(data['prices'])
        
        # Usa novo método se disponível
        if hasattr(self.risk_manager, 'calculate_position_size'):
            position_info = self.risk_manager.calculate_position_size(price, confidence, vol)
            value = position_info['size']
            qty = position_info['quantity']
        else:
            value = self.risk_manager.calculate_position_size(confidence, vol, price)
            qty = value / price
            
        if value <= 0:
            return

        fee = value * self.config.fee_rate  # ex: 0.001
        
        # Cria posição com novo sistema se disponível
        if hasattr(self.risk_manager, 'open_position'):
            position = self.risk_manager.open_position(
                side=action,
                price=price,
                quantity=qty,
                size=value,
                current_time=time.time()
            )
            
            if not position:
                logger.warning("⚠️ Não foi possível abrir posição")
                return
                
            self.position = {
                'side': action,
                'entry_price': price,
                'quantity': qty,
                'entry_time': time.time(),
                'confidence': confidence,
                'entry_fee': fee,
                'paper_trade': True,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            }
        else:
            # Fallback para sistema antigo
            self.position = {
                'side': action,
                'entry_price': price,
                'quantity': qty,
                'entry_time': time.time(),
                'confidence': confidence,
                'entry_fee': fee,
                'paper_trade': True,
            }
            self.risk_manager.set_position(self.position)
            
        self.risk_manager.current_balance -= fee

        logger.info(f"🟢 POSIÇÃO ABERTA [{action}] price=${price:.2f} size=${value:.2f} qty={qty:.6f} fee=${fee:.2f}")
        
        # Log stop loss e take profit se disponível
        if hasattr(self.position, 'stop_loss') or 'stop_loss' in self.position:
            sl = self.position.get('stop_loss', 0)
            tp = self.position.get('take_profit', 0)
            if sl and tp:
                logger.info(f"   Stop Loss: ${sl:.2f} | Take Profit: ${tp:.2f}")
        
        await self.alert_system.send_alert(
            f"Nova Posição {action}",
            f"Preço: ${price:.2f}\nValor: ${value:.2f}\nConf: {confidence:.1%}",
            "info"
        )

    async def _manage_position(self, data: Dict, action: str, confidence: float):
        """Verifica se deve fechar a posição atual com lógica aprimorada"""
        price = float(data['prices'][-1])
        
        # Se tem o novo sistema, verifica primeiro
        if hasattr(self.risk_manager, 'has_open_position') and self.risk_manager.has_open_position():
            position = self.risk_manager.get_open_position()
            
            # Só processa sinal contrário
            if (position.side == 'BUY' and action == 'SELL' and confidence >= self.config.min_confidence) or \
               (position.side == 'SELL' and action == 'BUY' and confidence >= self.config.min_confidence):
                # Fecha por reversão de sinal
                result = self.risk_manager.close_position(position, price, 'SIGNAL_REVERSAL')
                
                # Atualiza estatísticas
                self.performance_stats['total_trades'] += 1
                if result['pnl'] > 0:
                    self.performance_stats['winning_trades'] += 1
                self.performance_stats['total_pnl'] += result['pnl']
                
                logger.info(f"🔄 Fechando posição por reversão de sinal")
                self.position = None
                
                # Espera um pouco antes de abrir nova posição
                await asyncio.sleep(2)
                
                # Pode abrir nova posição se configurado
                if self.config.get('allow_reversal_trades', True):
                    await self._open_position(data, action, confidence)
            else:
                # Ignora sinal na mesma direção
                if action != 'HOLD':
                    logger.debug(f"⏭️ Ignorando sinal {action} - posição {position.side} aberta")
        
        elif self.position:
            # Fallback para sistema antigo
            should_close, reason = self.risk_manager.should_close_position(
                price,
                self.position['entry_price'],
                self.position['side']
            )
            if should_close:
                await self._close_position(price, reason)

    async def _close_position(self, exit_price: float, reason: str):
        """Fecha posição simulada, atualiza P&L e risk metrics"""
        if not self.position:
            return
            
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

        logger.info(f"🔴 POSIÇÃO FECHADA exit=${exit_price:.2f} P&L=${pnl_net:.2f} fees=${total_fees:.2f} reason={reason}")
        
        # Calcula win rate
        win_rate = (self.performance_stats['winning_trades'] / self.performance_stats['total_trades'] * 100) if self.performance_stats['total_trades'] > 0 else 0
        logger.info(f"   Win Rate: {win_rate:.1f}% | Total P&L: ${self.performance_stats['total_pnl']:.2f}")
        
        await self.alert_system.send_alert(
            "Posição Fechada",
            f"P&L: ${pnl_net:.2f}\nFees: ${total_fees:.2f}\nMotivo: {reason}\nWin Rate: {win_rate:.1f}%",
            "info"
        )

    async def _save_checkpoint(self):
        """Salva checkpoint com informações adicionais"""
        state = {
            'balance': self.risk_manager.current_balance,
            'position': self.position,
            'performance_stats': self.performance_stats,
            'paper_trading': self.paper_trading,
            'last_signal_time': self.last_signal_time,
            # Adiciona estatísticas do risk manager se disponível
            'risk_stats': {
                'daily_pnl': getattr(self.risk_manager, 'daily_pnl', 0),
                'total_trades': getattr(self.risk_manager, 'total_trades', 0),
                'winning_trades': getattr(self.risk_manager, 'winning_trades', 0),
                'losing_trades': getattr(self.risk_manager, 'losing_trades', 0)
            } if hasattr(self.risk_manager, 'daily_pnl') else {}
        }
        self.checkpoint_manager.save_checkpoint(state)
        self.checkpoint_manager.update_checkpoint_time()
        logger.debug("✅ Checkpoint salvo")

    def _restore_from_checkpoint(self, ckpt: Dict):
        """Restaura estado do checkpoint com informações adicionais"""
        self.risk_manager.current_balance = ckpt.get('balance', self.risk_manager.initial_balance)
        self.position = ckpt.get('position')
        self.performance_stats = ckpt.get('performance_stats', self.performance_stats)
        self.last_signal_time = ckpt.get('last_signal_time', 0)
        
        # Restaura estatísticas do risk manager se disponível
        risk_stats = ckpt.get('risk_stats', {})
        if risk_stats and hasattr(self.risk_manager, 'daily_pnl'):
            self.risk_manager.daily_pnl = risk_stats.get('daily_pnl', 0)
            self.risk_manager.total_trades = risk_stats.get('total_trades', 0)
            self.risk_manager.winning_trades = risk_stats.get('winning_trades', 0)
            self.risk_manager.losing_trades = risk_stats.get('losing_trades', 0)
        
        # Restaura posição no risk manager se disponível
        if self.position and hasattr(self.risk_manager, 'positions'):
            # Recria objeto Position se necessário
            from trade_system.risk import Position
            pos = Position(
                side=self.position['side'],
                entry_price=self.position['entry_price'],
                quantity=self.position['quantity'],
                size=self.position.get('size', self.position['quantity'] * self.position['entry_price']),
                entry_time=self.position.get('entry_time', time.time()),
                stop_loss=self.position.get('stop_loss'),
                take_profit=self.position.get('take_profit')
            )
            self.risk_manager.positions = [pos]
        
        logger.info("✅ Estado restaurado do checkpoint")
        logger.info(f"   Balanço: ${self.risk_manager.current_balance:.2f}")
        logger.info(f"   Trades: {self.performance_stats['total_trades']}")
        if self.position:
            logger.info(f"   Posição aberta: {self.position['side']} @ ${self.position['entry_price']:.2f}")

    async def shutdown(self):
        """Encerra sistema com segurança aprimorada"""
        logger.info("🛑 Desligando sistema...")
        self.is_running = False
        
        # Fecha posições abertas se configurado
        if self.config.get('close_on_shutdown', False) and self.last_price:
            if hasattr(self.risk_manager, 'has_open_position') and self.risk_manager.has_open_position():
                position = self.risk_manager.get_open_position()
                result = self.risk_manager.close_position(position, self.last_price, 'SYSTEM_SHUTDOWN')
                logger.info(f"📊 Posição fechada no desligamento - P&L: ${result['pnl']:.2f}")
            elif self.position:
                await self._close_position(self.last_price, 'SYSTEM_SHUTDOWN')
        
        self.ws_manager.stop()
        await self._save_checkpoint()
        
        # Log estatísticas finais
        total_return = ((self.risk_manager.current_balance - self.performance_stats['start_balance']) / 
                       self.performance_stats['start_balance'] * 100)
        win_rate = (self.performance_stats['winning_trades'] / self.performance_stats['total_trades'] * 100) if self.performance_stats['total_trades'] > 0 else 0
        
        logger.info("📊 Estatísticas da Sessão:")
        logger.info(f"   Total de trades: {self.performance_stats['total_trades']}")
        logger.info(f"   Win rate: {win_rate:.1f}%")
        logger.info(f"   P&L total: ${self.performance_stats['total_pnl']:.2f}")
        logger.info(f"   Fees totais: ${self.performance_stats['total_fees']:.2f}")
        logger.info(f"   Retorno: {total_return:.2f}%")
        logger.info(f"   Balanço final: ${self.risk_manager.current_balance:.2f}")
        
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
    """Ponto de entrada para paper trading com configurações aprimoradas"""
    setup_logging()
    cfg = config or TradingConfig.from_env()
    cfg.debug_mode = debug_mode

    # Adiciona configurações de proteção se não existirem
    if not hasattr(cfg, 'signal_cooldown'):
        cfg.signal_cooldown = 60  # 60s entre trades
    if not hasattr(cfg, 'allow_reversal_trades'):
        cfg.allow_reversal_trades = True
    if not hasattr(cfg, 'close_on_shutdown'):
        cfg.close_on_shutdown = False

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
