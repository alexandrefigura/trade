"""
Módulo principal para orquestração do sistema de trading
"""
import os
import asyncio
import signal
from datetime import datetime
from typing import Optional
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
from trade_system.backtester import run_backtest_validation

logger = get_logger(__name__)


class TradingSystem:
    """Sistema principal de trading"""
    
    def __init__(self, config=None, paper_trading=True):
        """
        Inicializa o sistema de trading
        
        Args:
            config: Configuração do sistema
            paper_trading: Se True, executa em modo simulado
        """
        self.config = config or get_config()
        self.paper_trading = paper_trading
        
        # Componentes principais
        self.cache = UltraFastCache(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.alert_system = AlertSystem(self.config)
        self.ws_manager = UltraFastWebSocketManager(self.config, self.cache)
        
        # Análise
        self.technical_analyzer = UltraFastTechnicalAnalysis(self.config)
        self.orderbook_analyzer = ParallelOrderbookAnalyzer(self.config)
        self.ml_predictor = SimplifiedMLPredictor()
        self.signal_consolidator = OptimizedSignalConsolidator()
        
        # Risk e validação
        self.risk_manager = UltraFastRiskManager(self.config)
        self.market_validator = MarketConditionValidator(self.config)
        
        # Checkpoint
        self.checkpoint_manager = CheckpointManager()
        
        # Estado
        self.position = None
        self.is_running = False
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'start_balance': self.risk_manager.current_balance,
            'session_start': datetime.now()
        }
        
        logger.info(f"🚀 Sistema inicializado - Modo: {'PAPER TRADING' if paper_trading else 'LIVE'}")
    
    async def initialize(self):
        """Inicializa componentes assíncronos"""
        # Carregar checkpoint se existir
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            self._restore_from_checkpoint(checkpoint)
        
        # Iniciar WebSocket
        self.ws_manager.start_delayed()
        
        # Alerta de início
        await self.alert_system.send_startup_alert(
            mode="PAPER" if self.paper_trading else "LIVE"
        )
        
        # Aguardar dados
        logger.info("⏳ Aguardando dados do WebSocket...")
        for i in range(50):
            await asyncio.sleep(0.1)
            if self.ws_manager.buffer_filled or self.ws_manager.buffer_index > 100:
                logger.info("✅ Dados recebidos!")
                break
    
    async def run(self):
        """Loop principal do sistema"""
        self.is_running = True
        cycle_count = 0
        
        try:
            while self.is_running:
                # Obter dados
                market_data = self.ws_manager.get_latest_data()
                if not market_data:
                    await asyncio.sleep(0.1)
                    continue
                
                # Validar mercado
                is_safe, reasons = await self.market_validator.validate_market_conditions(
                    market_data
                )
                
                if not is_safe and not self.config.debug_mode:
                    if cycle_count % 100 == 0:  # Log a cada 10s
                        logger.warning(f"⚠️ Mercado inseguro: {', '.join(reasons)}")
                    await asyncio.sleep(0.1)
                    continue
                
                # Analisar mercado
                signals = await self._analyze_market(market_data)
                
                # Consolidar sinais
                action, confidence = self.signal_consolidator.consolidate(signals)
                
                # Gerenciar posições
                if self.position:
                    await self._manage_position(market_data, action, confidence)
                else:
                    if action != 'HOLD' and confidence >= self.config.min_confidence:
                        await self._open_position(market_data, action, confidence)
                
                # Checkpoint periódico
                if self.checkpoint_manager.should_checkpoint():
                    await self._save_checkpoint()
                
                # Incrementar contador
                cycle_count += 1
                
                # Sleep
                await asyncio.sleep(self.config.main_loop_interval_ms / 1000)
                
        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}", exc_info=True)
            await self.alert_system.send_alert(
                "Erro no Sistema",
                str(e),
                "critical"
            )
        finally:
            await self.shutdown()
    
    async def _analyze_market(self, market_data):
        """Analisa mercado e retorna sinais"""
        # Análise técnica
        tech_action, tech_conf, tech_details = self.technical_analyzer.analyze(
            market_data['prices'],
            market_data['volumes']
        )
        
        # Análise orderbook
        ob_action, ob_conf, ob_details = self.orderbook_analyzer.analyze(
            market_data['orderbook_bids'],
            market_data['orderbook_asks'],
            self.cache
        )
        
        # ML
        features = {
            'rsi': tech_details.get('rsi', 50),
            'momentum': self._calculate_momentum(market_data['prices']),
            'volume_ratio': self._calculate_volume_ratio(market_data['volumes']),
            'spread_bps': ob_details.get('spread_bps', 10),
            'volatility': self._calculate_volatility(market_data['prices'])
        }
        
        ml_action, ml_conf = self.ml_predictor.predict(features)
        
        return [
            ('technical', tech_action, tech_conf),
            ('orderbook', ob_action, ob_conf),
            ('ml', ml_action, ml_conf)
        ]
    
    async def _open_position(self, market_data, action, confidence):
        """Abre nova posição"""
        if self.paper_trading:
            await self._open_paper_position(market_data, action, confidence)
        else:
            # TODO: Implementar trading real
            logger.error("Trading real ainda não implementado")
    
    async def _open_paper_position(self, market_data, action, confidence):
        """Abre posição simulada"""
        current_price = float(market_data['prices'][-1])
        volatility = self._calculate_volatility(market_data['prices'])
        
        # Calcular tamanho
        position_size = self.risk_manager.calculate_position_size(
            confidence, volatility, current_price
        )
        
        if position_size == 0:
            return
        
        # Calcular taxas
        entry_fee = position_size * 0.001
        
        # Criar posição
        self.position = {
            'side': action,
            'entry_price': current_price,
            'quantity': position_size / current_price,
            'entry_time': datetime.now(),
            'confidence': confidence,
            'entry_fee': entry_fee,
            'paper_trade': True
        }
        
        # Atualizar risk manager
        self.risk_manager.set_position_info(self.position)
        self.risk_manager.current_balance -= entry_fee
        
        logger.info(f"""
🟢 POSIÇÃO ABERTA [{action}]
- Preço: ${current_price:,.2f}
- Quantidade: {self.position['quantity']:.6f}
- Valor: ${position_size:,.2f}
- Taxa: ${entry_fee:.2f}
- Confiança: {confidence*100:.1f}%
        """)
        
        await self.alert_system.send_alert(
            f"Nova Posição {action}",
            f"Preço: ${current_price:,.2f}\nValor: ${position_size:,.2f}",
            "info"
        )
    
    async def _manage_position(self, market_data, signal_action, signal_confidence):
        """Gerencia posição existente"""
        current_price = float(market_data['prices'][-1])
        
        should_close, reason = self.risk_manager.should_close_position(
            current_price,
            self.position['entry_price'],
            self.position['side']
        )
        
        if should_close:
            await self._close_position(current_price, reason)
    
    async def _close_position(self, exit_price, reason):
        """Fecha posição"""
        if not self.position:
            return
        
        # Calcular resultado
        entry_price = self.position['entry_price']
        quantity = self.position['quantity']
        side = self.position['side']
        
        if side == 'BUY':
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        # Taxas
        exit_fee = exit_price * quantity * 0.001
        total_fees = self.position['entry_fee'] + exit_fee
        pnl_net = pnl - exit_fee
        
        # Atualizar estatísticas
        self.performance_stats['total_trades'] += 1
        if pnl_net > 0:
            self.performance_stats['winning_trades'] += 1
        self.performance_stats['total_pnl'] += pnl_net
        self.performance_stats['total_fees'] += total_fees
        
        # Atualizar risk manager
        self.risk_manager.update_pnl(pnl_net, exit_fee)
        self.risk_manager.clear_position()
        
        logger.info(f"""
🔴 POSIÇÃO FECHADA
- Motivo: {reason}
- Saída: ${exit_price:,.2f}
- P&L: ${pnl_net:,.2f}
- Taxas: ${total_fees:.2f}
        """)
        
        self.position = None
    
    async def _save_checkpoint(self):
        """Salva checkpoint do sistema"""
        state = {
            'balance': self.risk_manager.current_balance,
            'position': self.position,
            'performance_stats': self.performance_stats,
            'paper_trading': self.paper_trading
        }
        
        self.checkpoint_manager.save_checkpoint(state)
        self.checkpoint_manager.update_checkpoint_time()
    
    def _restore_from_checkpoint(self, checkpoint):
        """Restaura estado do checkpoint"""
        self.risk_manager.current_balance = checkpoint.get('balance', 10000)
        self.position = checkpoint.get('position')
        self.performance_stats = checkpoint.get('performance_stats', self.performance_stats)
        
        logger.info("✅ Estado restaurado do checkpoint")
    
    async def shutdown(self):
        """Desliga o sistema"""
        logger.info("🛑 Desligando sistema...")
        self.is_running = False
        
        # Parar WebSocket
        self.ws_manager.stop()
        
        # Salvar checkpoint final
        await self._save_checkpoint()
        
        # Alerta de desligamento
        await self.alert_system.send_shutdown_alert()
        
        logger.info("✅ Sistema desligado")
    
    # Métodos auxiliares
    def _calculate_momentum(self, prices):
        if len(prices) < 20:
            return 0
        return (prices[-1] - prices[-20]) / prices[-20]
    
    def _calculate_volume_ratio(self, volumes):
        if len(volumes) < 20:
            return 1
        avg = np.mean(volumes[-20:])
        return volumes[-1] / avg if avg > 0 else 1
    
    def _calculate_volatility(self, prices):
        if len(prices) < 50:
            return 0.01
        return np.std(prices[-50:]) / np.mean(prices[-50:])


async def run_paper_trading(
    config=None,
    initial_balance=10000,
    debug_mode=False
):
    """
    Executa paper trading
    
    Args:
        config: Configuração do sistema
        initial_balance: Balance inicial
        debug_mode: Modo debug
    """
    # Setup
    setup_logging()
    
    if config is None:
        config = get_config(debug_mode=debug_mode)
    
    # Criar sistema
    system = TradingSystem(config, paper_trading=True)
    system.risk_manager.current_balance = initial_balance
    system.risk_manager.initial_balance = initial_balance
    
    # Inicializar
    await system.initialize()
    
    # Executar
    try:
        await system.run()
    except KeyboardInterrupt:
        logger.info("⏹️ Interrompido pelo usuário")
    finally:
        await system.shutdown()


def handle_signals():
    """Configura handlers de sinais"""
    def signal_handler(sig, frame):
        logger.info(f"Sinal {sig} recebido")
        # Será tratado no loop principal
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # Exemplo de uso direto
    handle_signals()
    asyncio.run(run_paper_trading(debug_mode=True))
