"""
Módulo principal para orquestração do sistema de trading
"""
import os
import asyncio
import signal
import numpy as np
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
        
        # Ativar modo debug no consolidador se necessário
        if self.config.debug_mode:
            self.signal_consolidator.set_debug_mode(True)
        
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
        
        # Contadores para debug - valores padrão garantidos
        self.debug_counters = {
            'cycles_without_trade': 0,
            'force_trade_threshold': 100,
            'force_first_trade': getattr(self.config, 'force_first_trade', True)
        }
        
        logger.info(f"🚀 Sistema inicializado - Modo: {'PAPER TRADING' if paper_trading else 'LIVE'}")
        
        # Log configurações importantes
        logger.info(f"""
🔧 DEBUG CONFIG:
- min_confidence: {self.config.min_confidence}
- rsi_buy: {self.config.rsi_buy_threshold} / rsi_sell: {self.config.rsi_sell_threshold}
- buy_threshold: {self.config.buy_threshold} / sell_threshold: {self.config.sell_threshold}
- force_first_trade: {self.debug_counters['force_first_trade']}
- use_simulated_data: {getattr(self.config, 'use_simulated_data', False)}
        """)
    
    async def initialize(self):
        """Inicializa componentes assíncronos"""
        # Carregar checkpoint se existir
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            self._restore_from_checkpoint(checkpoint)
        
        # Iniciar WebSocket ou modo simulado
        if getattr(self.config, 'use_simulated_data', False):
            logger.info("🎮 Usando dados simulados para debug")
            self.ws_manager.enable_simulation_mode()
        else:
            self.ws_manager.start_delayed()
        
        # Aguardar dados
        logger.info("⏳ Aguardando dados do WebSocket...")
        for i in range(50):
            await asyncio.sleep(0.1)
            if self.ws_manager.buffer_filled or self.ws_manager.buffer_index > 100:
                logger.info("✅ Dados recebidos!")
                break
            elif i % 10 == 0:
                status = self.ws_manager.get_connection_status()
                logger.debug(f"Buffer: {status['buffer_size']} | Conectado: {status['connected']}")
        
        # Se ainda não tem dados suficientes, mas está em debug, continuar
        if self.config.debug_mode and not self.ws_manager.buffer_filled:
            logger.warning("⚠️ Buffer não preenchido, mas continuando em modo debug")
    
    async def run(self):
        """Loop principal do sistema"""
        self.is_running = True
        cycle_count = 0
        cycles_without_position = 0
        
        try:
            while self.is_running:
                # Obter dados
                market_data = self.ws_manager.get_latest_data()
                if not market_data or len(market_data.get('prices', [])) == 0:
                    await asyncio.sleep(0.1)
                    continue
                
                # Validar mercado (pular se debug)
                if not self.config.debug_mode:
                    is_safe, reasons = await self.market_validator.validate_market_conditions(
                        market_data
                    )
                    
                    if not is_safe:
                        if cycle_count % 100 == 0:
                            logger.warning(f"⚠️ Mercado inseguro: {', '.join(reasons)}")
                        await asyncio.sleep(0.1)
                        continue
                
                # Analisar mercado
                signals = await self._analyze_market(market_data)
                
                # Consolidar sinais
                action, confidence = self.signal_consolidator.consolidate(signals)
                
                # Preço atual
                current_price = float(market_data['prices'][-1])
                
                # Debug log detalhado
                if cycle_count % 10 == 0 or (action != 'HOLD' and confidence > 0.3):
                    logger.info(f"""
📊 DEBUG - Ciclo {cycle_count}:
- Preço: ${current_price:,.2f}
- Sinais: {[(s[0], s[1], f"{s[2]:.2%}") for s in signals]}
- Decisão: {action} (confiança: {confidence:.2%})
- Min confiança requerida: {self.config.min_confidence:.2%}
- Tem posição: {self.position is not None}
- Condições para abrir: action={action!='HOLD'}, conf={confidence >= self.config.min_confidence}
- Ciclos sem posição: {cycles_without_position}
                    """)
                
                # Contar ciclos sem posição
                if not self.position:
                    cycles_without_position += 1
                else:
                    cycles_without_position = 0
                
                # Gerenciar posições
                if self.position:
                    await self._manage_position(market_data, action, confidence)
                else:
                    # Verificar se deve abrir posição
                    should_open = False
                    reason = ""
                    
                    # Condição normal
                    if action != 'HOLD' and confidence >= self.config.min_confidence:
                        should_open = True
                        reason = "Condições normais atendidas"
                    
                    # Forçar primeira posição se configurado
                    elif (self.debug_counters.get('force_first_trade', False) and 
                          self.performance_stats['total_trades'] == 0 and
                          cycles_without_position > self.debug_counters['force_trade_threshold']):
                        
                        # Usar o sinal mais forte disponível
                        if action != 'HOLD':
                            should_open = True
                            reason = f"FORÇANDO primeira posição após {cycles_without_position} ciclos"
                            # Boost de confiança artificial
                            confidence = max(confidence, self.config.min_confidence)
                        else:
                            # Se todos os sinais são HOLD, forçar baseado em análise simples
                            if current_price > 0:
                                # Decisão aleatória ponderada pelo preço
                                should_open = True
                                action = 'BUY' if np.random.random() > 0.5 else 'SELL'
                                confidence = self.config.min_confidence
                                reason = f"FORÇANDO posição aleatória após {cycles_without_position} ciclos"
                    
                    if should_open:
                        logger.info(f"✅ {reason}")
                        await self._open_position(market_data, action, confidence)
                    elif action != 'HOLD':
                        logger.debug(f"❌ Confiança insuficiente: {confidence:.2%} < {self.config.min_confidence:.2%}")
                
                # Checkpoint periódico
                if cycle_count > 0 and cycle_count % 300 == 0:  # A cada 5 minutos
                    await self._save_checkpoint()
                
                # Status periódico
                if cycle_count > 0 and cycle_count % 60 == 0:  # A cada minuto
                    await self._log_status()
                
                # Incrementar contador
                cycle_count += 1
                
                # Sleep
                await asyncio.sleep(self.config.main_loop_interval_ms / 1000)
                
        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def _analyze_market(self, market_data):
        """Analisa mercado e retorna sinais"""
        signals = []
        tech_details = {}
        ob_details = {}
        
        # Análise técnica
        try:
            tech_action, tech_conf, tech_details = self.technical_analyzer.analyze(
                market_data['prices'],
                market_data['volumes']
            )
            signals.append(('technical', tech_action, tech_conf))
        except Exception as e:
            logger.error(f"Erro em análise técnica: {e}")
            signals.append(('technical', 'HOLD', 0.5))
        
        # Análise orderbook
        try:
            ob_action, ob_conf, ob_details = self.orderbook_analyzer.analyze(
                market_data['orderbook_bids'],
                market_data['orderbook_asks'],
                self.cache
            )
            signals.append(('orderbook', ob_action, ob_conf))
        except Exception as e:
            logger.error(f"Erro em análise orderbook: {e}")
            signals.append(('orderbook', 'HOLD', 0.5))
        
        # ML - com features da análise técnica
        try:
            features = {
                'rsi': tech_details.get('rsi', 50),
                'momentum': self._calculate_momentum(market_data['prices']),
                'volume_ratio': self._calculate_volume_ratio(market_data['volumes']),
                'spread_bps': ob_details.get('spread_bps', 10),
                'volatility': self._calculate_volatility(market_data['prices'])
            }
            
            ml_action, ml_conf = self.ml_predictor.predict(features)
            
            # Em modo debug, dar boost ao ML se estiver muito baixo
            if self.config.debug_mode and ml_conf < 0.4:
                ml_conf = 0.4
            
            signals.append(('ml', ml_action, ml_conf))
        except Exception as e:
            logger.error(f"Erro em ML: {e}")
            signals.append(('ml', 'HOLD', 0.3))
        
        return signals
    
    async def _open_position(self, market_data, action, confidence):
        """Abre nova posição"""
        if self.paper_trading:
            await self._open_paper_position(market_data, action, confidence)
        else:
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
            logger.warning("❌ Tamanho da posição calculado como 0")
            return
        
        # Calcular taxas
        entry_fee = position_size * 0.001
        
        # Criar posição
        self.position = {
            'side': action,
            'entry_price': current_price,
            'quantity': position_size / current_price,
            'entry_time': datetime.now(),
            'entry_timestamp': datetime.now().timestamp(),
            'confidence': confidence,
            'entry_fee': entry_fee,
            'paper_trade': True,
            'highest_price': current_price,
            'lowest_price': current_price,
            'volatility': volatility,
            'stop_loss_pct': 0.01,  # 1% padrão
            'take_profit_pct': 0.015,  # 1.5% padrão
            'max_duration': 3600,  # 1 hora padrão
            'highest_pnl': 0  # Para trailing stop
        }
        
        # Atualizar risk manager
        self.risk_manager.set_position_info(self.position)
        self.risk_manager.current_balance -= entry_fee
        
        # Reset contador de trades forçados
        self.debug_counters['cycles_without_trade'] = 0
        
        logger.info(f"""
🟢 POSIÇÃO ABERTA [{action}]
- Preço: ${current_price:,.2f}
- Quantidade: {self.position['quantity']:.6f}
- Valor: ${position_size:,.2f}
- Taxa: ${entry_fee:.2f}
- Confiança: {confidence*100:.1f}%
- Balance: ${self.risk_manager.current_balance:,.2f}
        """)
    
    async def _manage_position(self, market_data, signal_action, signal_confidence):
        """Gerencia posição existente"""
        if not self.position:
            return
            
        current_price = float(market_data['prices'][-1])
        
        # Atualizar preços máximo/mínimo
        self.position['highest_price'] = max(self.position['highest_price'], current_price)
        self.position['lowest_price'] = min(self.position['lowest_price'], current_price)
        
        # Verificar stops
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
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price
        
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
        
        # Calcular duração
        duration = datetime.now() - self.position['entry_time']
        
        logger.info(f"""
🔴 POSIÇÃO FECHADA
- Motivo: {reason}
- Lado: {side}
- Entrada: ${entry_price:,.2f}
- Saída: ${exit_price:,.2f}
- Duração: {duration}
- P&L: ${pnl_net:,.2f} ({pnl_pct*100:+.2f}%)
- Balance: ${self.risk_manager.current_balance:,.2f}
        """)
        
        self.position = None
    
    async def _log_status(self):
        """Log status periódico"""
        win_rate = (self.performance_stats['winning_trades'] / self.performance_stats['total_trades'] * 100 
                   if self.performance_stats['total_trades'] > 0 else 0)
        
        ta_stats = self.technical_analyzer.get_signal_stats()
        ml_stats = self.ml_predictor.get_prediction_stats()
        signal_stats = self.signal_consolidator.get_signal_statistics()
        
        logger.info(f"""
📊 STATUS DO SISTEMA:
- Trades: {self.performance_stats['total_trades']} (Win rate: {win_rate:.1f}%)
- P&L Total: ${self.performance_stats['total_pnl']:,.2f}
- Taxas Totais: ${self.performance_stats['total_fees']:,.2f}
- Balance: ${self.risk_manager.current_balance:,.2f}
- Retorno: {(self.risk_manager.current_balance - self.risk_manager.initial_balance) / self.risk_manager.initial_balance * 100:+.2f}%
- Posição: {'SIM - ' + self.position['side'] if self.position else 'NÃO'}

📈 ANÁLISE TÉCNICA:
- Total: {ta_stats['total_signals']} | BUY: {ta_stats['buy_signals']} | SELL: {ta_stats['sell_signals']} | HOLD: {ta_stats['hold_signals']}

🤖 MACHINE LEARNING:
- Predições: {ml_stats['total_predictions']} | Confiança média: {ml_stats['avg_confidence']:.2%}

🎯 SINAIS CONSOLIDADOS:
- Total: {signal_stats['total_signals']} | Confiança média: {signal_stats['avg_confidence']:.2%}
        """)
    
    async def _save_checkpoint(self):
        """Salva checkpoint"""
        state = {
            'balance': self.risk_manager.current_balance,
            'position': self.position,
            'performance_stats': self.performance_stats,
            'paper_trading': self.paper_trading,
            'debug_counters': self.debug_counters
        }
        
        self.checkpoint_manager.save_checkpoint(state)
        self.checkpoint_manager.update_checkpoint_time()
    
    def _restore_from_checkpoint(self, checkpoint):
        """Restaura estado do checkpoint com validação"""
        self.risk_manager.current_balance = checkpoint.get('balance', 10000)
        self.position = checkpoint.get('position')
        self.performance_stats = checkpoint.get('performance_stats', self.performance_stats)
        
        # Restaurar debug_counters com merge seguro
        saved_counters = checkpoint.get('debug_counters', {})
        
        # Atualizar apenas chaves existentes no checkpoint
        for key, value in saved_counters.items():
            if key in self.debug_counters:
                self.debug_counters[key] = value
        
        # Garantir que force_first_trade existe
        if 'force_first_trade' not in self.debug_counters:
            self.debug_counters['force_first_trade'] = getattr(self.config, 'force_first_trade', True)
        
        logger.info("✅ Estado restaurado do checkpoint")
    
    async def shutdown(self):
        """Desliga sistema"""
        logger.info("🛑 Desligando sistema...")
        self.is_running = False
        
        await self._log_status()
        
        self.ws_manager.stop()
        
        await self._save_checkpoint()
        
        logger.info("✅ Sistema desligado")
    
    # Helpers
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
    """Executa paper trading"""
    setup_logging()
    
    if config is None:
        config = get_config(debug_mode=debug_mode)
    
    system = TradingSystem(config, paper_trading=True)
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