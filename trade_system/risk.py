"""
Sistema Principal de Trading com Proteções Aprimoradas
"""
import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Optional
import json

from .websocket_manager import WebSocketManager
from .analysis.technical import TechnicalAnalyzer
from .analysis.ml import MLPredictor
from .risk import RiskManager
from .alerts import AlertSystem
from .checkpoint import CheckpointManager
from .signals import SignalConsolidator

logger = logging.getLogger(__name__)

class TradingSystem:
    """Sistema principal de trading com proteções aprimoradas"""
    
    def __init__(self, config: dict, mode: str = 'PAPER'):
        self.config = config
        self.mode = mode
        self.running = False
        
        # Componentes do sistema
        self.websocket = WebSocketManager(config)
        self.ta_analyzer = TechnicalAnalyzer(config)
        self.ml_predictor = MLPredictor(config)
        self.risk_manager = RiskManager(config)
        self.alert_system = AlertSystem(config)
        self.checkpoint = CheckpointManager(config)
        self.signal_consolidator = SignalConsolidator(config)
        
        # Estado do sistema
        self.last_price = None
        self.last_signal_time = 0
        self.signal_cooldown = config.get('signals', {}).get('trade_cooldown', 60)
        self.trades_count = 0
        self.paper_position = None
        self.market_data = {}
        
        # Estatísticas
        self.session_start_balance = self.risk_manager.balance
        self.session_pnl = 0
        
        logger.info(f"🚀 Sistema inicializado - Modo: {mode}")
        
    async def run(self):
        """Loop principal do sistema com proteções aprimoradas"""
        try:
            self.running = True
            
            # Carrega checkpoint se existir
            self._load_checkpoint()
            
            # Inicia websocket
            asyncio.create_task(self.websocket.start())
            
            # Aguarda dados iniciais
            logger.info("⏳ Aguardando dados do WebSocket...")
            while not self.websocket.is_ready:
                await asyncio.sleep(1)
            
            logger.info("✅ Dados recebidos!")
            
            # Loop principal
            while self.running:
                try:
                    # Obtém dados de mercado
                    data = self.websocket.get_latest_data()
                    if data:
                        self.market_data = data
                        self.last_price = float(data.get('price', 0))
                        
                        # Verifica condições de saída das posições abertas
                        await self._check_positions_exit()
                        
                        # Processa sinais apenas se não estiver em cooldown
                        if time.time() - self.last_signal_time >= self.signal_cooldown:
                            await self._process_signals(data)
                    
                    # Salva checkpoint periodicamente
                    if self.trades_count % 5 == 0:  # A cada 5 trades
                        self._save_checkpoint()
                    
                    # Pequena pausa para não sobrecarregar
                    await asyncio.sleep(self.config.get('system', {}).get('loop_interval', 1))
                    
                except Exception as e:
                    logger.error(f"❌ Erro no loop principal: {e}")
                    await asyncio.sleep(5)  # Pausa maior em caso de erro
                    
        except KeyboardInterrupt:
            logger.info("🛑 Interrupção do usuário detectada")
        finally:
            await self.shutdown()
    
    async def _check_positions_exit(self):
        """Verifica condições de saída das posições abertas"""
        if not hasattr(self.risk_manager, 'check_exit_conditions'):
            return
            
        position = self.risk_manager.get_open_position()
        if not position or not self.last_price:
            return
        
        # Verifica stop loss e take profit
        exit_reason = self.risk_manager.check_exit_conditions(position, self.last_price)
        
        if exit_reason:
            # Fecha a posição
            result = self.risk_manager.close_position(position, self.last_price, exit_reason)
            
            # Atualiza estatísticas
            self.trades_count += 1
            self.session_pnl += result['pnl']
            
            # Log detalhado
            logger.info(f"📊 Trade #{self.trades_count} fechado por {exit_reason}")
            logger.info(f"   Entrada: ${position.entry_price:,.2f} | Saída: ${self.last_price:,.2f}")
            logger.info(f"   P&L: ${result['pnl']:.2f} ({result['pnl_pct']:.2f}%)")
            logger.info(f"   Novo balanço: ${result['new_balance']:,.2f}")
            logger.info(f"   P&L da sessão: ${self.session_pnl:.2f}")
            
            # Limpa posição paper
            self.paper_position = None
            
            # Envia alerta
            await self.alert_system.send_trade_closed(
                exit_reason,
                result['pnl'],
                result['new_balance']
            )
            
            # Salva checkpoint
            self._save_checkpoint()
    
    async def _process_signals(self, data: Dict):
        """Processa sinais de trading com filtros aprimorados"""
        # Análise técnica
        ta_signal = self.ta_analyzer.analyze(self.market_data)
        
        # Predição ML
        ml_signal = self.ml_predictor.predict(self.market_data)
        
        # Consolida sinais
        final_signal = self.signal_consolidator.consolidate([ta_signal, ml_signal])
        
        if final_signal and final_signal['confidence'] >= self.config.get('min_confidence', 0.1):
            action = final_signal['action']
            confidence = final_signal['confidence']
            
            # Log do sinal
            logger.info(f"📊 Sinal: {action} (conf: {confidence:.1%})")
            
            # Verifica se pode executar
            if await self._can_execute_trade(action, confidence):
                await self._execute_trade(data, action, confidence)
    
    async def _can_execute_trade(self, action: str, confidence: float) -> bool:
        """Verifica se pode executar o trade com todas as proteções"""
        # Verifica se tem posição aberta
        if hasattr(self.risk_manager, 'has_open_position') and self.risk_manager.has_open_position():
            position = self.risk_manager.get_open_position()
            
            # Só executa se for sinal contrário (para fechar)
            if (position.side == 'BUY' and action == 'SELL') or \
               (position.side == 'SELL' and action == 'BUY'):
                logger.info(f"🔄 Sinal contrário detectado - preparando para reverter posição")
                return True
            else:
                logger.debug(f"⏭️ Ignorando sinal {action} - já tem posição {position.side} aberta")
                return False
        
        # Verifica limite diário de trades
        max_daily_trades = self.config.get('risk', {}).get('max_daily_trades', 10)
        if self.trades_count >= max_daily_trades:
            logger.warning(f"⚠️ Limite diário de trades atingido: {self.trades_count}/{max_daily_trades}")
            return False
        
        # Verifica stop loss diário
        max_daily_loss = self.config.get('risk', {}).get('max_daily_loss', 0.05)
        daily_loss_pct = abs(self.session_pnl / self.session_start_balance)
        if daily_loss_pct >= max_daily_loss:
            logger.warning(f"⚠️ Stop loss diário atingido: {daily_loss_pct:.1%}")
            return False
        
        return True
    
    async def _execute_trade(self, data: Dict, action: str, confidence: float):
        """Executa o trade com gestão de risco aprimorada"""
        price = float(data['price'])
        
        # Se tem posição aberta, fecha primeiro
        if hasattr(self.risk_manager, 'has_open_position') and self.risk_manager.has_open_position():
            position = self.risk_manager.get_open_position()
            if (position.side == 'BUY' and action == 'SELL') or \
               (position.side == 'SELL' and action == 'BUY'):
                # Fecha posição atual
                result = self.risk_manager.close_position(position, price, 'SIGNAL_REVERSAL')
                self.trades_count += 1
                self.session_pnl += result['pnl']
                logger.info(f"🔄 Posição {position.side} fechada por reversão")
                
                # Aguarda um pouco antes de abrir nova
                await asyncio.sleep(2)
        
        # Calcula tamanho da posição
        position_size = self.risk_manager.calculate_position_size(
            price=price,
            confidence=confidence,
            volatility=self.market_data.get('volatility', 0.01)
        )
        
        # Abre nova posição
        if self.mode == 'PAPER':
            await self._open_paper_position(data, action, confidence, position_size)
        else:
            await self._open_live_position(data, action, confidence, position_size)
        
        # Atualiza timestamp do último sinal
        self.last_signal_time = time.time()
    
    async def _open_paper_position(self, data: Dict, action: str, confidence: float, position_size: Dict):
        """Abre posição no modo paper trading"""
        price = float(data['price'])
        size = position_size['size']
        quantity = position_size['quantity']
        
        # Calcula taxas
        fee = size * 0.001  # 0.1% de taxa
        
        # Registra posição
        if hasattr(self.risk_manager, 'open_position'):
            position = self.risk_manager.open_position(
                side=action,
                price=price,
                quantity=quantity,
                size=size,
                current_time=time.time()
            )
            
            if position:
                self.paper_position = {
                    'side': action,
                    'entry_price': price,
                    'quantity': quantity,
                    'size': size,
                    'entry_time': time.time(),
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit
                }
                
                # Atualiza balanço (desconta taxa)
                self.risk_manager.balance -= fee
                
                # Log
                logger.info(f"🟢 POSIÇÃO ABERTA [{action}] price=${price:,.2f} size=${size:.2f} qty={quantity:.6f} fee=${fee:.2f}")
                logger.info(f"   Stop Loss: ${position.stop_loss:,.2f} | Take Profit: ${position.take_profit:,.2f}")
                
                # Alerta
                await self.alert_system.send_position_opened(
                    action=action,
                    price=price,
                    size=size,
                    confidence=confidence,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit
                )
        else:
            # Fallback para lógica antiga
            self.paper_position = {
                'side': action,
                'entry_price': price,
                'quantity': quantity,
                'size': size,
                'entry_time': time.time()
            }
            
            self.risk_manager.balance -= fee
            
            logger.info(f"🟢 POSIÇÃO ABERTA [{action}] price=${price:,.2f} size=${size:.2f} qty={quantity:.6f} fee=${fee:.2f}")
    
    async def _open_live_position(self, data: Dict, action: str, confidence: float, position_size: Dict):
        """Abre posição no modo live trading"""
        # TODO: Implementar integração com exchange
        logger.warning("⚠️ Modo LIVE ainda não implementado")
        pass
    
    def _save_checkpoint(self):
        """Salva estado atual do sistema"""
        checkpoint_data = {
            'balance': self.risk_manager.balance,
            'trades': self.trades_count,
            'timestamp': datetime.now(),
            'position': self.paper_position if self.mode == 'PAPER' else None,
            'session_pnl': self.session_pnl,
            'daily_pnl': getattr(self.risk_manager, 'daily_pnl', 0),
            'winning_trades': getattr(self.risk_manager, 'winning_trades', 0),
            'losing_trades': getattr(self.risk_manager, 'losing_trades', 0),
            'total_trades': getattr(self.risk_manager, 'total_trades', 0),
            'last_price': self.last_price
        }
        
        self.checkpoint.save(checkpoint_data)
        logger.info(f"✅ Checkpoint salvo: {self.checkpoint.get_latest_filename()}")
    
    def _load_checkpoint(self):
        """Carrega estado salvo se existir"""
        checkpoint_data = self.checkpoint.load()
        
        if checkpoint_data:
            self.risk_manager.balance = checkpoint_data.get('balance', self.risk_manager.balance)
            self.trades_count = checkpoint_data.get('trades', 0)
            self.paper_position = checkpoint_data.get('position')
            self.session_pnl = checkpoint_data.get('session_pnl', 0)
            
            # Restaura estatísticas se existirem
            if hasattr(self.risk_manager, 'daily_pnl'):
                self.risk_manager.daily_pnl = checkpoint_data.get('daily_pnl', 0)
                self.risk_manager.winning_trades = checkpoint_data.get('winning_trades', 0)
                self.risk_manager.losing_trades = checkpoint_data.get('losing_trades', 0)
                self.risk_manager.total_trades = checkpoint_data.get('total_trades', 0)
            
            # Restaura posição aberta se existir
            if self.paper_position and hasattr(self.risk_manager, 'positions'):
                from .risk import Position
                position = Position(
                    side=self.paper_position['side'],
                    entry_price=self.paper_position['entry_price'],
                    quantity=self.paper_position['quantity'],
                    size=self.paper_position['size'],
                    entry_time=self.paper_position['entry_time'],
                    stop_loss=self.paper_position.get('stop_loss'),
                    take_profit=self.paper_position.get('take_profit')
                )
                self.risk_manager.positions.append(position)
            
            logger.info("✅ Estado restaurado do checkpoint")
            logger.info(f"   Balanço: ${self.risk_manager.balance:,.2f}")
            logger.info(f"   Trades: {self.trades_count}")
            logger.info(f"   P&L da sessão: ${self.session_pnl:.2f}")
            
            if self.paper_position:
                logger.info(f"   Posição aberta: {self.paper_position['side']} @ ${self.paper_position['entry_price']:,.2f}")
    
    async def shutdown(self):
        """Desliga o sistema de forma segura"""
        logger.info("🛑 Desligando sistema...")
        
        self.running = False
        
        # Para o websocket
        if self.websocket:
            await self.websocket.stop()
        
        # Salva checkpoint final
        self._save_checkpoint()
        
        # Envia alerta
        await self.alert_system.send_system_stopped()
        
        # Mostra resumo da sessão
        logger.info("📊 Resumo da sessão:")
        logger.info(f"   Balanço inicial: ${self.session_start_balance:,.2f}")
        logger.info(f"   Balanço final: ${self.risk_manager.balance:,.2f}")
        logger.info(f"   P&L total: ${self.session_pnl:.2f}")
        logger.info(f"   Trades executados: {self.trades_count}")
        
        if hasattr(self.risk_manager, 'winning_trades'):
            win_rate = (self.risk_manager.winning_trades / self.risk_manager.total_trades * 100) if self.risk_manager.total_trades > 0 else 0
            logger.info(f"   Win rate: {win_rate:.1f}%")
        
        logger.info("✅ Sistema desligado")

# Funções auxiliares para executar o sistema
async def run_paper_trading(config: dict):
    """Executa o sistema em modo paper trading"""
    system = TradingSystem(config, mode='PAPER')
    await system.run()

async def run_live_trading(config: dict):
    """Executa o sistema em modo live trading"""
    system = TradingSystem(config, mode='LIVE')
    await system.run()
