#!/usr/bin/env python3
"""
Sistema de Paper Trading Integrado
Utiliza TODOS os módulos existentes do projeto
"""
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar TODOS os módulos do sistema
from trade_system.config import TradingConfig
from trade_system.logging_config import setup_logging
from trade_system.cache import TradingCache
from trade_system.alerts import AlertSystem
from trade_system.websocket_manager import WebSocketManager
from trade_system.analysis.technical import TechnicalAnalyzer
from trade_system.analysis.orderbook import OrderbookAnalyzer
from trade_system.analysis.ml import SimplifiedMLPredictor
from trade_system.risk import RiskManager
from trade_system.signals import SignalConsolidator
from trade_system.paper_trader import PaperTrader
from trade_system.checkpoint import CheckpointManager
from trade_system.rate_limiter import RateLimiter

# Carregar variáveis de ambiente
load_dotenv()

class IntegratedPaperTrading:
    """Sistema integrado que usa TODOS os módulos existentes"""
    
    def __init__(self):
        # Verificar API keys
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API keys não encontradas no .env")
        
        print(f"✅ API Key: {self.api_key[:8]}...")
        
        # Configurar logging
        setup_logging()
        
        # Carregar configuração
        self.config = TradingConfig()
        
        # Inicializar TODOS os componentes
        print("\n🔄 Inicializando componentes do sistema...")
        
        # Cache e alertas
        self.cache = TradingCache(self.config)
        self.alerts = AlertSystem(self.config)
        
        # WebSocket Manager
        self.ws_manager = WebSocketManager(self.config)
        
        # Análises
        self.technical = TechnicalAnalyzer(self.config)
        self.orderbook = OrderbookAnalyzer()
        self.ml_predictor = SimplifiedMLPredictor(self.config)
        
        # Risk e Signals
        self.risk_manager = RiskManager(self.config)
        self.signal_consolidator = SignalConsolidator()
        
        # Paper Trader
        self.paper_trader = PaperTrader(self.config)
        
        # Checkpoint e Rate Limiter
        self.checkpoint = CheckpointManager(self.config)
        self.rate_limiter = RateLimiter()
        
        # Estado
        self.market_data = {}
        self.running = True
        
        print("✅ Todos os componentes inicializados!")
    
    async def start_websocket(self):
        """Inicia o WebSocket para receber dados em tempo real"""
        try:
            # Callback para processar dados
            async def on_market_data(data):
                self.market_data = data
                
                # Atualizar cache
                await self.cache.set(
                    f"price:{self.config.SYMBOL}", 
                    data.get('price', 0),
                    ttl=60
                )
            
            # Conectar WebSocket
            await self.ws_manager.connect(on_market_data)
            
        except Exception as e:
            print(f"❌ Erro no WebSocket: {e}")
            self.alerts.send_alert(
                "ERROR",
                f"WebSocket error: {str(e)}",
                {"component": "websocket"}
            )
    
    async def analyze_market(self):
        """Executa análise completa do mercado usando todos os módulos"""
        while self.running:
            try:
                if not self.market_data:
                    await asyncio.sleep(1)
                    continue
                
                # Rate limiting
                if not self.rate_limiter.can_proceed("analysis"):
                    await asyncio.sleep(0.1)
                    continue
                
                # 1. Análise Técnica (com Numba otimizado)
                ta_signal = await self.technical.analyze(self.market_data)
                
                # 2. Análise de Orderbook
                orderbook_signal = self.orderbook.analyze(
                    self.market_data.get('bids', []),
                    self.market_data.get('asks', [])
                )
                
                # 3. Predição ML
                ml_prediction = self.ml_predictor.predict(self.market_data)
                
                # 4. Consolidar sinais
                consolidated_signal = self.signal_consolidator.consolidate([
                    ta_signal,
                    orderbook_signal,
                    ml_prediction
                ])
                
                # 5. Validação de risco
                if consolidated_signal['action'] != 'HOLD':
                    risk_approved = self.risk_manager.validate_trade(
                        consolidated_signal,
                        self.market_data,
                        self.paper_trader.get_balance()
                    )
                    
                    if not risk_approved:
                        consolidated_signal['action'] = 'HOLD'
                        consolidated_signal['reasons'].append('Rejeitado pelo Risk Manager')
                
                # 6. Executar trade se aprovado
                if consolidated_signal['action'] != 'HOLD':
                    await self.execute_trade(consolidated_signal)
                
                # 7. Salvar checkpoint
                if self.paper_trader.trades_count % 5 == 0:
                    await self.checkpoint.save_state({
                        'trades': self.paper_trader.get_trades(),
                        'balance': self.paper_trader.get_balance(),
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Display status
                self.display_status(consolidated_signal)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ Erro na análise: {e}")
                self.alerts.send_alert("ERROR", f"Analysis error: {str(e)}")
                await asyncio.sleep(5)
    
    async def execute_trade(self, signal):
        """Executa trade através do Paper Trader"""
        price = self.market_data.get('price', 0)
        
        if signal['action'] == 'BUY':
            success = await self.paper_trader.buy(
                price=price,
                amount=None,  # Paper trader calcula baseado no balance
                reason=signal.get('reasons', [])
            )
            
            if success:
                self.alerts.send_alert(
                    "TRADE",
                    f"COMPRA executada @ ${price:,.2f}",
                    signal
                )
        
        elif signal['action'] == 'SELL':
            success = await self.paper_trader.sell(
                price=price,
                reason=signal.get('reasons', [])
            )
            
            if success:
                self.alerts.send_alert(
                    "TRADE",
                    f"VENDA executada @ ${price:,.2f}",
                    signal
                )
    
    def display_status(self, signal):
        """Exibe status do sistema"""
        if not self.market_data:
            return
        
        price = self.market_data.get('price', 0)
        volume = self.market_data.get('volume', 0)
        
        # Obter métricas do paper trader
        metrics = self.paper_trader.get_metrics()
        
        status = f"💹 {self.config.SYMBOL}: ${price:,.2f} | "
        status += f"Vol: {volume:,.0f} | "
        status += f"Balance: ${metrics['balance']:,.2f} | "
        status += f"P&L: ${metrics['total_pnl']:,.2f} ({metrics['total_pnl_pct']:+.2f}%) | "
        status += f"Trades: {metrics['total_trades']} | "
        status += f"Win Rate: {metrics['win_rate']:.1f}% | "
        status += f"Sinal: {signal['action']} ({signal.get('confidence', 0)*100:.0f}%)"
        
        print(f"\r{status}", end='', flush=True)
    
    async def run(self):
        """Executa o sistema integrado"""
        print(f"""
{'='*80}
🤖 SISTEMA DE PAPER TRADING INTEGRADO
{'='*80}
📊 Par: {self.config.SYMBOL}
💰 Balance Inicial: ${self.config.INITIAL_BALANCE:,.2f}
⏰ Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
🔧 Componentes Ativos:
   ✅ WebSocket Manager (dados em tempo real)
   ✅ Technical Analyzer (com Numba otimizado)
   ✅ Orderbook Analyzer (análise de profundidade)
   ✅ ML Predictor (predições adaptativas)
   ✅ Risk Manager (proteção avançada)
   ✅ Signal Consolidator (sinais unificados)
   ✅ Paper Trader (simulação realista)
   ✅ Alert System (notificações)
   ✅ Cache System (Redis com fallback)
   ✅ Checkpoint Manager (recuperação automática)
   ✅ Rate Limiter (proteção de API)
{'='*80}
        """)
        
        # Carregar último checkpoint se existir
        last_state = await self.checkpoint.load_state()
        if last_state:
            print(f"📂 Checkpoint carregado: {last_state.get('timestamp', 'N/A')}")
            self.paper_trader.restore_state(last_state)
        
        # Iniciar tarefas assíncronas
        tasks = [
            asyncio.create_task(self.start_websocket()),
            asyncio.create_task(self.analyze_market())
        ]
        
        try:
            # Aguardar até ser interrompido
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            print("\n\n⏹️ Sistema interrompido pelo usuário")
            
        finally:
            self.running = False
            
            # Parar WebSocket
            await self.ws_manager.disconnect()
            
            # Salvar estado final
            await self.checkpoint.save_state({
                'trades': self.paper_trader.get_trades(),
                'balance': self.paper_trader.get_balance(),
                'metrics': self.paper_trader.get_metrics(),
                'timestamp': datetime.now().isoformat()
            })
            
            # Mostrar resumo final
            self.show_final_summary()
    
    def show_final_summary(self):
        """Mostra resumo detalhado usando todos os dados disponíveis"""
        metrics = self.paper_trader.get_metrics()
        trades = self.paper_trader.get_trades()
        
        print(f"""
{'='*80}
📊 RESUMO FINAL - SISTEMA INTEGRADO
{'='*80}
📈 Performance:
   Balance Inicial: ${self.config.INITIAL_BALANCE:,.2f}
   Balance Final: ${metrics['balance']:,.2f}
   Lucro/Prejuízo: ${metrics['total_pnl']:,.2f} ({metrics['total_pnl_pct']:+.2f}%)
   
📊 Estatísticas:
   Total de Trades: {metrics['total_trades']}
   Trades Vencedores: {metrics['winning_trades']}
   Trades Perdedores: {metrics['losing_trades']}
   Taxa de Acerto: {metrics['win_rate']:.1f}%
   Maior Ganho: ${metrics.get('best_trade', 0):,.2f}
   Maior Perda: ${metrics.get('worst_trade', 0):,.2f}
   
🔧 Componentes Utilizados:
   ✓ Análises Técnicas: {self.technical.analyses_count}
   ✓ Predições ML: {self.ml_predictor.predictions_count}
   ✓ Validações de Risco: {self.risk_manager.validations_count}
   ✓ Alertas Enviados: {self.alerts.alerts_sent}
   ✓ Checkpoints Salvos: {self.checkpoint.checkpoints_saved}
{'='*80}
        """)
        
        # Listar últimos 10 trades
        if trades:
            print("\n📜 ÚLTIMOS 10 TRADES:")
            for trade in trades[-10:]:
                print(f"   {trade['timestamp']} - {trade['action']} @ ${trade['price']:,.2f} - {trade.get('reason', 'N/A')}")

async def main():
    try:
        # Criar e executar sistema integrado
        system = IntegratedPaperTrading()
        await system.run()
        
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Se houver erros de importação, use:")
        print("   python working_paper_trading.py")

if __name__ == "__main__":
    asyncio.run(main())
