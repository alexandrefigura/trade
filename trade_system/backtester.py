"""Sistema de backtesting"""
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp

from trade_system.analysis.technical import TechnicalAnalyzer
from trade_system.analysis.ml import MLPredictor
from trade_system.analysis.orderbook import OrderbookAnalyzer
from trade_system.risk import RiskManager
from trade_system.signals import SignalAggregator

class Backtester:
    """Sistema de backtesting para validação de estratégias"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Componentes
        self.ta_analyzer = TechnicalAnalyzer(config)
        self.ml_predictor = MLPredictor(config)
        self.risk_manager = RiskManager(config)
        self.signal_aggregator = SignalAggregator(config)
        
        # Estado
        self.initial_balance = config.base_balance
        self.balance = config.base_balance
        self.trades = []
        self.position = None
        
        # Dados
        self.candles = None
        
    async def run(self, days: int = 30, symbol: Optional[str] = None):
        """
        Executa backtest
        
        Args:
            days: Número de dias para testar
            symbol: Símbolo para testar (ou usa da config)
        """
        try:
            self.logger.info(f"🔬 Executando backtest de validação ({days} dias)...")
            
            # Reset estado
            self.balance = self.initial_balance
            self.trades = []
            self.position = None
            
            # Baixar dados
            symbol = symbol or self.config.symbol
            self.candles = await self._fetch_data(symbol, days)
            
            if len(self.candles) < 200:
                self.logger.warning("Dados insuficientes para backtest")
                return
            
            self.logger.info(f"✅ Dados carregados: {len(self.candles)} candles")
            self.logger.info(f"   Período: {self.candles.index[0]} até {self.candles.index[-1]}")
            self.logger.info(f"   Preço atual: ${self.candles['close'].iloc[-1]:,.2f}")
            
            # Treinar ML com dados iniciais
            self.logger.info("🔄 Iniciando backtest...")
            train_data = self.candles.iloc[:200]
            self.ml_predictor.train(train_data, self.ta_analyzer)
            
            # Simular trading
            await self._simulate_trading()
            
            self.logger.info("✅ Backtest concluído!")
            
        except Exception as e:
            self.logger.error(f"Erro no backtest: {e}")
            raise
    
    async def _fetch_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Busca dados históricos"""
        self.logger.info(f"📊 Baixando {days} dias de dados para {symbol}...")
        
        # Calcular timestamps
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        # Binance limita a 1000 candles por request
        limit = 1000
        interval = '1m'  # 1 minuto
        
        all_candles = []
        current_start = start_time
        
        async with aiohttp.ClientSession() as session:
            while current_start < end_time:
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_time,
                    'limit': limit
                }
                
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"Erro ao buscar dados: {response.status}")
                    
                    data = await response.json()
                    
                    if not data:
                        break
                    
                    all_candles.extend(data)
                    
                    # Próximo batch
                    last_timestamp = int(data[-1][0])
                    current_start = last_timestamp + 1
                    
                    # Evitar rate limit
                    await asyncio.sleep(0.1)
        
        # Converter para DataFrame
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'buy_base_volume',
            'buy_quote_volume', 'ignore'
        ])
        
        # Converter tipos
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Remover duplicatas
        df = df[~df.index.duplicated(keep='last')]
        
        self.logger.info(f"✅ {len(df)} candles baixados")
        
        return df
    
    async def _simulate_trading(self):
        """Simula trading com os dados históricos"""
        lookback = 100  # Candles necessários para análise
        
        for i in range(lookback, len(self.candles)):
            # Slice de dados até o momento atual
            current_candles = self.candles.iloc[:i+1]
            current_price = current_candles['close'].iloc[-1]
            
            # Analisar
            analysis_window = current_candles.iloc[-lookback:]
            signal = await self._analyze_market(analysis_window)
            
            # Processar sinal
            if signal['signal'] != 'HOLD' and not self.position:
                # Validar trade
                market_data = {
                    'volatility': signal['indicators'].get('volatility', 0),
                    'spread_bps': 10,  # Simulado
                    'momentum': signal['indicators'].get('momentum', 0)
                }
                
                can_trade, reason = self.risk_manager.validate_trade(
                    signal['signal'], signal['confidence'], market_data
                )
                
                if can_trade:
                    self._open_position(signal['signal'], current_price, signal['confidence'])
            
            # Gerenciar posição existente
            elif self.position:
                self._manage_position(current_price)
            
            # Atualizar ML periodicamente
            if i % 500 == 0 and i > 200:
                train_data = current_candles.iloc[-1000:]
                self.ml_predictor.train(train_data, self.ta_analyzer)
    
    async def _analyze_market(self, candles: pd.DataFrame) -> Dict[str, Any]:
        """Analisa mercado no ponto atual"""
        # Análise técnica
        ta_analysis = self.ta_analyzer.analyze(candles)
        
        # ML prediction
        ml_signal = {'signal': 'HOLD', 'confidence': 0.0}
        if ta_analysis['indicators']:
            features = self.ml_predictor.prepare_features(candles, ta_analysis['indicators'])
            ml_signal['signal'], ml_signal['confidence'] = self.ml_predictor.predict(features)
        
        # Orderbook simulado (não disponível em backtest)
        ob_analysis = {
            'signal': 'NEUTRAL',
            'buy_pressure': 0.5,
            'spread_bps': 10
        }
        
        # Agregar sinais
        return self.signal_aggregator.aggregate({
            'technical': ta_analysis,
            'ml': ml_signal,
            'orderbook': ob_analysis
        })
    
    def _open_position(self, signal: str, price: float, confidence: float):
        """Abre nova posição"""
        position_size = self.risk_manager.calculate_position_size(
            self.balance, price, confidence
        )
        
        self.position = {
            'type': signal,
            'entry_price': price,
            'size': position_size,
            'stop_loss': self.risk_manager.calculate_stop_loss(price, signal),
            'take_profit': self.risk_manager.calculate_take_profit(price, signal, confidence),
            'entry_time': self.candles.index[len(self.candles) - 1],
            'confidence': confidence
        }
        
        self.risk_manager.register_position(self.position)
    
    def _manage_position(self, current_price: float):
        """Gerencia posição aberta"""
        if not self.position:
            return
        
        # Verificar stop loss e take profit
        if self.position['type'] == 'BUY':
            if current_price <= self.position['stop_loss']:
                self._close_position(current_price, 'STOP_LOSS')
            elif current_price >= self.position['take_profit']:
                self._close_position(current_price, 'TAKE_PROFIT')
            else:
                # Trailing stop
                new_stop = self.risk_manager.update_trailing_stop(self.position, current_price)
                self.position['stop_loss'] = new_stop
        else:  # SELL
            if current_price >= self.position['stop_loss']:
                self._close_position(current_price, 'STOP_LOSS')
            elif current_price <= self.position['take_profit']:
                self._close_position(current_price, 'TAKE_PROFIT')
            else:
                # Trailing stop
                new_stop = self.risk_manager.update_trailing_stop(self.position, current_price)
                self.position['stop_loss'] = new_stop
    
    def _close_position(self, exit_price: float, reason: str):
        """Fecha posição"""
        if not self.position:
            return
        
        # Calcular resultado
        if self.position['type'] == 'BUY':
            profit_pct = (exit_price - self.position['entry_price']) / self.position['entry_price']
        else:
            profit_pct = (self.position['entry_price'] - exit_price) / self.position['entry_price']
        
        profit_usd = profit_pct * self.position['size'] * self.position['entry_price']
        
        # Atualizar balanço
        self.balance += profit_usd
        
        # Registrar trade
        trade = {
            **self.position,
            'exit_price': exit_price,
            'exit_time': self.candles.index[len(self.candles) - 1],
            'exit_reason': reason,
            'profit_pct': profit_pct,
            'profit_usd': profit_usd
        }
        self.trades.append(trade)
        
        # Atualizar risk manager
        self.risk_manager.close_position(self.position, exit_price, reason)
        
        self.position = None
    
    def get_metrics(self) -> Dict[str, float]:
        """Calcula métricas do backtest"""
        if not self.trades:
            self.logger.warning("⚠️ Nenhum trade executado no backtest")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        # Trades vencedores/perdedores
        winning_trades = [t for t in self.trades if t['profit_pct'] > 0]
        losing_trades = [t for t in self.trades if t['profit_pct'] <= 0]
        
        # Win rate
        win_rate = len(winning_trades) / len(self.trades)
        
        # Profit factor
        gross_profit = sum(t['profit_usd'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['profit_usd'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss
        
        # Sharpe ratio
        returns = [t['profit_pct'] for t in self.trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        equity_curve = [self.initial_balance]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade['profit_usd'])
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Retorno total
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'final_balance': self.balance,
            'total_profit': self.balance - self.initial_balance
        }
    
    def print_results(self):
        """Imprime resultados do backtest"""
        metrics = self.get_metrics()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    RESULTADOS DO BACKTEST                    ║
╚══════════════════════════════════════════════════════════════════════╝

📊 MÉTRICAS DE PERFORMANCE:
   Total de Trades: {metrics['total_trades']}
   Taxa de Acerto: {metrics['win_rate']:.2%}
   Profit Factor: {metrics['profit_factor']:.2f}
   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
   Max Drawdown: {metrics['max_drawdown']:.2%}

💰 RESULTADOS FINANCEIROS:
   Balance Inicial: ${self.initial_balance:,.2f}
   Balance Final: ${metrics['final_balance']:,.2f}
   Lucro/Prejuízo: ${metrics['total_profit']:+,.2f}
   Retorno Total: {metrics['total_return']:.2%}
        """)
        
        if self.trades:
            # Top trades
            sorted_trades = sorted(self.trades, key=lambda x: x['profit_pct'], reverse=True)
            
            print("\n📈 MELHORES TRADES:")
            for i, trade in enumerate(sorted_trades[:3]):
                print(f"   {i+1}. {trade['profit_pct']:+.2%} - {trade['exit_reason']}")
            
            print("\n📉 PIORES TRADES:")
            for i, trade in enumerate(sorted_trades[-3:]):
                print(f"   {i+1}. {trade['profit_pct']:+.2%} - {trade['exit_reason']}")
