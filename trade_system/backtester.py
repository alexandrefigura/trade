"""
Sistema de backtesting integrado para validação de estratégias
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
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
        self.results = []
        self.trades = []
        self.equity_curve = []
        
    async def backtest_strategy(
        self,
        historical_data: pd.DataFrame,
        initial_balance: float = 10000
    ) -> Dict:
        """
        Executa backtest com dados históricos
        
        Args:
            historical_data: DataFrame com OHLCV
            initial_balance: Capital inicial
            
        Returns:
            Dicionário com métricas do backtest
        """
        logger.info("🔄 Iniciando backtest...")
        
        # Validar dados
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in historical_data.columns:
                logger.error(f"Coluna {col} não encontrada nos dados")
                return {}
        
        # Preparar dados
        balance = initial_balance
        position = None
        trades = []
        equity_curve = []
        
        # Converter para arrays NumPy
        prices = historical_data['close'].values.astype(np.float32)
        volumes = historical_data['volume'].values.astype(np.float32)
        
        # Arrays adicionais se disponíveis
        highs = historical_data.get('high', prices).values.astype(np.float32)
        lows = historical_data.get('low', prices).values.astype(np.float32)
        
        # Calcular ATR
        atr_series = calculate_atr(highs, lows, prices, period=self.config.atr_period)
        
        # Componentes de análise
        technical_analyzer = UltraFastTechnicalAnalysis(self.config)
        ml_predictor = SimplifiedMLPredictor()
        consolidator = OptimizedSignalConsolidator()
        
        # Loop principal
        start_idx = max(200, self.config.atr_period)  # Mínimo para indicadores
        
        # Contadores
        total_signals = 0
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for i in range(start_idx, len(prices)):
            # Slice de dados
            price_slice = prices[:i+1]
            volume_slice = volumes[:i+1]
            
            # Análise técnica
            tech_action, tech_conf, tech_details = technical_analyzer.analyze(
                price_slice[-1000:],  # Últimos 1000 pontos
                volume_slice[-1000:]
            )
            
            # Features para ML
            features = self._extract_features(
                prices, volumes, i, tech_details
            )
            
            # Predição ML
            ml_action, ml_conf = ml_predictor.predict(features)
            
            # Consolidar sinais
            signals = [
                ('technical', tech_action, tech_conf),
                ('ml', ml_action, ml_conf)
            ]
            
            action, confidence = consolidator.consolidate(signals)
            
            # Contar sinais
            signal_counts[action] += 1
            if action != 'HOLD':
                total_signals += 1
            
            # Log periódico em modo debug
            if self.config.debug_mode and i % 500 == 0:
                logger.debug(f"Backtest progresso: {i}/{len(prices)} ({i/len(prices)*100:.1f}%)")
            
            # Gerenciar posições
            current_price = prices[i]

            # Ignorar sinais SELL sem posição aberta (sem short selling)
            if position is None and action == 'SELL':
                continue
            
            # Entrada LONG apenas
            if position is None and action == 'BUY' and confidence > self.config.min_confidence:
                # Calcular tamanho da posição
                position_size = self._calculate_position_size(
                    balance, confidence, features.get('volatility', 0.01)
                )
                
                if position_size > 0:
                    # ATR para stops
                    atr = atr_series[i] if i < len(atr_series) and not np.isnan(atr_series[i]) else None
                    
                    if atr and atr > 0:
                        # Calcular stops baseados em ATR
                        tp_price = current_price + (atr * self.config.tp_multiplier)
                        sl_price = current_price - (atr * self.config.sl_multiplier)
                    else:
                        # Fallback para percentuais fixos
                        tp_price = current_price * 1.015
                        sl_price = current_price * 0.99
                    
                    # Abrir posição
                    position = {
                        'side': 'BUY',
                        'entry_price': current_price,
                        'entry_idx': i,
                        'entry_time': historical_data.index[i] if hasattr(historical_data, 'index') else i,
                        'size': position_size / current_price,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'confidence': confidence
                    }
            
            # Saída
            elif position is not None:
                should_close, reason = self._check_exit_conditions(
                    position, current_price, action, confidence
                )
                
                if should_close:
                    # Fechar posição
                    trade_result = self._close_position(
                        position, current_price, i, reason, balance
                    )
                    
                    trades.append(trade_result)
                    balance += trade_result['pnl_net']
                    position = None
            
            # Registrar equity
            equity_curve.append({
                'index': i,
                'timestamp': historical_data.index[i] if hasattr(historical_data, 'index') else i,
                'balance': balance,
                'in_position': position is not None,
                'price': current_price
            })
        
        # Fechar posição aberta no final
        if position is not None:
            trade_result = self._close_position(
                position, prices[-1], len(prices)-1, "Fim do backtest", balance
            )
            trades.append(trade_result)
            balance += trade_result['pnl_net']
        
        # Log final
        logger.info(f"""
📊 Backtest concluído:
- Período: {len(prices)} candles
- Sinais gerados: {total_signals}
- Distribuição: BUY={signal_counts['BUY']}, SELL={signal_counts['SELL']}, HOLD={signal_counts['HOLD']}
- Trades executados: {len(trades)}
- Balance final: ${balance:,.2f}
        """ )
        
        # Calcular métricas
        return self._calculate_metrics(
            trades, initial_balance, balance, equity_curve
        )
    
    def _extract_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        idx: int,
        tech_details: Dict
    ) -> Dict:
        """Extrai features para ML"""
        # Momentum
        momentum = 0
        if idx >= 20:
            momentum = (prices[idx] - prices[idx-20]) / prices[idx-20]
        
        # Volume ratio
        volume_ratio = 1
        if idx >= 20:
            avg_volume = np.mean(volumes[max(0, idx-20):idx])
            if avg_volume > 0:
                volume_ratio = volumes[idx] / avg_volume
        
        # Volatilidade
        volatility = 0.01
        if idx >= 50:
            volatility = np.std(prices[max(0, idx-50):idx]) / np.mean(prices[max(0, idx-50):idx])
        
        # Trend
        price_trend = 0
        if idx >= 50:
            # Regressão linear simples
            x = np.arange(50)
            y = prices[idx-50:idx]
            if len(y) == 50:
                slope = np.polyfit(x, y, 1)[0]
                price_trend = slope / np.mean(y)
        
        return {
            'rsi': tech_details.get('rsi', 50),
            'momentum': momentum,
            'volume_ratio': volume_ratio,
            'spread_bps': 5,  # Assumir spread fixo no backtest
            'volatility': volatility,
            'price_trend': price_trend
        }
    
    def _calculate_position_size(
        self,
        balance: float,
        confidence: float,
        volatility: float
    ) -> float:
        """Calcula tamanho da posição para backtest"""
        # Kelly simplificado
        base_size = balance * self.config.max_position_pct * confidence
        
        # Ajustar por volatilidade
        if volatility > 0.03:
            base_size *= 0.5
        elif volatility > 0.02:
            base_size *= 0.7
        
        # Limites
        min_size = 50  # USD
        max_size = balance * 0.1  # 10% máximo
        
        return max(min_size, min(base_size, max_size))
    
    def _check_exit_conditions(
        self,
        position: Dict,
        current_price: float,
        signal_action: str,
        signal_confidence: float
    ) -> Tuple[bool, str]:
        """Verifica condições de saída"""
        side = position['side']
        entry_price = position['entry_price']
        tp_price = position['tp_price']
        sl_price = position['sl_price']
        
        # Stop Loss / Take Profit
        if side == 'BUY':
            if current_price >= tp_price:
                return True, "Take Profit"
            elif current_price <= sl_price:
                return True, "Stop Loss"
        else:  # SELL
            if current_price <= tp_price:
                return True, "Take Profit"
            elif current_price >= sl_price:
                return True, "Stop Loss"
        
        # Sinal contrário forte
        if side == 'BUY' and signal_action == 'SELL' and signal_confidence > 0.8:
            return True, "Sinal Contrário"
        elif side == 'SELL' and signal_action == 'BUY' and signal_confidence > 0.8:
            return True, "Sinal Contrário"
        
        return False, ""
    
    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_idx: int,
        reason: str,
        current_balance: float
    ) -> Dict:
        """Fecha posição e calcula resultado"""
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        # Calcular P&L
        if side == 'BUY':
            pnl_gross = (exit_price - entry_price) * size
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SELL
            pnl_gross = (entry_price - exit_price) * size
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # Taxas (0.1% entrada + saída)
        entry_fee = entry_price * size * 0.001
        exit_fee = exit_price * size * 0.001
        total_fees = entry_fee + exit_fee
        
        pnl_net = pnl_gross - total_fees
        
        return {
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_idx': position['entry_idx'],
            'exit_idx': exit_idx,
            'entry_time': position['entry_time'],
            'size': size,
            'pnl_gross': pnl_gross,
            'pnl_net': pnl_net,
            'pnl_pct': pnl_pct,
            'fees': total_fees,
            'reason': reason,
            'duration': exit_idx - position['entry_idx'],
            'confidence': position['confidence']
        }
    
    def _calculate_metrics(
        self,
        trades: List[Dict],
        initial_balance: float,
        final_balance: float,
        equity_curve: List[Dict]
    ) -> Dict:
        """Calcula métricas detalhadas do backtest"""
        if not trades:
            logger.warning("Nenhum trade executado no backtest!")
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_fees': 0,
                'net_profit': 0
            }
        
        # Converter para DataFrame para análise
        df_trades = pd.DataFrame(trades)
        
        # Separar ganhos e perdas
        wins = df_trades[df_trades['pnl_net'] > 0]
        losses = df_trades[df_trades['pnl_net'] < 0]
        
        # Métricas básicas
        num_trades = len(trades)
        win_rate = len(wins) / num_trades
        
        # Profit factor
        total_wins = wins['pnl_net'].sum() if not wins.empty else 0
        total_losses = abs(losses['pnl_net'].sum()) if not losses.empty else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        # Médias
        avg_win = wins['pnl_net'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl_net'].mean()) if not losses.empty else 0
        
        # Retorno total
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Taxas
        total_fees = df_trades['fees'].sum()
        gross_profit = df_trades['pnl_gross'].sum()
        net_profit = df_trades['pnl_net'].sum()
        
        # Sharpe ratio
        returns = df_trades['pnl_pct'].values
        sharpe = self._calculate_sharpe(returns)
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown(equity_curve)
        
        # Estatísticas adicionais
        avg_trade_duration = df_trades['duration'].mean()
        max_consecutive_wins = self._count_consecutive(df_trades['pnl_net'] > 0)
        max_consecutive_losses = self._count_consecutive(df_trades['pnl_net'] < 0)
        
        # Win/loss por tipo
        buy_trades = df_trades[df_trades['side'] == 'BUY']
        sell_trades = df_trades[df_trades['side'] == 'SELL']
        
        buy_win_rate = (buy_trades['pnl_net'] > 0).mean() if not buy_trades.empty else 0
        sell_win_rate = (sell_trades['pnl_net'] > 0).mean() if not sell_trades.empty else 0
        
        # Log detalhado
        logger.info(f"""
📊 MÉTRICAS DETALHADAS DO BACKTEST
═══════════════════════════════════════════════
💰 FINANCEIRO:
• Capital inicial: ${initial_balance:,.2f}
• Capital final: ${final_balance:,.2f}
• Retorno Total: {total_return*100:+.2f}%
• Lucro Bruto: ${gross_profit:,.2f}
• Taxas Totais: ${total_fees:,.2f}
• Lucro Líquido: ${net_profit:,.2f}

📈 PERFORMANCE:
• Número de Trades: {num_trades}
• Taxa de Acerto: {win_rate*100:.1f}%
• Profit Factor: {profit_factor:.2f}
• Sharpe Ratio: {sharpe:.2f}
• Max Drawdown: {max_dd*100:.2f}%

💵 MÉDIAS:
• Ganho Médio: ${avg_win:.2f}
• Perda Média: ${avg_loss:.2f}
• Duração Média: {avg_trade_duration:.0f} candles

📊 ANÁLISE POR TIPO:
• BUY Win Rate: {buy_win_rate*100:.1f}% ({len(buy_trades)} trades)
• SELL Win Rate: {sell_win_rate*100:.1f}% ({len(sell_trades)} trades)

🔢 SEQUÊNCIAS:
• Máx. vitórias consecutivas: {max_consecutive_wins}
• Máx. derrotas consecutivas: {max_consecutive_losses}
═══════════════════════════════════════════════
        """)
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_fees': total_fees,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'avg_trade_duration': avg_trade_duration,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calcula Sharpe Ratio"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Assumir 252 períodos por ano (ajustar conforme timeframe)
        return np.sqrt(252) * mean_return / std_return
    
    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        """Calcula drawdown máximo"""
        if not equity_curve:
            return 0
        
        balances = [e['balance'] for e in equity_curve]
        
        peak = balances[0]
        max_dd = 0
        
        for balance in balances[1:]:
            if balance > peak:
                peak = balance
            else:
                dd = (peak - balance) / peak
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _count_consecutive(self, series: pd.Series) -> int:
        """Conta máximo de valores True consecutivos"""
        if series.empty:
            return 0
        
        max_count = 0
        current_count = 0
        
        for value in series:
            if value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count


async def run_backtest_validation(
    config = None,
    days: int = 7,
    debug_mode: bool = False
) -> Optional[Dict]:
    """
    Executa backtest para validação de estratégia
    """
    from trade_system.config import get_config
    from binance.client import Client
    
    # Configuração
    if config is None:
        config = get_config(debug_mode=debug_mode)
    
    logger.info(f"🔬 Executando backtest de validação ({days} dias)...")
    
    # Verificar credenciais
    if not config.api_key or not config.api_secret:
        logger.error("❌ Credenciais da Binance não configuradas")
        return None
    
    try:
        # Cliente Binance
        client = Client(config.api_key, config.api_secret)
        
        # Determinar intervalo
        if days <= 1:
            interval = Client.KLINE_INTERVAL_1MINUTE
            expected_candles = days * 24 * 60
        elif days <= 7:
            interval = Client.KLINE_INTERVAL_5MINUTE
            expected_candles = days * 24 * 12
        elif days <= 30:
            interval = Client.KLINE_INTERVAL_15MINUTE
            expected_candles = days * 24 * 4
        else:
            interval = Client.KLINE_INTERVAL_1HOUR
            expected_candles = days * 24
        
        # Limitar para não exceder limite da API
        limit = min(expected_candles, 1000)
        
        logger.info(f"📊 Baixando {limit} candles de {config.symbol}...")
        
        # Obter dados
        klines = client.get_klines(
            symbol=config.symbol,
            interval=interval,
            limit=limit
        )
        
        # Converter para DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Processar dados
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        logger.info(f"✅ Dados carregados: {len(df)} candles")
        logger.info(f"   Período: {df.index[0]} até {df.index[-1]}")
        logger.info(f"   Preço atual: ${df['close'].iloc[-1]:,.2f}")
        
        # Executar backtest
        backtester = IntegratedBacktester(config)
        results = await backtester.backtest_strategy(df)
        
        # Validar resultados
        if results and results['num_trades'] > 0:
            if results['win_rate'] < 0.40:
                logger.warning("⚠️ Taxa de acerto baixa no backtest")
            if results['profit_factor'] < 1.0:
                logger.warning("⚠️ Profit factor abaixo de 1.0")
            if results['max_drawdown'] > 0.20:
                logger.warning("⚠️ Drawdown máximo acima de 20%")
        else:
            logger.warning("⚠️ Nenhum trade gerado no backtest")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro no backtest: {e}")
        return None
