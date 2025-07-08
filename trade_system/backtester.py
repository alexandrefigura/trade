"""
Sistema avançado de backtesting com walk-forward analysis e simulação realista
"""
import asyncio
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from random import uniform
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from trade_system.logging_config import get_logger
from trade_system.analysis.technical import UltraFastTechnicalAnalysis
from trade_system.analysis.ml import SimplifiedMLPredictor, AdvancedMLPredictor
from trade_system.signals import OptimizedSignalConsolidator
from trade_system.utils import calculate_atr

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuração para backtesting"""
    # Walk-forward
    train_window_days: int = 30
    test_window_days: int = 7
    step_days: int = 1
    min_train_samples: int = 1000
    
    # Simulação de mercado
    slippage_pct: float = 0.0005  # 0.05%
    maker_fee_pct: float = 0.0002  # 0.02%
    taker_fee_pct: float = 0.0004  # 0.04%
    max_latency_ms: float = 50.0  # Máximo 50ms
    order_rejection_rate: float = 0.01  # 1% de rejeição
    partial_fill_rate: float = 0.05  # 5% preenchimento parcial
    
    # Limites de posição
    max_position_pct: float = 0.02
    min_position_value: float = 50.0
    max_positions: int = 1
    
    # Relatórios
    export_trades: bool = True
    export_equity_curve: bool = True
    export_metrics: bool = True
    generate_plots: bool = True
    output_dir: str = "backtest_results"


@dataclass
class WalkForwardWindow:
    """Janela de walk-forward"""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_id: int
    
    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days
    
    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days


class AdvancedBacktester:
    """Sistema avançado de backtesting com walk-forward e simulação realista"""
    
    def __init__(self, config: BacktestConfig, trading_config = None):
        self.config = config
        self.trading_config = trading_config
        self.results = []
        self.walk_forward_results = []
        
        # Criar diretório de saída
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Timestamp para arquivos
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"🚀 Backtester inicializado - Output: {self.output_dir}")
    
    async def run_walk_forward_analysis(
        self,
        historical_data: pd.DataFrame,
        initial_balance: float = 10000,
        parallel: bool = True
    ) -> Dict:
        """
        Executa análise walk-forward completa
        
        Args:
            historical_data: DataFrame com OHLCV
            initial_balance: Capital inicial
            parallel: Usar processamento paralelo
            
        Returns:
            Resultados consolidados
        """
        logger.info("🔄 Iniciando Walk-Forward Analysis...")
        
        # Gerar janelas
        windows = self._generate_walk_forward_windows(historical_data)
        logger.info(f"📊 Geradas {len(windows)} janelas walk-forward")
        
        # Executar backtests
        if parallel and len(windows) > 1:
            results = await self._run_parallel_backtests(
                historical_data, windows, initial_balance
            )
        else:
            results = await self._run_sequential_backtests(
                historical_data, windows, initial_balance
            )
        
        # Consolidar resultados
        consolidated = self._consolidate_walk_forward_results(results)
        
        # Exportar relatórios
        if self.config.export_metrics:
            self._export_walk_forward_report(results, consolidated)
        
        if self.config.generate_plots:
            self._generate_walk_forward_plots(results, consolidated)
        
        return consolidated
    
    def _generate_walk_forward_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """Gera janelas de walk-forward"""
        windows = []
        
        # Índices de tempo
        timestamps = data.index
        start_date = timestamps[0]
        end_date = timestamps[-1]
        
        # Configurações
        train_days = self.config.train_window_days
        test_days = self.config.test_window_days
        step_days = self.config.step_days
        
        # Gerar janelas
        current_start = start_date
        window_id = 0
        
        while current_start + timedelta(days=train_days + test_days) <= end_date:
            train_start = current_start
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)
            
            # Verificar se há dados suficientes
            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]
            
            if len(train_data) >= self.config.min_train_samples and len(test_data) > 0:
                windows.append(WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id
                ))
                window_id += 1
            
            # Avançar
            current_start += timedelta(days=step_days)
        
        return windows
    
    async def _run_parallel_backtests(
        self,
        data: pd.DataFrame,
        windows: List[WalkForwardWindow],
        initial_balance: float
    ) -> List[Dict]:
        """Executa backtests em paralelo"""
        results = []
        
        with ProcessPoolExecutor() as executor:
            # Submeter tarefas
            futures = {}
            for window in windows:
                future = executor.submit(
                    self._run_single_window_backtest,
                    data, window, initial_balance
                )
                futures[future] = window
            
            # Coletar resultados
            for future in as_completed(futures):
                window = futures[future]
                try:
                    result = future.result()
                    result['window'] = window
                    results.append(result)
                    logger.info(f"✅ Janela {window.window_id} concluída")
                except Exception as e:
                    logger.error(f"❌ Erro na janela {window.window_id}: {e}")
        
        return sorted(results, key=lambda x: x['window'].window_id)
    
    async def _run_sequential_backtests(
        self,
        data: pd.DataFrame,
        windows: List[WalkForwardWindow],
        initial_balance: float
    ) -> List[Dict]:
        """Executa backtests sequencialmente"""
        results = []
        
        for i, window in enumerate(windows):
            logger.info(f"🔄 Processando janela {i+1}/{len(windows)}")
            
            result = await self._run_single_window_backtest_async(
                data, window, initial_balance
            )
            result['window'] = window
            results.append(result)
        
        return results
    
    def _run_single_window_backtest(
        self,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        initial_balance: float
    ) -> Dict:
        """Executa backtest para uma única janela (síncrono para parallelização)"""
        # Dividir dados
        train_data = data[window.train_start:window.train_end]
        test_data = data[window.test_start:window.test_end]
        
        # Treinar modelo se aplicável
        ml_predictor = self._train_ml_model(train_data)
        
        # Executar backtest no período de teste
        backtest_result = self._run_backtest_core(
            test_data, initial_balance, ml_predictor
        )
        
        # Adicionar informações da janela
        backtest_result['train_samples'] = len(train_data)
        backtest_result['test_samples'] = len(test_data)
        
        return backtest_result
    
    async def _run_single_window_backtest_async(
        self,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        initial_balance: float
    ) -> Dict:
        """Versão assíncrona para execução sequencial"""
        return self._run_single_window_backtest(data, window, initial_balance)
    
    def _train_ml_model(self, train_data: pd.DataFrame) -> Optional[object]:
        """Treina modelo ML com dados de treino"""
        if not self.trading_config or not hasattr(self.trading_config, 'ml'):
            return SimplifiedMLPredictor()
        
        try:
            # Usar AdvancedMLPredictor se disponível
            ml_predictor = AdvancedMLPredictor(model_type='auto')
            
            # Extrair features e labels dos dados de treino
            features_list = []
            labels_list = []
            
            prices = train_data['close'].values
            volumes = train_data['volume'].values
            
            for i in range(100, len(prices) - 1):
                # Features
                features = self._extract_features_for_training(
                    prices, volumes, i
                )
                features_list.append(features)
                
                # Label (direção do próximo movimento)
                price_change = (prices[i+1] - prices[i]) / prices[i]
                if price_change > 0.001:
                    labels_list.append('BUY')
                elif price_change < -0.001:
                    labels_list.append('SELL')
                else:
                    labels_list.append('HOLD')
            
            # Adicionar amostras de treinamento
            for features, label in zip(features_list, labels_list):
                ml_predictor.add_training_sample(features, label)
            
            # Forçar retreinamento
            if hasattr(ml_predictor, '_retrain_model'):
                ml_predictor._retrain_model()
            
            return ml_predictor
            
        except Exception as e:
            logger.error(f"Erro treinando ML: {e}")
            return SimplifiedMLPredictor()
    
    def _run_backtest_core(
        self,
        test_data: pd.DataFrame,
        initial_balance: float,
        ml_predictor = None
    ) -> Dict:
        """Núcleo do backtest com simulação realista"""
        # Preparar dados
        balance = initial_balance
        position = None
        trades = []
        equity_curve = []
        
        # Arrays NumPy
        prices = test_data['close'].values.astype(np.float32)
        volumes = test_data['volume'].values.astype(np.float32)
        highs = test_data.get('high', prices).values.astype(np.float32)
        lows = test_data.get('low', prices).values.astype(np.float32)
        
        # ATR
        atr_series = calculate_atr(highs, lows, prices, period=14)
        
        # Componentes
        technical_analyzer = UltraFastTechnicalAnalysis(self.trading_config)
        if ml_predictor is None:
            ml_predictor = SimplifiedMLPredictor()
        consolidator = OptimizedSignalConsolidator()
        
        # Estatísticas de simulação
        total_latency_ms = 0
        rejected_orders = 0
        partial_fills = 0
        
        # Loop principal
        for i in range(200, len(prices)):
            # Simular latência de rede
            if self.config.max_latency_ms > 0:
                latency = uniform(0, self.config.max_latency_ms)
                time.sleep(latency / 1000.0)  # Converter para segundos
                total_latency_ms += latency
            
            # Análise técnica
            price_slice = prices[:i+1]
            volume_slice = volumes[:i+1]
            
            tech_action, tech_conf, tech_details = technical_analyzer.analyze(
                price_slice[-1000:],
                volume_slice[-1000:]
            )
            
            # Features ML
            features = self._extract_features_for_trading(
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
            
            # Preço atual com slippage
            current_price = self._apply_slippage(prices[i], action)
            
            # Gerenciar posições
            if position is None and action != 'HOLD':
                # Simular rejeição de ordem
                if uniform(0, 1) < self.config.order_rejection_rate:
                    rejected_orders += 1
                    continue
                
                # Calcular tamanho
                position_size = self._calculate_realistic_position_size(
                    balance, confidence, features.get('volatility', 0.01)
                )
                
                if position_size > 0:
                    # Simular preenchimento parcial
                    if uniform(0, 1) < self.config.partial_fill_rate:
                        position_size *= uniform(0.5, 0.9)
                        partial_fills += 1
                    
                    # Calcular stops com ATR
                    atr = atr_series[i] if i < len(atr_series) else prices[i] * 0.02
                    
                    position = {
                        'side': action,
                        'entry_price': current_price,
                        'entry_idx': i,
                        'size': position_size / current_price,
                        'tp_price': current_price + (atr * 2 if action == 'BUY' else -atr * 2),
                        'sl_price': current_price - (atr * 1.5 if action == 'BUY' else -atr * 1.5),
                        'confidence': confidence,
                        'entry_fee': position_size * self.config.taker_fee_pct
                    }
                    
                    balance -= position['entry_fee']
            
            elif position is not None:
                # Verificar saída
                should_close, reason = self._check_realistic_exit(
                    position, current_price, action, confidence, i
                )
                
                if should_close:
                    # Fechar com slippage
                    exit_price = self._apply_slippage(current_price, 
                                                    'SELL' if position['side'] == 'BUY' else 'BUY')
                    
                    trade_result = self._close_realistic_position(
                        position, exit_price, i, reason, balance
                    )
                    
                    trades.append(trade_result)
                    balance += trade_result['pnl_net']
                    position = None
            
            # Registrar equity
            equity_curve.append({
                'index': i,
                'timestamp': test_data.index[i],
                'balance': balance,
                'in_position': position is not None,
                'price': current_price
            })
        
        # Estatísticas de simulação
        simulation_stats = {
            'avg_latency_ms': total_latency_ms / len(prices) if len(prices) > 0 else 0,
            'rejected_orders': rejected_orders,
            'partial_fills': partial_fills,
            'rejection_rate': rejected_orders / (len(trades) + rejected_orders) if trades else 0,
            'partial_fill_rate': partial_fills / len(trades) if trades else 0
        }
        
        # Calcular métricas
        metrics = self._calculate_enhanced_metrics(
            trades, initial_balance, balance, equity_curve, simulation_stats
        )
        
        return metrics
    
    def _apply_slippage(self, price: float, action: str) -> float:
        """Aplica slippage ao preço"""
        slippage = self.config.slippage_pct
        
        # Slippage sempre contra o trader
        if action == 'BUY':
            return price * (1 + slippage)
        elif action == 'SELL':
            return price * (1 - slippage)
        else:
            return price
    
    def _calculate_realistic_position_size(
        self,
        balance: float,
        confidence: float,
        volatility: float
    ) -> float:
        """Calcula tamanho com limites realistas"""
        # Base size
        base_pct = self.config.max_position_pct * confidence
        
        # Ajustar por volatilidade
        if volatility > 0.03:
            base_pct *= 0.5
        elif volatility > 0.02:
            base_pct *= 0.7
        
        position_value = balance * base_pct
        
        # Limites
        if position_value < self.config.min_position_value:
            return 0.0
        
        max_value = balance * 0.1  # Máximo 10%
        return min(position_value, max_value)
    
    def _check_realistic_exit(
        self,
        position: Dict,
        current_price: float,
        signal_action: str,
        signal_confidence: float,
        current_idx: int
    ) -> Tuple[bool, str]:
        """Verifica saída com lógica realista"""
        # Stops padrão
        if position['side'] == 'BUY':
            if current_price >= position['tp_price']:
                return True, "Take Profit"
            elif current_price <= position['sl_price']:
                return True, "Stop Loss"
        else:
            if current_price <= position['tp_price']:
                return True, "Take Profit"
            elif current_price >= position['sl_price']:
                return True, "Stop Loss"
        
        # Tempo na posição
        bars_in_position = current_idx - position['entry_idx']
        if bars_in_position > 100:  # Máximo 100 barras
            return True, "Tempo máximo"
        
        # Sinal contrário forte
        opposite_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
        if signal_action == opposite_side and signal_confidence > 0.8:
            return True, "Sinal contrário"
        
        return False, ""
    
    def _close_realistic_position(
        self,
        position: Dict,
        exit_price: float,
        exit_idx: int,
        reason: str,
        current_balance: float
    ) -> Dict:
        """Fecha posição com cálculos realistas"""
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        # P&L bruto
        if side == 'BUY':
            pnl_gross = (exit_price - entry_price) * size
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_gross = (entry_price - exit_price) * size
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # Taxas
        exit_fee = exit_price * size * self.config.taker_fee_pct
        total_fees = position['entry_fee'] + exit_fee
        
        pnl_net = pnl_gross - exit_fee
        
        return {
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_idx': position['entry_idx'],
            'exit_idx': exit_idx,
            'size': size,
            'pnl_gross': pnl_gross,
            'pnl_net': pnl_net,
            'pnl_pct': pnl_pct,
            'fees': total_fees,
            'reason': reason,
            'duration_bars': exit_idx - position['entry_idx'],
            'confidence': position['confidence']
        }
    
    def _extract_features_for_training(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        idx: int
    ) -> Dict:
        """Extrai features para treinamento do ML"""
        features = {}
        
        # RSI
        if idx >= 14:
            gains = []
            losses = []
            for i in range(idx-14, idx):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
            else:
                features['rsi'] = 100
        else:
            features['rsi'] = 50
        
        # Momentum
        if idx >= 20:
            features['momentum'] = (prices[idx] - prices[idx-20]) / prices[idx-20]
        else:
            features['momentum'] = 0
        
        # Volume ratio
        if idx >= 20:
            avg_volume = np.mean(volumes[max(0, idx-20):idx])
            features['volume_ratio'] = volumes[idx] / avg_volume if avg_volume > 0 else 1
        else:
            features['volume_ratio'] = 1
        
        # Volatilidade
        if idx >= 50:
            features['volatility'] = np.std(prices[max(0, idx-50):idx]) / np.mean(prices[max(0, idx-50):idx])
        else:
            features['volatility'] = 0.01
        
        # Spread simulado
        features['spread_bps'] = uniform(3, 15)
        
        return features
    
    def _extract_features_for_trading(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        idx: int,
        tech_details: Dict
    ) -> Dict:
        """Extrai features para trading (inclui detalhes técnicos)"""
        features = self._extract_features_for_training(prices, volumes, idx)
        
        # Adicionar detalhes técnicos se disponíveis
        if tech_details:
            features['rsi'] = tech_details.get('rsi', features['rsi'])
            if 'sma_cross' in tech_details:
                features['sma_cross'] = tech_details['sma_cross']
            if 'bb_position' in tech_details:
                features['bb_position'] = tech_details['bb_position']
        
        return features
    
    def _calculate_enhanced_metrics(
        self,
        trades: List[Dict],
        initial_balance: float,
        final_balance: float,
        equity_curve: List[Dict],
        simulation_stats: Dict
    ) -> Dict:
        """Calcula métricas expandidas incluindo estatísticas de simulação"""
        if not trades:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_fees': 0,
                'net_profit': 0,
                'simulation_stats': simulation_stats
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Métricas básicas
        wins = df_trades[df_trades['pnl_net'] > 0]
        losses = df_trades[df_trades['pnl_net'] < 0]
        
        num_trades = len(trades)
        win_rate = len(wins) / num_trades
        
        total_wins = wins['pnl_net'].sum() if not wins.empty else 0
        total_losses = abs(losses['pnl_net'].sum()) if not losses.empty else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        avg_win = wins['pnl_net'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl_net'].mean()) if not losses.empty else 0
        
        total_return = (final_balance - initial_balance) / initial_balance
        total_fees = df_trades['fees'].sum()
        net_profit = df_trades['pnl_net'].sum()
        
        # Sharpe Ratio
        returns = df_trades['pnl_pct'].values
        sharpe = self._calculate_sharpe(returns)
        
        # Sortino Ratio
        sortino = self._calculate_sortino(returns)
        
        # Calmar Ratio
        max_dd = self._calculate_max_drawdown(equity_curve)
        annual_return = total_return * (252 / len(equity_curve)) if len(equity_curve) > 0 else 0
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        # Estatísticas adicionais
        avg_duration = df_trades['duration_bars'].mean()
        win_duration = wins['duration_bars'].mean() if not wins.empty else 0
        loss_duration = losses['duration_bars'].mean() if not losses.empty else 0
        
        # Recovery factor
        recovery_factor = net_profit / (max_dd * initial_balance) if max_dd > 0 else np.inf
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'recovery_factor': recovery_factor,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_duration': avg_duration,
            'win_duration': win_duration,
            'loss_duration': loss_duration,
            'total_fees': total_fees,
            'net_profit': net_profit,
            'trades': trades,
            'equity_curve': equity_curve,
            'simulation_stats': simulation_stats
        }
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calcula Sharpe Ratio"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        return np.sqrt(252) * mean_return / std_return
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calcula Sortino Ratio (penaliza apenas volatilidade negativa)"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0
        
        return np.sqrt(252) * mean_return / downside_std
    
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
    
    def _consolidate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Consolida resultados de múltiplas janelas walk-forward"""
        if not results:
            return {}
        
        # Agregar métricas
        total_trades = sum(r['num_trades'] for r in results)
        
        # Métricas ponderadas por número de trades
        weighted_metrics = {}
        for metric in ['win_rate', 'sharpe_ratio', 'sortino_ratio', 'profit_factor']:
            weighted_sum = sum(r[metric] * r['num_trades'] for r in results if r['num_trades'] > 0)
            weighted_metrics[metric] = weighted_sum / total_trades if total_trades > 0 else 0
        
        # Máximos e mínimos
        max_drawdown = max(r['max_drawdown'] for r in results)
        min_return = min(r['total_return'] for r in results)
        max_return = max(r['total_return'] for r in results)
        
        # Estatísticas de estabilidade
        returns = [r['total_return'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        
        stability_metrics = {
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'return_cv': np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else np.inf,
            'sharpe_mean': np.mean(sharpes),
            'sharpe_std': np.std(sharpes),
            'consistency_ratio': sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        }
        
        # Consolidar trades
        all_trades = []
        for result in results:
            if 'trades' in result:
                all_trades.extend(result['trades'])
        
        # Análise por período
        period_analysis = []
        for i, result in enumerate(results):
            window = result.get('window')
            period_analysis.append({
                'window_id': window.window_id if window else i,
                'period': f"{window.test_start.date()} to {window.test_end.date()}" if window else f"Period {i}",
                'return': result['total_return'],
                'trades': result['num_trades'],
                'win_rate': result['win_rate'],
                'sharpe': result['sharpe_ratio'],
                'drawdown': result['max_drawdown']
            })
        
        return {
            'summary': {
                'total_windows': len(results),
                'total_trades': total_trades,
                'avg_trades_per_window': total_trades / len(results) if results else 0,
                **weighted_metrics,
                'max_drawdown': max_drawdown,
                'return_range': (min_return, max_return),
                **stability_metrics
            },
            'period_analysis': period_analysis,
            'all_trades': all_trades,
            'individual_results': results
        }
    
    def _export_walk_forward_report(self, results: List[Dict], consolidated: Dict):
        """Exporta relatório detalhado para JSON e CSV"""
        timestamp = self.run_timestamp
        
        # JSON completo
        json_file = self.output_dir / f"walk_forward_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'summary': consolidated['summary'],
                'period_analysis': consolidated['period_analysis'],
                'detailed_results': [
                    {k: v for k, v in r.items() if k not in ['trades', 'equity_curve']}
                    for r in results
                ]
            }, f, indent=2, default=str)
        
        logger.info(f"📄 Relatório JSON salvo: {json_file}")
        
        # CSV resumido
        csv_file = self.output_dir / f"walk_forward_summary_{timestamp}.csv"
        df_summary = pd.DataFrame(consolidated['period_analysis'])
        df_summary.to_csv(csv_file, index=False)
        
        logger.info(f"📊 Resumo CSV salvo: {csv_file}")
        
        # CSV de trades
        if self.config.export_trades and consolidated['all_trades']:
            trades_file = self.output_dir / f"all_trades_{timestamp}.csv"
            df_trades = pd.DataFrame(consolidated['all_trades'])
            df_trades.to_csv(trades_file, index=False)
            logger.info(f"💹 Trades CSV salvo: {trades_file}")
    
    def _generate_walk_forward_plots(self, results: List[Dict], consolidated: Dict):
        """Gera visualizações dos resultados"""
        timestamp = self.run_timestamp
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Figure com subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Analysis Results', fontsize=16)
        
        # 1. Returns por período
        ax1 = axes[0, 0]
        periods = [r.get('window', {}).window_id for r in results]
        returns = [r['total_return'] * 100 for r in results]
        
        ax1.bar(periods, returns, color=['green' if r > 0 else 'red' for r in returns])
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Window ID')
        ax1.set_ylabel('Return (%)')
        ax1.set_title('Returns by Period')
        
        # 2. Métricas de performance
        ax2 = axes[0, 1]
        metrics = ['win_rate', 'sharpe_ratio', 'profit_factor']
        metric_values = {m: [r[m] for r in results] for m in metrics}
        
        for metric, values in metric_values.items():
            ax2.plot(periods, values, marker='o', label=metric)
        
        ax2.set_xlabel('Window ID')
        ax2.set_ylabel('Value')
        ax2.set_title('Performance Metrics Evolution')
        ax2.legend()
        
        # 3. Drawdown
        ax3 = axes[1, 0]
        drawdowns = [r['max_drawdown'] * 100 for r in results]
        
        ax3.plot(periods, drawdowns, marker='o', color='red')
        ax3.fill_between(periods, drawdowns, alpha=0.3, color='red')
        ax3.set_xlabel('Window ID')
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.set_title('Maximum Drawdown by Period')
        
        # 4. Distribuição de returns
        ax4 = axes[1, 1]
        ax4.hist(returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=np.mean(returns), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(returns):.1f}%')
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Return Distribution')
        ax4.legend()
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar
        plot_file = self.output_dir / f"walk_forward_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Gráficos salvos: {plot_file}")
        
        # Gráfico adicional: Equity curve combinada
        if any('equity_curve' in r for r in results):
            self._plot_combined_equity_curve(results, timestamp)
    
    def _plot_combined_equity_curve(self, results: List[Dict], timestamp: str):
        """Plota equity curve combinada de todos os períodos"""
        plt.figure(figsize=(12, 6))
        
        combined_balance = []
        
        for i, result in enumerate(results):
            if 'equity_curve' in result and result['equity_curve']:
                curve = result['equity_curve']
                balances = [e['balance'] for e in curve]
                
                if i == 0:
                    combined_balance.extend(balances)
                else:
                    # Ajustar para continuar do último balance
                    last_balance = combined_balance[-1]
                    first_balance = balances[0]
                    adjustment = last_balance - first_balance
                    adjusted_balances = [b + adjustment for b in balances[1:]]
                    combined_balance.extend(adjusted_balances)
        
        if combined_balance:
            plt.plot(combined_balance, linewidth=2)
            plt.title('Combined Equity Curve - Walk-Forward Analysis')
            plt.xlabel('Time Steps')
            plt.ylabel('Balance ($)')
            plt.grid(True, alpha=0.3)
            
            # Adicionar estatísticas
            initial = combined_balance[0]
            final = combined_balance[-1]
            total_return = (final - initial) / initial * 100
            
            plt.text(0.02, 0.95, f'Total Return: {total_return:.1f}%', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plot_file = self.output_dir / f"equity_curve_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 Equity curve salva: {plot_file}")


# Manter compatibilidade
class IntegratedBacktester(AdvancedBacktester):
    """Alias para manter compatibilidade com código existente"""
    
    def __init__(self, config):
        # Converter config antiga para BacktestConfig
        backtest_config = BacktestConfig(
            max_position_pct=getattr(config, 'max_position_pct', 0.02),
            min_position_value=50.0,
            output_dir="backtest_results"
        )
        super().__init__(backtest_config, config)
    
    async def backtest_strategy(
        self,
        historical_data: pd.DataFrame,
        initial_balance: float = 10000
    ) -> Dict:
        """Wrapper para manter interface antiga"""
        # Executar um único backtest sem walk-forward
        result = self._run_backtest_core(
            historical_data,
            initial_balance,
            None
        )
        
        # Log simplificado
        if result['num_trades'] > 0:
            logger.info(f"""
📊 Backtest concluído:
- Trades: {result['num_trades']}
- Win Rate: {result['win_rate']*100:.1f}%
- Return: {result['total_return']*100:.1f}%
- Sharpe: {result['sharpe_ratio']:.2f}
- Max DD: {result['max_drawdown']*100:.1f}%
            """)
        
        return result


async def run_walk_forward_validation(
    config = None,
    days: int = 90,
    train_days: int = 30,
    test_days: int = 7,
    step_days: int = 1
) -> Optional[Dict]:
    """
    Executa validação walk-forward completa
    
    Args:
        config: Configuração de trading
        days: Total de dias de dados
        train_days: Dias para treinamento
        test_days: Dias para teste
        step_days: Dias para avançar entre janelas
    """
    from trade_system.config import get_config
    from binance.client import Client
    
    # Configuração
    if config is None:
        config = get_config()
    
    logger.info(f"🔬 Executando Walk-Forward Analysis ({days} dias)...")
    
    try:
        # Cliente Binance
        client = Client(config.api_key, config.api_secret)
        
        # Obter dados
        interval = Client.KLINE_INTERVAL_15MINUTE
        limit = min(days * 96, 1000)  # 96 candles por dia em 15min
        
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
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        logger.info(f"✅ Dados carregados: {len(df)} candles")
        
        # Configurar backtester
        backtest_config = BacktestConfig(
            train_window_days=train_days,
            test_window_days=test_days,
            step_days=step_days,
            slippage_pct=0.0005,
            max_latency_ms=50,
            export_metrics=True,
            generate_plots=True
        )
        
        backtester = AdvancedBacktester(backtest_config, config)
        
        # Executar walk-forward
        results = await backtester.run_walk_forward_analysis(
            df,
            initial_balance=10000,
            parallel=False  # Evitar problemas com async
        )
        
        # Validar resultados
        if results and 'summary' in results:
            summary = results['summary']
            logger.info(f"""
🎯 Walk-Forward Analysis Completa:
- Janelas testadas: {summary['total_windows']}
- Total de trades: {summary['total_trades']}
- Win rate médio: {summary['win_rate']*100:.1f}%
- Sharpe médio: {summary['sharpe_mean']:.2f} (±{summary['sharpe_std']:.2f})
- Consistência: {summary['consistency_ratio']*100:.1f}%
- Max drawdown: {summary['max_drawdown']*100:.1f}%
            """)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro no walk-forward: {e}")
        return None