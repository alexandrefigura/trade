"""Sistema de gestão de risco"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
import numpy as np

class RiskManager:
    """Gerenciador de risco avançado"""
    
    def __init__(self, config: Any):
        """
        Inicializa o gerenciador de risco
        
        Args:
            config: Objeto TradingConfig ou dicionário de configuração
        """
        self.logger = logging.getLogger(__name__)
        
        # Extrair configurações de risco
        if hasattr(config, 'risk'):
            self.config = config.risk
        elif isinstance(config, dict) and 'risk' in config:
            self.config = config['risk']
        elif isinstance(config, dict):
            self.config = config
        else:
            # Configurações padrão se nada for fornecido
            self.config = {
                'max_volatility': 0.05,
                'max_spread_bps': 20,
                'max_daily_loss': 0.02,
                'stop_loss_pct': 0.015,
                'take_profit_pct': 0.03,
                'trailing_stop_pct': 0.01,
                'max_positions': 1,
                'position_timeout_hours': 24
            }
            self.logger.warning("Usando configurações de risco padrão")
        
        # Estado
        self.daily_stats = {
            'loss': 0.0,
            'trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'last_reset': datetime.now().date()
        }
        
        self.position_history = []
        self.current_positions = []
        
        self.logger.info("🛡️ Risk Manager inicializado")
        
    def validate_trade(self, signal: str, confidence: float, 
                      market_data: Dict[str, float]) -> Tuple[bool, str]:
        """
        Valida se um trade pode ser executado
        
        Args:
            signal: BUY, SELL ou HOLD
            confidence: Confiança do sinal (0-1)
            market_data: Dados de mercado incluindo volatilidade e spread
            
        Returns:
            Tuple (pode_executar, motivo)
        """
        # Reset diário se necessário
        self._check_daily_reset()
        
        # Verificações de validação
        checks = [
            self._check_confidence(confidence),
            self._check_daily_loss(),
            self._check_volatility(market_data.get('volatility', 0)),
            self._check_spread(market_data.get('spread_bps', 0)),
            self._check_position_limit(),
            self._check_market_hours(),
            self._check_momentum(market_data.get('momentum', 0))
        ]
        
        # Executar todas as verificações
        for is_valid, reason in checks:
            if not is_valid:
                self.logger.warning(f"❌ Trade rejeitado: {reason}")
                return False, reason
        
        self.logger.info(f"✅ Trade validado: {signal} com confiança {confidence:.2%}")
        return True, "Trade aprovado"
    
    def _check_confidence(self, confidence: float) -> Tuple[bool, str]:
        """Verifica confiança mínima"""
        min_confidence = self.config.get('min_confidence', 0.75)
        if confidence < min_confidence:
            return False, f"Confiança baixa: {confidence:.2%} < {min_confidence:.2%}"
        return True, ""
    
    def _check_daily_loss(self) -> Tuple[bool, str]:
        """Verifica limite de perda diária"""
        max_daily_loss = self.config.get('max_daily_loss', 0.02)
        if abs(self.daily_stats['loss']) >= max_daily_loss:
            return False, f"Limite de perda diária atingido: {self.daily_stats['loss']:.2%}"
        return True, ""
    
    def _check_volatility(self, volatility: float) -> Tuple[bool, str]:
        """Verifica volatilidade do mercado"""
        max_volatility = self.config.get('max_volatility', 0.05)
        if volatility > max_volatility:
            return False, f"Volatilidade muito alta: {volatility:.2%} > {max_volatility:.2%}"
        return True, ""
    
    def _check_spread(self, spread_bps: float) -> Tuple[bool, str]:
        """Verifica spread bid-ask"""
        max_spread = self.config.get('max_spread_bps', 20)
        if spread_bps > max_spread:
            return False, f"Spread muito alto: {spread_bps:.1f} bps > {max_spread} bps"
        return True, ""
    
    def _check_position_limit(self) -> Tuple[bool, str]:
        """Verifica limite de posições abertas"""
        max_positions = self.config.get('max_positions', 1)
        if len(self.current_positions) >= max_positions:
            return False, f"Limite de posições atingido: {len(self.current_positions)}/{max_positions}"
        return True, ""
    
    def _check_market_hours(self) -> Tuple[bool, str]:
        """Verifica horário de trading"""
        current_hour = datetime.now().hour
        # Evitar horários de baixa liquidez (2-6 UTC)
        if 2 <= current_hour <= 6:
            return False, "Horário de baixa liquidez (2-6 UTC)"
        return True, ""
    
    def _check_momentum(self, momentum: float) -> Tuple[bool, str]:
        """Verifica momentum do mercado"""
        # Se momentum muito negativo, evitar compras
        if momentum < -5:
            return False, f"Momentum muito negativo: {momentum:.2f}"
        return True, ""
    
    def calculate_position_size(self, balance: float, price: float, 
                              confidence: float = 0.75) -> float:
        """
        Calcula tamanho da posição usando Kelly Criterion modificado
        
        Args:
            balance: Saldo disponível
            price: Preço atual do ativo
            confidence: Confiança do sinal
            
        Returns:
            Tamanho da posição em unidades do ativo
        """
        # Kelly Criterion: f = (p*b - q) / b
        # p = probabilidade de ganho, q = probabilidade de perda, b = odds
        
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_avg_win()
        avg_loss = self._calculate_avg_loss()
        
        if avg_loss == 0:
            odds = 2.0  # Default
        else:
            odds = avg_win / abs(avg_loss)
        
        # Kelly fraction
        kelly = (win_rate * odds - (1 - win_rate)) / odds
        
        # Aplicar fator de segurança (25% do Kelly)
        kelly_safe = kelly * 0.25
        
        # Ajustar por confiança
        kelly_adjusted = kelly_safe * confidence
        
        # Limitar ao máximo configurado
        max_position_pct = self.config.get('max_position_pct', 0.02)
        position_pct = min(kelly_adjusted, max_position_pct)
        
        # Garantir posição mínima
        position_size = max(position_pct * balance / price, 10 / price)
        
        self.logger.info(f"📊 Position size: {position_size:.8f} ({position_pct:.2%} do balance)")
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, signal: str, 
                          volatility: float = 0.01) -> float:
        """Calcula stop loss dinâmico baseado em volatilidade"""
        base_stop = self.config.get('stop_loss_pct', 0.015)
        
        # Ajustar stop por volatilidade
        volatility_multiplier = max(1, volatility / 0.01)
        adjusted_stop = base_stop * volatility_multiplier
        
        if signal == 'BUY':
            return entry_price * (1 - adjusted_stop)
        else:  # SELL
            return entry_price * (1 + adjusted_stop)
    
    def calculate_take_profit(self, entry_price: float, signal: str,
                            confidence: float = 0.75) -> float:
        """Calcula take profit baseado em confiança"""
        base_tp = self.config.get('take_profit_pct', 0.03)
        
        # Ajustar TP por confiança
        confidence_multiplier = 0.5 + confidence  # 0.5x a 1.5x
        adjusted_tp = base_tp * confidence_multiplier
        
        if signal == 'BUY':
            return entry_price * (1 + adjusted_tp)
        else:  # SELL
            return entry_price * (1 - adjusted_tp)
    
    def update_trailing_stop(self, position: Dict, current_price: float) -> float:
        """Atualiza trailing stop de uma posição"""
        trailing_pct = self.config.get('trailing_stop_pct', 0.01)
        
        if position['type'] == 'BUY':
            new_stop = current_price * (1 - trailing_pct)
            if new_stop > position['stop_loss']:
                self.logger.info(f"📈 Trailing stop atualizado: ${new_stop:.2f}")
                return new_stop
        else:  # SELL
            new_stop = current_price * (1 + trailing_pct)
            if new_stop < position['stop_loss']:
                self.logger.info(f"📉 Trailing stop atualizado: ${new_stop:.2f}")
                return new_stop
        
        return position['stop_loss']
    
    def register_position(self, position: Dict):
        """Registra nova posição"""
        position['open_time'] = datetime.now()
        position['timeout_time'] = datetime.now() + timedelta(
            hours=self.config.get('position_timeout_hours', 24)
        )
        self.current_positions.append(position)
        self.daily_stats['trades'] += 1
    
    def close_position(self, position: Dict, exit_price: float, reason: str):
        """Fecha posição e atualiza estatísticas"""
        # Calcular resultado
        if position['type'] == 'BUY':
            profit_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:  # SELL
            profit_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        # Atualizar estatísticas
        self.daily_stats['loss'] -= profit_pct  # Negativo se for lucro
        
        if profit_pct > 0:
            self.daily_stats['winning_trades'] += 1
        else:
            self.daily_stats['losing_trades'] += 1
        
        # Registrar no histórico
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['profit_pct'] = profit_pct
        position['exit_reason'] = reason
        self.position_history.append(position)
        
        # Remover das posições abertas
        self.current_positions = [p for p in self.current_positions 
                                 if p != position]
        
        self.logger.info(f"💰 Posição fechada: {profit_pct:+.2%} ({reason})")
    
    def check_position_timeout(self, position: Dict) -> bool:
        """Verifica se posição expirou"""
        return datetime.now() > position.get('timeout_time', datetime.max)
    
    def _calculate_win_rate(self) -> float:
        """Calcula taxa de acerto histórica"""
        if not self.position_history:
            return 0.5  # Default
        
        winning = sum(1 for p in self.position_history if p['profit_pct'] > 0)
        return winning / len(self.position_history)
    
    def _calculate_avg_win(self) -> float:
        """Calcula ganho médio"""
        wins = [p['profit_pct'] for p in self.position_history if p['profit_pct'] > 0]
        return np.mean(wins) if wins else 0.02  # Default 2%
    
    def _calculate_avg_loss(self) -> float:
        """Calcula perda média"""
        losses = [p['profit_pct'] for p in self.position_history if p['profit_pct'] <= 0]
        return np.mean(losses) if losses else -0.01  # Default -1%
    
    def _check_daily_reset(self):
        """Reseta estatísticas diárias se necessário"""
        if datetime.now().date() > self.daily_stats['last_reset']:
            self.daily_stats = {
                'loss': 0.0,
                'trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'last_reset': datetime.now().date()
            }
            self.logger.info("📊 Estatísticas diárias resetadas")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de risco atuais"""
        return {
            'daily_loss': self.daily_stats['loss'],
            'daily_trades': self.daily_stats['trades'],
            'win_rate': self._calculate_win_rate(),
            'avg_win': self._calculate_avg_win(),
            'avg_loss': self._calculate_avg_loss(),
            'open_positions': len(self.current_positions),
            'total_trades': len(self.position_history)
        }
