"""
Gestão de risco ultra-rápida e validação de condições de mercado
"""
import time
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class UltraFastRiskManager:
    """Gestão de risco com cálculos otimizados"""
    
    def __init__(self, config):
        self.config = config
        self.current_balance = 10000.0  # Balance inicial padrão
        self.initial_balance = 10000.0
        self.daily_pnl = 0.0
        self.max_daily_loss = config.max_daily_loss
        self.position_info = None
        
        # Histórico em array NumPy
        self.pnl_history = np.zeros(1000, dtype=np.float32)
        self.pnl_index = 0
        
        # Métricas
        self.total_fees_paid = 0.0
        self.peak_balance = 10000.0
        self.drawdown_start = None
        self.max_drawdown = 0.0
        self.daily_trades = 0
        self.last_reset_day = datetime.now().date()
        
        # Limites de risco
        self.max_position_value = None
        self.max_positions = 1
        self.current_positions = 0
        
        logger.info(f"💰 Risk Manager inicializado - Balance: ${self.current_balance:,.2f}")
    
    def calculate_position_size(
        self,
        confidence: float,
        volatility: float,
        current_price: Optional[float] = None
    ) -> float:
        """
        Cálculo rápido do tamanho da posição
        
        Args:
            confidence: Confiança do sinal (0-1)
            volatility: Volatilidade atual do mercado
            current_price: Preço atual (opcional)
            
        Returns:
            Valor da posição em USD
        """
        # Reset diário se necessário
        self._check_daily_reset()
        
        # Verificar limites de risco
        if not self._check_risk_limits():
            return 0.0
        
        # Kelly Criterion simplificado
        win_rate = 0.55  # Assumir 55% de win rate
        avg_win_loss_ratio = 1.5  # Assumir 1.5:1
        
        kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Limitar a 25%
        
        # Ajustar por confiança
        position_pct = kelly_fraction * confidence * self.config.max_position_pct
        
        # Ajustar por volatilidade
        volatility_factor = self._calculate_volatility_factor(volatility)
        position_pct *= volatility_factor
        
        # Calcular valor da posição
        position_value = self.current_balance * position_pct
        
        # Aplicar limites
        position_value = self._apply_position_limits(position_value)
        
        # Log detalhado
        if current_price and position_value > 0:
            quantity = position_value / current_price
            logger.debug(
                f"📊 Posição calculada: ${position_value:.2f} "
                f"({position_pct*100:.1f}%) = {quantity:.6f} unidades @ ${current_price:.2f}"
            )
        
        return position_value
    
    def _check_risk_limits(self) -> bool:
        """Verifica se pode abrir nova posição"""
        # Stop loss diário
        if self.daily_pnl < -self.max_daily_loss * self.current_balance:
            logger.warning(f"🛑 Stop loss diário atingido: ${self.daily_pnl:.2f}")
            return False
        
        # Número máximo de posições
        if self.current_positions >= self.max_positions:
            logger.debug("Máximo de posições atingido")
            return False
        
        # Margem mínima
        min_balance = 100  # USD
        if self.current_balance < min_balance:
            logger.warning(f"Balance insuficiente: ${self.current_balance:.2f}")
            return False
        
        return True
    
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """Calcula fator de ajuste baseado na volatilidade"""
        if volatility < 0.01:  # Baixa volatilidade
            return 1.2
        elif volatility < 0.02:  # Normal
            return 1.0
        elif volatility < 0.03:  # Alta
            return 0.5
        else:  # Muito alta
            return 0.3
    
    def _apply_position_limits(self, position_value: float) -> float:
        """Aplica limites ao tamanho da posição"""
        # Limite mínimo
        min_position = 50.0  # USD
        if position_value < min_position:
            return 0.0
        
        # Limite máximo absoluto
        if self.max_position_value:
            position_value = min(position_value, self.max_position_value)
        
        # Limite por percentual do balance
        max_allowed = self.current_balance * 0.1  # Máximo 10% por posição
        position_value = min(position_value, max_allowed)
        
        return position_value
    
    def should_close_position(
        self,
        current_price: float,
        entry_price: float,
        side: str = 'BUY'
    ) -> Tuple[bool, str]:
        """
        Verificação rápida se deve fechar posição
        
        Returns:
            Tupla (should_close, reason)
        """
        if self.position_info is None:
            return False, ""
        
        # Calcular P&L
        if side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SELL
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Atualizar highest PnL
        if pnl_pct > self.position_info.get('highest_pnl', 0):
            self.position_info['highest_pnl'] = pnl_pct
        
        # Stop loss
        stop_loss_pct = self.position_info.get('stop_loss_pct', 0.01)
        if pnl_pct < -stop_loss_pct:
            return True, "Stop Loss"
        
        # Take profit
        take_profit_pct = self.position_info.get('take_profit_pct', 0.015)
        if pnl_pct > take_profit_pct:
            return True, "Take Profit"
        
        # Trailing stop
        if self.position_info.get('highest_pnl', 0) > 0.005:
            trailing_pct = 0.7  # Manter 70% do lucro máximo
            if pnl_pct < self.position_info['highest_pnl'] * trailing_pct:
                return True, "Trailing Stop"
        
        # Time-based stop
        position_duration = time.time() - self.position_info.get('entry_time', time.time())
        max_duration = self.position_info.get('max_duration', 3600)  # 1 hora padrão
        
        if position_duration > max_duration and abs(pnl_pct) < 0.002:
            return True, "Time Stop"
        
        return False, ""
    
    def update_pnl(self, pnl: float, fees: float = 0):
        """Atualiza P&L com array circular"""
        self.daily_pnl += pnl
        self.current_balance += pnl
        self.total_fees_paid += fees
        
        # Atualizar histórico
        idx = self.pnl_index % 1000
        self.pnl_history[idx] = pnl
        self.pnl_index += 1
        
        # Atualizar peak e drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.drawdown_start = None
        else:
            if self.drawdown_start is None:
                self.drawdown_start = datetime.now()
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _check_daily_reset(self):
        """Verifica e reseta métricas diárias"""
        current_day = datetime.now().date()
        if current_day > self.last_reset_day:
            logger.info(f"📅 Novo dia - Reset de métricas diárias")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_day = current_day
    
    def get_risk_metrics(self) -> Dict:
        """Retorna métricas de risco atuais"""
        current_drawdown = 0
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        return {
            'current_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': current_drawdown,
            'total_fees': self.total_fees_paid,
            'daily_trades': self.daily_trades,
            'can_trade': self._check_risk_limits(),
            'risk_level': self._calculate_risk_level()
        }
    
    def _calculate_risk_level(self) -> str:
        """Calcula nível de risco atual"""
        metrics = {
            'daily_loss_pct': abs(self.daily_pnl / self.current_balance) if self.current_balance > 0 else 0,
            'drawdown': (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0,
            'balance_pct': self.current_balance / self.initial_balance
        }
        
        # Classificar risco
        if metrics['daily_loss_pct'] > 0.015 or metrics['drawdown'] > 0.1:
            return "ALTO"
        elif metrics['daily_loss_pct'] > 0.01 or metrics['drawdown'] > 0.05:
            return "MÉDIO"
        else:
            return "BAIXO"
    
    def set_position_info(self, position: Dict):
        """Define informações da posição atual"""
        self.position_info = position
        self.current_positions = 1 if position else 0
        self.daily_trades += 1
    
    def clear_position(self):
        """Limpa informações da posição"""
        self.position_info = None
        self.current_positions = 0


class MarketConditionValidator:
    """Valida se as condições de mercado são seguras para operar"""
    
    def __init__(self, config):
        self.config = config
        self.last_validation = time.time()
        self.validation_interval = 60  # Validar a cada minuto
        self.is_market_safe = True
        self.unsafe_reasons = []
        self.market_score = 100  # Score de 0-100
        
        # Histórico de condições
        self.volatility_history = []
        self.spread_history = []
        self.volume_history = []
        
        logger.info("🛡️ Market Validator inicializado")
    
    async def validate_market_conditions(
        self,
        market_data: Dict,
        client = None
    ) -> Tuple[bool, List[str]]:
        """
        Valida condições de mercado
        
        Args:
            market_data: Dados atuais do mercado
            client: Cliente da exchange (opcional)
            
        Returns:
            Tupla (is_safe, reasons)
        """
        reasons = []
        score = 100
        
        # Em modo debug, sempre retorna mercado seguro
        if self.config.debug_mode:
            return True, []
        
        # 1. Verificar volatilidade
        volatility = self._check_volatility(market_data)
        if volatility:
            vol_pct = volatility * 100
            self.volatility_history.append(volatility)
            
            if volatility > self.config.max_volatility:
                reasons.append(f"Volatilidade muito alta: {vol_pct:.2f}%")
                score -= 30
            elif volatility > self.config.max_volatility * 0.8:
                reasons.append(f"Volatilidade elevada: {vol_pct:.2f}%")
                score -= 15
        
        # 2. Verificar spread
        spread_bps = self._check_spread(market_data)
        if spread_bps:
            self.spread_history.append(spread_bps)
            
            if spread_bps > self.config.max_spread_bps:
                reasons.append(f"Spread muito alto: {spread_bps:.1f} bps")
                score -= 25
            elif spread_bps > self.config.max_spread_bps * 0.8:
                reasons.append(f"Spread elevado: {spread_bps:.1f} bps")
                score -= 10
        
        # 3. Verificar volume (async)
        if client and time.time() - self.last_validation > self.validation_interval:
            volume_ok, volume_reason = await self._check_volume_async(client)
            if not volume_ok:
                reasons.append(volume_reason)
                score -= 20
            self.last_validation = time.time()
        
        # 4. Verificar liquidez do orderbook
        liquidity_ok, liquidity_reason = self._check_liquidity(market_data)
        if not liquidity_ok:
            reasons.append(liquidity_reason)
            score -= 15
        
        # 5. Verificar horário
        time_ok, time_reason = self._check_trading_time()
        if not time_ok:
            reasons.append(time_reason)
            score -= 10
        
        # 6. Verificar condições extremas
        if self._detect_flash_crash(market_data):
            reasons.append("⚠️ FLASH CRASH DETECTADO!")
            score = 0
        
        # Atualizar estado
        self.market_score = max(0, score)
        self.is_market_safe = score >= 50  # Mínimo 50 pontos
        self.unsafe_reasons = reasons
        
        return self.is_market_safe, reasons
    
    def _check_volatility(self, market_data: Dict) -> Optional[float]:
        """Calcula volatilidade atual"""
        if 'prices' not in market_data or len(market_data['prices']) < 100:
            return None
        
        prices = market_data['prices'][-100:]
        return np.std(prices) / np.mean(prices)
    
    def _check_spread(self, market_data: Dict) -> Optional[float]:
        """Calcula spread em basis points"""
        if 'orderbook_asks' not in market_data or 'orderbook_bids' not in market_data:
            return None
        
        asks = market_data['orderbook_asks']
        bids = market_data['orderbook_bids']
        
        if len(asks) > 0 and len(bids) > 0 and asks[0, 0] > 0 and bids[0, 0] > 0:
            spread = asks[0, 0] - bids[0, 0]
            return (spread / bids[0, 0]) * 10000
        
        return None
    
    async def _check_volume_async(self, client) -> Tuple[bool, str]:
        """Verifica volume 24h (assíncrono)"""
        try:
            ticker = await client.get_ticker(symbol=self.config.symbol)
            volume_24h = float(ticker['quoteVolume'])
            
            self.volume_history.append(volume_24h)
            
            if volume_24h < self.config.min_volume_24h:
                return False, f"Volume 24h baixo: ${volume_24h:,.0f}"
            
            return True, ""
            
        except Exception as e:
            logger.debug(f"Erro ao verificar volume: {e}")
            return True, ""  # Assumir OK se erro
    
    def _check_liquidity(self, market_data: Dict) -> Tuple[bool, str]:
        """Verifica liquidez do orderbook"""
        if 'orderbook_bids' not in market_data or 'orderbook_asks' not in market_data:
            return True, ""
        
        bids = market_data['orderbook_bids']
        asks = market_data['orderbook_asks']
        
        # Verificar profundidade
        if len(bids) < 5 or len(asks) < 5:
            return False, "Orderbook raso"
        
        # Verificar volume nos primeiros níveis
        bid_volume = np.sum(bids[:5, 1])
        ask_volume = np.sum(asks[:5, 1])
        
        min_volume = 10  # Mínimo em unidades base
        if bid_volume < min_volume or ask_volume < min_volume:
            return False, "Baixa liquidez no orderbook"
        
        return True, ""
    
    def _check_trading_time(self) -> Tuple[bool, str]:
        """Verifica horário de trading"""
        current_hour = datetime.now().hour
        
        # Evitar horários de baixa liquidez (UTC)
        if 2 <= current_hour <= 6:
            return False, "Horário de baixa liquidez (2-6 UTC)"
        
        # Domingo tem liquidez reduzida
        if datetime.now().weekday() == 6:
            return False, "Domingo - liquidez reduzida"
        
        return True, ""
    
    def _detect_flash_crash(self, market_data: Dict) -> bool:
        """Detecta movimentos extremos de preço"""
        if 'prices' not in market_data or len(market_data['prices']) < 50:
            return False
        
        prices = market_data['prices']
        
        # Verificar queda/subida súbita
        recent_prices = prices[-10:]
        older_prices = prices[-50:-40]
        
        if len(recent_prices) > 0 and len(older_prices) > 0:
            recent_avg = np.mean(recent_prices)
            older_avg = np.mean(older_prices)
            
            if older_avg > 0:
                change = abs(recent_avg - older_avg) / older_avg
                
                # Movimento > 5% em poucos minutos
                if change > 0.05:
                    return True
        
        return False
    
    def get_market_health(self) -> Dict:
        """Retorna saúde geral do mercado"""
        return {
            'score': self.market_score,
            'is_safe': self.is_market_safe,
            'status': self._get_market_status(),
            'reasons': self.unsafe_reasons,
            'metrics': {
                'avg_volatility': np.mean(self.volatility_history[-10:]) if self.volatility_history else 0,
                'avg_spread': np.mean(self.spread_history[-10:]) if self.spread_history else 0,
                'last_volume': self.volume_history[-1] if self.volume_history else 0
            }
        }
    
    def _get_market_status(self) -> str:
        """Retorna status textual do mercado"""
        if self.market_score >= 80:
            return "EXCELENTE"
        elif self.market_score >= 60:
            return "BOM"
        elif self.market_score >= 50:
            return "REGULAR"
        elif self.market_score >= 30:
            return "RUIM"
        else:
            return "PERIGOSO"
