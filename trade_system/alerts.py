"""
Sistema de alertas multi-canal com cooldown
"""
import time
import asyncio
import aiohttp
from typing import Dict, Optional
from datetime import datetime
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class AlertSystem:
    """Sistema de alertas multi-canal"""
    
    def __init__(self, config):
        self.config = config
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutos entre alertas do mesmo tipo
        self.alert_count = {}
        
    async def send_alert(
        self, 
        title: str, 
        message: str, 
        priority: str = 'info',
        force: bool = False
    ) -> bool:
        """
        Envia alerta por múltiplos canais
        
        Args:
            title: Título do alerta
            message: Mensagem detalhada
            priority: 'info', 'warning', 'critical'
            force: Ignora cooldown se True
            
        Returns:
            True se alerta foi enviado
        """
        if not self.config.enable_alerts:
            return False
        
        # Verificar cooldown
        alert_key = f"{title}_{priority}"
        
        if not force and alert_key in self.last_alert_time:
            if time.time() - self.last_alert_time[alert_key] < self.alert_cooldown:
                logger.debug(f"Alerta ignorado por cooldown: {alert_key}")
                return False
        
        self.last_alert_time[alert_key] = time.time()
        self.alert_count[alert_key] = self.alert_count.get(alert_key, 0) + 1
        
        # Log sempre
        self._log_alert(title, message, priority)
        
        # Enviar por canais configurados
        tasks = []
        
        if self.config.telegram_token and self.config.telegram_chat_id:
            tasks.append(self._send_telegram(title, message, priority))
        
        if priority == 'critical' and self.config.alert_email:
            tasks.append(self._send_email(title, message))
        
        # Executar envios em paralelo
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log erros
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Erro ao enviar alerta: {result}")
        
        return True
    
    def _log_alert(self, title: str, message: str, priority: str):
        """Log local do alerta"""
        emoji = {
            'critical': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️'
        }.get(priority, '📢')
        
        log_message = f"{emoji} {title}: {message}"
        
        if priority == 'critical':
            logger.critical(log_message)
        elif priority == 'warning':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    async def _send_telegram(self, title: str, message: str, priority: str):
        """Envia alerta via Telegram"""
        try:
            emoji = {
                'critical': '🚨',
                'warning': '⚠️',
                'info': 'ℹ️'
            }.get(priority, '📢')
            
            # Formatar mensagem
            timestamp = datetime.now().strftime('%H:%M:%S')
            text = f"{emoji} *{title}*\n\n{message}\n\n__{timestamp}__"
            
            # Limitar tamanho da mensagem
            if len(text) > 4096:
                text = text[:4093] + "..."
            
            url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
            data = {
                'chat_id': self.config.telegram_chat_id,
                'text': text,
                'parse_mode': 'Markdown',
                'disable_notification': priority == 'info'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Erro Telegram API: {response.status} - {error_text}")
                    else:
                        logger.debug(f"Alerta Telegram enviado: {title}")
                
        except Exception as e:
            logger.error(f"Erro ao enviar alerta Telegram: {e}")
    
    async def _send_email(self, title: str, message: str):
        """
        Envia alerta via email (implementar conforme necessário)
        Placeholder para integração com serviço de email
        """
        logger.info(f"Email alert (não implementado): {title}")
        # TODO: Implementar envio de email via SMTP ou API (SendGrid, etc)
    
    def get_alert_stats(self) -> Dict:
        """Retorna estatísticas de alertas"""
        stats = {
            'total_alerts': sum(self.alert_count.values()),
            'alerts_by_type': dict(self.alert_count),
            'channels_configured': {
                'telegram': bool(self.config.telegram_token),
                'email': bool(self.config.alert_email)
            }
        }
        
        # Alertas recentes
        now = time.time()
        recent_alerts = []
        for alert_key, last_time in self.last_alert_time.items():
            if now - last_time < 3600:  # Última hora
                recent_alerts.append({
                    'key': alert_key,
                    'time_ago': int(now - last_time),
                    'count': self.alert_count.get(alert_key, 0)
                })
        
        stats['recent_alerts'] = recent_alerts
        return stats
