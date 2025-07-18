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

    async def send_startup_alert(self, mode: str) -> bool:
        """
        Envia um alerta informando o início do sistema
        Args:
            mode: 'PAPER' ou 'LIVE'
        """
        title = f"🚀 Sistema iniciado - Modo {mode}"
        message = f"O sistema iniciou em modo {mode}."
        # Força envio mesmo que esteja em cooldown
        return await self.send_alert(title, message, priority='info', force=True)

    async def send_shutdown_alert(self) -> bool:
        """
        Envia um alerta informando o desligamento do sistema
        """
        title = "🛑 Sistema desligado"
        message = "O sistema foi encerrado com sucesso."
        return await self.send_alert(title, message, priority='info', force=True)

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

        alert_key = f"{title}_{priority}"
        if not force and alert_key in self.last_alert_time:
            if time.time() - self.last_alert_time[alert_key] < self.alert_cooldown:
                logger.debug(f"Alerta ignorado por cooldown: {alert_key}")
                return False

        self.last_alert_time[alert_key] = time.time()
        self.alert_count[alert_key] = self.alert_count.get(alert_key, 0) + 1
        self._log_alert(title, message, priority)

        tasks = []
        if self.config.telegram_token and self.config.telegram_chat_id:
            tasks.append(self._send_telegram(title, message, priority))
        if priority == 'critical' and self.config.alert_email:
            tasks.append(self._send_email(title, message))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Erro ao enviar alerta: {result}")
        return True

    def _log_alert(self, title: str, message: str, priority: str):
        emoji = {'critical': '🚨', 'warning': '⚠️', 'info': 'ℹ️'}.get(priority, '📢')
        log_message = f"{emoji} {title}: {message}"
        if priority == 'critical':
            logger.critical(log_message)
        elif priority == 'warning':
            logger.warning(log_message)
        else:
            logger.info(log_message)

    async def _send_telegram(self, title: str, message: str, priority: str):
        emoji = {'critical': '🚨', 'warning': '⚠️', 'info': 'ℹ️'}.get(priority, '📢')
        timestamp = datetime.now().strftime('%H:%M:%S')
        text = f"{emoji} *{title}*\n\n{message}\n\n__{timestamp}__"
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

    async def _send_email(self, title: str, message: str):
        logger.info(f"Email alert (não implementado): {title}")
        # TODO: implementar envio de email

    def get_alert_stats(self) -> Dict:
        stats = {
            'total_alerts': sum(self.alert_count.values()),
            'alerts_by_type': dict(self.alert_count),
            'channels_configured': {
                'telegram': bool(self.config.telegram_token),
                'email': bool(self.config.alert_email)
            }
        }
        return stats
