"""Sistema de alertas multi-canal"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

class AlertManager:
    """Gerencia envio de alertas por múltiplos canais"""
    
    def __init__(self, config: Any):
        self.config = config.alerts if hasattr(config, 'alerts') else config.get('alerts', {})
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.alert_history = []
        self.max_alerts_per_hour = 20
        
        # Canais
        self.telegram_enabled = self.config.get('telegram_enabled', False)
        self.email_enabled = self.config.get('email_enabled', False)
        
        self.logger.info("📢 Sistema de alertas inicializado")
    
    async def send_alert(self, message: str, alert_type: str = "info"):
        """
        Envia alerta por todos os canais habilitados
        
        Args:
            message: Mensagem do alerta
            alert_type: Tipo do alerta (info, warning, error, trade, report)
        """
        # Verificar rate limit
        if not self._check_rate_limit():
            self.logger.warning("Rate limit de alertas atingido")
            return
        
        # Formatar mensagem
        formatted_message = self._format_message(message, alert_type)
        
        # Enviar por cada canal
        tasks = []
        
        if self.telegram_enabled:
            tasks.append(self._send_telegram(formatted_message, alert_type))
        
        if self.email_enabled:
            tasks.append(self._send_email(formatted_message, alert_type))
        
        # Executar envios em paralelo
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Registrar no histórico
        self.alert_history.append({
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message
        })
    
    async def _send_telegram(self, message: str, alert_type: str):
        """Envia alerta via Telegram"""
        token = self.config.get('telegram_token', '')
        chat_id = self.config.get('telegram_chat_id', '')
        
        if not token or not chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            
            # Adicionar emoji baseado no tipo
            emojis = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '❌',
                'trade': '💰',
                'report': '📊'
            }
            
            emoji = emojis.get(alert_type, '📌')
            
            data = {
                'chat_id': chat_id,
                'text': f"{emoji} {message}",
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        self.logger.error(f"Erro ao enviar Telegram: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Erro ao enviar alerta Telegram: {e}")
    
    async def _send_email(self, message: str, alert_type: str):
        """Envia alerta via email"""
        smtp_server = self.config.get('email_smtp', '')
        from_email = self.config.get('email_from', '')
        to_email = self.config.get('email_to', '')
        
        if not all([smtp_server, from_email, to_email]):
            return
        
        try:
            # Criar mensagem
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = f"Trading Bot Alert: {alert_type.upper()}"
            
            # Corpo do email
            body = f"""
            <html>
                <body>
                    <h2>Trading Bot Alert</h2>
                    <p><strong>Type:</strong> {alert_type}</p>
                    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <hr>
                    <pre>{message}</pre>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Enviar email (simplificado - em produção usar async)
            # with smtplib.SMTP(smtp_server) as server:
            #     server.send_message(msg)
            
        except Exception as e:
            self.logger.error(f"Erro ao enviar email: {e}")
    
    def _format_message(self, message: str, alert_type: str) -> str:
        """Formata mensagem para envio"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Remover espaços extras e formatar
        lines = [line.strip() for line in message.strip().split('\n')]
        formatted = '\n'.join(lines)
        
        return f"[{timestamp}] {formatted}"
    
    def _check_rate_limit(self) -> bool:
        """Verifica se pode enviar alerta (rate limiting)"""
        # Limpar alertas antigos
        one_hour_ago = datetime.now() - timedelta(hours=1)
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert['timestamp'] > one_hour_ago
        ]
        
        # Verificar limite
        return len(self.alert_history) < self.max_alerts_per_hour
    
    def get_alert_stats(self) -> Dict[str, int]:
        """Retorna estatísticas de alertas"""
        stats = {}
        
        for alert in self.alert_history:
            alert_type = alert['type']
            stats[alert_type] = stats.get(alert_type, 0) + 1
        
        return stats
