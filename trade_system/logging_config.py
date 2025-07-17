"""
Configuração centralizada de logging
"""
import os
import sys
import io
import logging
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_dir: str = "logs"
) -> None:
    """
    Configura o sistema de logging de forma centralizada
    
    Args:
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Nome do arquivo de log (padrão: ultra_v5.log)
        log_dir: Diretório para logs
    """
    # Força saída e erro em UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Criar diretório de logs se não existir
    os.makedirs(log_dir, exist_ok=True)
    
    # Nome do arquivo de log
    if log_file is None:
        log_file = f"trading_{datetime.now().strftime('%Y%m%d')}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Handlers
    handlers = []
    
    # File handler com UTF-8
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    handlers.append(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    handlers.append(console_handler)
    
    # Formato detalhado
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    for handler in handlers:
        handler.setFormatter(formatter)
    
    # Configurar root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )
    
    # Configurar loggers específicos
    # Reduzir ruído de bibliotecas externas
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websocket').setLevel(logging.WARNING)
    logging.getLogger('binance').setLevel(logging.WARNING)
    
    # Log inicial
    logger = logging.getLogger(__name__)
    logger.info(f"Sistema de logging configurado - Nível: {log_level}")
    logger.info(f"Logs salvos em: {log_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger configurado para o módulo
    
    Args:
        name: Nome do módulo (geralmente __name__)
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)
