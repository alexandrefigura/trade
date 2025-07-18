"""Configuração de logging centralizada"""
import logging
import sys
from datetime import datetime
from pathlib import Path
import colorlog

def setup_logging(level: str = "INFO", log_dir: str = "logs"):
    """Configura sistema de logging com cores e arquivo"""
    # Criar diretório de logs
    Path(log_dir).mkdir(exist_ok=True)
    
    # Nome do arquivo de log
    log_file = Path(log_dir) / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Formato para arquivo
    file_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Formato colorido para console
    console_formatter = colorlog.ColoredFormatter(
        '%(asctime)s.%(msecs)03d %(log_color)s[%(levelname)s]%(reset)s %(blue)s%(name)s%(reset)s: %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Configurar logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suprimir logs excessivos de bibliotecas
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Sistema de logging configurado - Nível: {level}")
    logger.info(f"Logs salvos em: {log_file}")
    
    return logger
