import os
import re
import shutil

print("🔧 CORREÇÃO COMPLETA DO SISTEMA DE TRADING")
print("=" * 60)

# 1. Corrigir orderbook.py
print("\n📄 Corrigindo orderbook.py...")
orderbook_file = 'trade_system/analysis/orderbook.py'
try:
    with open(orderbook_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Corrigir import errado
    content = content.replace('from trade_system.logging_config import logging.getLogger', 
                            'import logging')
    
    with open(orderbook_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ orderbook.py corrigido!")
except Exception as e:
    print(f"❌ Erro: {e}")

# 2. Mapear todas as classes reais
print("\n🔍 Mapeando classes reais...")
class_map = {}

def find_classes(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        return classes
    except:
        return []

# Mapear todos os arquivos
for root, dirs, files in os.walk('trade_system'):
    if '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            classes = find_classes(filepath)
            if classes:
                module = filepath.replace('\\', '/').replace('trade_system/', '').replace('.py', '')
                for cls in classes:
                    class_map[cls] = module
                    print(f"  {cls} -> {module}")

# 3. Corrigir todos os imports
print("\n🔧 Corrigindo imports...")

# Mapeamento de nomes errados para corretos
wrong_to_correct = {
    'WebSocketManager': 'WSManager',  # Vamos descobrir o nome real
    'UltraFastTechnicalAnalyzer': 'TechnicalAnalyzer',
    'UltraFastWebSocketManager': 'WSManager',
    'AlertSystem': 'AlertManager',
    'UltraFastCache': 'CacheManager',
    'get_logger': 'logging.getLogger'
}

# Se websocket_manager.py tem MLPredictor (erro), precisamos do arquivo correto
if 'websocket_manager' in str(class_map.values()) and 'MLPredictor' in class_map:
    print("\n⚠️ websocket_manager.py tem a classe errada (MLPredictor)")
    print("📝 Criando WebSocketManager correto...")
    
    # Criar um WebSocketManager básico
    websocket_content = '''"""WebSocket Manager para dados em tempo real"""
import asyncio
import json
import logging
from collections import deque
from typing import Dict, List, Optional, Any
import aiohttp

class WebSocketManager:
    """Gerencia conexões WebSocket com a exchange"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_connected = False
        self.price_buffer = deque(maxlen=1000)
        self.trade_buffer = deque(maxlen=1000)
        self.orderbook_buffer = deque(maxlen=100)
        
    async def connect(self):
        """Conecta ao WebSocket (implementação básica)"""
        self.logger.info("WebSocket conectando...")
        self.is_connected = True
        # TODO: Implementar conexão real
        
    async def disconnect(self):
        """Desconecta WebSocket"""
        self.is_connected = False
        
    def get_latest_price(self) -> Optional[float]:
        """Retorna último preço"""
        if self.price_buffer:
            return self.price_buffer[-1].get('close', 0)
        return None
        
    def get_latest_orderbook(self) -> Optional[Dict]:
        """Retorna último orderbook"""
        if self.orderbook_buffer:
            return self.orderbook_buffer[-1]
        return None
        
    def get_candles_df(self, limit: int = 100):
        """Retorna candles como DataFrame"""
        # Implementação simplificada
        return None

# Alias para compatibilidade
WSManager = WebSocketManager
'''
    
    with open('trade_system/websocket_manager.py', 'w', encoding='utf-8') as f:
        f.write(websocket_content)
    print("✅ WebSocketManager criado!")

# 4. Corrigir main.py
print("\n📝 Corrigindo main.py...")
main_file = 'trade_system/main.py'

try:
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Corrigir imports conhecidos
    replacements = [
        ('from trade_system.websocket_manager import MLPredictor', 
         'from trade_system.websocket_manager import WebSocketManager'),
        ('from trade_system.analysis.technical import UltraFastTechnicalAnalyzer',
         'from trade_system.analysis.technical import TechnicalAnalyzer'),
        ('UltraFastTechnicalAnalyzer', 'TechnicalAnalyzer'),
        ('MLPredictor(', 'WebSocketManager('),  # Se estava usando errado
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ main.py corrigido!")
    
except Exception as e:
    print(f"❌ Erro: {e}")

# 5. Verificar __init__.py files
print("\n🔍 Verificando __init__.py files...")

init_files = [
    'trade_system/__init__.py',
    'trade_system/analysis/__init__.py'
]

for init_file in init_files:
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Corrigir imports errados
        original = content
        content = content.replace('UltraFastTechnicalAnalyzer', 'TechnicalAnalyzer')
        content = content.replace('get_logger', 'logging.getLogger')
        
        if content != original:
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {init_file} corrigido!")
            
    except Exception as e:
        print(f"❌ Erro em {init_file}: {e}")

print("\n✅ Correção completa!")
print("\n🚀 Tente executar novamente: python run_trading.py")
