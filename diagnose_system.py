#!/usr/bin/env python3
"""
Diagnóstico do Sistema de Trading
"""
import os
import importlib.util
import sys

def check_module(module_path, class_name):
    """Verifica se um módulo e classe existem"""
    try:
        if not os.path.exists(module_path):
            return f"❌ Arquivo não encontrado: {module_path}"
            
        # Carregar módulo
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Verificar classe
        if hasattr(module, class_name):
            return f"✅ {class_name} encontrado em {module_path}"
        else:
            # Listar classes disponíveis
            classes = [name for name in dir(module) if name[0].isupper()]
            return f"❌ {class_name} não encontrado. Classes disponíveis: {', '.join(classes)}"
            
    except Exception as e:
        return f"❌ Erro ao carregar {module_path}: {str(e)}"

print("🔍 DIAGNÓSTICO DO SISTEMA")
print("=" * 60)

# Verificar estrutura de diretórios
print("\n📁 Estrutura de diretórios:")
for root, dirs, files in os.walk('trade_system'):
    level = root.replace('trade_system', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        if file.endswith('.py'):
            print(f'{subindent}{file}')

# Verificar módulos críticos
print("\n🔧 Verificando módulos críticos:")
checks = [
    ('trade_system/analysis/technical.py', 'TechnicalAnalyzer'),
    ('trade_system/analysis/ml.py', 'SimplifiedMLPredictor'),
    ('trade_system/analysis/orderbook.py', 'OrderbookAnalyzer'),
    ('trade_system/websocket_manager.py', 'WebSocketManager'),
    ('trade_system/risk.py', 'RiskManager'),
]

for module_path, class_name in checks:
    print(check_module(module_path, class_name))

print("\n✅ Diagnóstico completo!")
print("\n🚀 Para executar o sistema funcional:")
print("   python working_paper_trading.py")
