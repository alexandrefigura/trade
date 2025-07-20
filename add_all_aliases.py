import os
import re

print("🔧 ADICIONANDO TODOS OS ALIASES NECESSÁRIOS")
print("=" * 60)

# Mapeamento completo de aliases necessários
aliases_map = {
    'trade_system/analysis/ml.py': {
        'SimplifiedMLPredictor': 'MLPredictor'
    },
    'trade_system/analysis/orderbook.py': {
        'ParallelOrderbookAnalyzer': 'OrderbookAnalyzer'
    },
    'trade_system/analysis/technical.py': {
        'UltraFastTechnicalAnalyzer': 'TechnicalAnalyzer'
    },
    'trade_system/signals.py': {
        'OptimizedSignalConsolidator': 'SignalAggregator'
    },
    'trade_system/websocket_manager.py': {
        'UltraFastWebSocketManager': 'WebSocketManager'
    },
    'trade_system/cache.py': {
        'UltraFastCache': 'CacheManager'
    },
    'trade_system/alerts.py': {
        'AlertSystem': 'AlertManager'
    }
}

# Adicionar aliases em cada arquivo
for filepath, aliases in aliases_map.items():
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verificar quais classes existem
            existing_classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            
            # Adicionar aliases necessários
            aliases_added = []
            for alias, original in aliases.items():
                if original in existing_classes and alias not in content:
                    # Adicionar alias no final do arquivo
                    if not content.endswith('\n'):
                        content += '\n'
                    content += f'\n# Alias para compatibilidade\n{alias} = {original}\n'
                    aliases_added.append(alias)
            
            # Salvar se adicionou aliases
            if aliases_added:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ {filepath}: Adicionados {', '.join(aliases_added)}")
            else:
                print(f"📌 {filepath}: Nenhum alias necessário")
                
        except Exception as e:
            print(f"❌ Erro em {filepath}: {e}")
    else:
        print(f"⚠️ Arquivo não encontrado: {filepath}")

print("\n✅ Aliases adicionados!")
print("\n🚀 Tente executar novamente: python run_trading.py")
