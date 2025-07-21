#!/usr/bin/env python3
"""
Script para corrigir o erro do MLPredictor config
"""
import os
import re

def fix_ml_predictor_issue():
    print("🔧 CORRIGINDO MLPredictor CONFIG")
    print("=" * 60)
    
    # 1. Corrigir main.py
    main_path = 'trade_system/main.py'
    if os.path.exists(main_path):
        print(f"📝 Corrigindo {main_path}...")
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Backup
        with open(f"{main_path}.backup2", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Corrigir a linha que cria o SimplifiedMLPredictor
        content = re.sub(
            r'self\.ml_predictor\s*=\s*SimplifiedMLPredictor\(\)',
            'self.ml_predictor = SimplifiedMLPredictor(self.config)',
            content
        )
        
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ main.py corrigido!")
    
    # 2. Verificar a assinatura do SimplifiedMLPredictor
    ml_path = 'trade_system/analysis/ml.py'
    if os.path.exists(ml_path):
        print(f"\n📝 Verificando {ml_path}...")
        with open(ml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Procurar pela definição da classe
        class_match = re.search(r'class SimplifiedMLPredictor.*?:\s*\n\s*def __init__\(self([^)]*)\)', content, re.DOTALL)
        if class_match:
            print(f"   Assinatura encontrada: def __init__(self{class_match.group(1)})")
    
    # 3. Criar uma versão corrigida do paper_trading_final.py
    print("\n📝 Criando versão corrigida do paper trading...")
    
    fixed_paper_trading = '''#!/usr/bin/env python3
"""
Paper Trading Final - Versão Corrigida
"""
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trade_system.config import TradingConfig
from trade_system.main import TradingSystem
from trade_system.logging_config import setup_logging

# Carregar variáveis de ambiente
load_dotenv()

async def main():
    """Função principal com tratamento de erros aprimorado"""
    try:
        # Verificar API keys
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            print("❌ Erro: BINANCE_API_KEY ou BINANCE_API_SECRET não encontradas no arquivo .env")
            print("💡 Crie um arquivo .env com:")
            print('   BINANCE_API_KEY="sua_api_key"')
            print('   BINANCE_API_SECRET="sua_api_secret"')
            return
            
        print(f"✅ API Key: {api_key[:8]}...")
        
        # Configurar logging
        print("\\n🔄 Carregando sistema...")
        setup_logging()
        
        # Carregar configuração
        config = TradingConfig()
        
        # Banner
        print(f"""
============================================================
🤖 ULTRA TRADING BOT - PAPER TRADING (CORRIGIDO)
============================================================
📊 Par: {config.SYMBOL}
💰 Balance: ${config.INITIAL_BALANCE:,.2f}
⏰ Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================
        """)
        
        # Criar sistema com paper_trading=True
        system = TradingSystem(config, paper_trading=True)
        
        # Executar
        print("\\n🚀 Iniciando Paper Trading...\\n")
        await system.run()
        
    except KeyboardInterrupt:
        print("\\n\\n⏹️ Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"\\n❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\\n💡 Use o sistema simplificado:")
        print("   python working_paper_trading.py")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open('paper_trading_fixed.py', 'w', encoding='utf-8') as f:
        f.write(fixed_paper_trading)
    print("✅ paper_trading_fixed.py criado!")
    
    # 4. Criar um patch manual se necessário
    print("\n📝 Criando patch manual...")
    
    manual_patch = '''#!/usr/bin/env python3
"""
Patch manual para corrigir o sistema principal
"""
import os
import sys

def apply_patch():
    print("🔧 APLICANDO PATCH MANUAL")
    print("=" * 60)
    
    # Caminho do main.py
    main_path = 'trade_system/main.py'
    
    if not os.path.exists(main_path):
        print("❌ Arquivo trade_system/main.py não encontrado!")
        return
    
    # Ler o arquivo
    with open(main_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Procurar e corrigir a linha problemática
    modified = False
    for i, line in enumerate(lines):
        if 'SimplifiedMLPredictor()' in line and 'self.ml_predictor' in line:
            # Corrigir para passar o config
            lines[i] = line.replace('SimplifiedMLPredictor()', 'SimplifiedMLPredictor(self.config)')
            print(f"✅ Linha {i+1} corrigida: {lines[i].strip()}")
            modified = True
            break
    
    if modified:
        # Salvar as alterações
        with open(main_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("✅ Patch aplicado com sucesso!")
    else:
        print("⚠️ Linha problemática não encontrada - pode já estar corrigida")
    
    print("\\n🚀 Tente executar novamente:")
    print("   python paper_trading_fixed.py")

if __name__ == "__main__":
    apply_patch()
'''
    
    with open('apply_patch.py', 'w', encoding='utf-8') as f:
        f.write(manual_patch)
    print("✅ apply_patch.py criado!")
    
    print("\n" + "=" * 60)
    print("✅ SCRIPTS DE CORREÇÃO CRIADOS!")
    print("\n🚀 Execute na seguinte ordem:")
    print("   1. python apply_patch.py")
    print("   2. python paper_trading_fixed.py")
    print("\n💡 Se ainda houver erros, use:")
    print("   python working_paper_trading.py (que já está funcionando)")

if __name__ == "__main__":
    fix_ml_predictor_issue()
