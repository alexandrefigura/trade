import os

print("🔧 CORRIGINDO CARREGAMENTO DO .env")
print("=" * 60)

# 1. Verificar se .env existe
if os.path.exists('.env'):
    print("✅ Arquivo .env encontrado!")
    
    # Mostrar conteúdo (sem valores sensíveis)
    print("\n📋 Variáveis no .env:")
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key = line.split('=')[0].strip()
                print(f"  - {key}")
else:
    print("❌ Arquivo .env não encontrado!")

# 2. Verificar config.py
config_file = 'trade_system/config.py'
if os.path.exists(config_file):
    print(f"\n📝 Corrigindo {config_file}...")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar se tem import do dotenv
    if 'from dotenv import load_dotenv' not in content:
        # Adicionar no início do arquivo
        lines = content.split('\n')
        
        # Encontrar onde inserir (após os imports)
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
            elif insert_pos > 0 and not line.startswith(('import ', 'from ')):
                break
        
        # Inserir imports do dotenv
        lines.insert(insert_pos, '')
        lines.insert(insert_pos + 1, '# Carregar variáveis de ambiente')
        lines.insert(insert_pos + 2, 'from dotenv import load_dotenv')
        lines.insert(insert_pos + 3, 'load_dotenv()')
        lines.insert(insert_pos + 4, '')
        
        content = '\n'.join(lines)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Import do dotenv adicionado!")
    
    # Verificar método from_env
    if 'def from_env' in content:
        print("✅ Método from_env encontrado")
    else:
        print("⚠️ Método from_env não encontrado - criando...")
        
        # Adicionar método from_env se não existir
        if 'class TradingConfig' in content:
            # Adicionar método na classe
            method_code = '''
    @classmethod
    def from_env(cls):
        """Carrega configuração das variáveis de ambiente"""
        # Garantir que .env foi carregado
        from dotenv import load_dotenv
        load_dotenv()
        
        config = cls()
        config.api_key = os.getenv('BINANCE_API_KEY', '')
        config.api_secret = os.getenv('BINANCE_API_SECRET', '')
        
        # Alertas opcionais
        if hasattr(config, 'alerts'):
            config.alerts['telegram_token'] = os.getenv('TELEGRAM_BOT_TOKEN', '')
            config.alerts['telegram_chat_id'] = os.getenv('TELEGRAM_CHAT_ID', '')
        
        return config
'''
            
            # Inserir antes do final da classe
            content = content.replace('\n\n# Fim', method_code + '\n\n# Fim')
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)

print("\n✅ Correções aplicadas!")

# 3. Criar wrapper que garante o carregamento do .env
print("\n📝 Criando wrapper seguro...")

wrapper = '''"""Wrapper que garante carregamento do .env"""
import os
import sys
import asyncio

# Carregar .env PRIMEIRO
from dotenv import load_dotenv
load_dotenv()

# Verificar se carregou
api_key = os.getenv('BINANCE_API_KEY')
if not api_key:
    # Tentar carregar manualmente
    if os.path.exists('.env'):
        print("📄 Carregando .env manualmente...")
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip('"').strip("'")
        
        api_key = os.getenv('BINANCE_API_KEY')

if not api_key:
    print("❌ BINANCE_API_KEY não encontrada!")
    print("\\nVerifique o arquivo .env")
    sys.exit(1)

print(f"✅ API Key carregada: {api_key[:8]}...")

# Agora importar e executar
from trade_system.config import TradingConfig
from trade_system.main import run_paper_trading

if __name__ == "__main__":
    asyncio.run(run_paper_trading())
'''

with open('run_with_env.py', 'w', encoding='utf-8') as f:
    f.write(wrapper)

print("✅ Wrapper criado: run_with_env.py")

# 4. Testar carregamento
print("\n🧪 Testando carregamento do .env...")

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('BINANCE_API_KEY')
if api_key:
    print(f"✅ Teste OK! API Key: {api_key[:8]}...")
else:
    print("❌ Teste falhou!")
    
    # Carregar manualmente
    if os.path.exists('.env'):
        print("\n📄 Tentando carregar manualmente...")
        with open('.env', 'r') as f:
            for line in f:
                if 'BINANCE_API_KEY=' in line:
                    value = line.split('=', 1)[1].strip().strip('"').strip("'")
                    if value:
                        print(f"✅ Encontrado no .env: {value[:8]}...")
                    break

print("\n🚀 Execute: python run_with_env.py")
