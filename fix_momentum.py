#!/usr/bin/env python3
"""
Script para corrigir o problema da feature momentum ausente
"""

import os
import re
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

def fix_ml_file():
    """Remove a dependência da feature momentum do arquivo ML"""
    ml_file = "trade_system/analysis/ml.py"
    
    if not os.path.exists(ml_file):
        print(f"❌ Arquivo {ml_file} não encontrado!")
        return False
    
    # Fazer backup
    with open(ml_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(ml_file + '.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Remover referências ao momentum
    # Procurar por linhas que verificam o momentum
    content = re.sub(r"if 'momentum' not in features:.*?\n.*?self\.logger\.warning.*?\n", "", content, flags=re.MULTILINE)
    content = re.sub(r".*'momentum'.*\n", "", content)
    
    # Salvar arquivo corrigido
    with open(ml_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Arquivo {ml_file} corrigido!")
    print(f"📁 Backup salvo em {ml_file}.backup")
    return True

def update_config():
    """Atualiza o config.yaml para permitir trades com menor confiança"""
    config_file = "config.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ Arquivo {config_file} não encontrado!")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fazer backup
    with open(config_file + '.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Atualizar min_confidence
    content = re.sub(r'min_confidence:\s*[\d.]+', 'min_confidence: 0.45', content)
    
    # Salvar arquivo
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Arquivo {config_file} atualizado!")
    print(f"   min_confidence ajustado para 0.45")
    return True

def add_momentum_calculation():
    """Adiciona o cálculo do momentum ao technical.py"""
    tech_file = "trade_system/analysis/technical.py"
    
    if not os.path.exists(tech_file):
        print(f"❌ Arquivo {tech_file} não encontrado!")
        return False
    
    with open(tech_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar se já tem momentum
    if 'momentum' in content:
        print(f"ℹ️ Momentum já existe em {tech_file}")
        return True
    
    # Fazer backup
    with open(tech_file + '.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Adicionar cálculo do momentum após o RSI
    momentum_code = '''
        # Momentum (taxa de mudança)
        momentum_period = 10
        if len(self.price_buffer) >= momentum_period + 1:
            current_price = self.price_buffer[-1]
            past_price = self.price_buffer[-(momentum_period + 1)]
            features['momentum'] = ((current_price - past_price) / past_price) * 100
        else:
            features['momentum'] = 0.0
'''
    
    # Encontrar onde inserir (após o cálculo do RSI)
    rsi_pattern = r"(features\['rsi'\] = rsi)"
    replacement = r"\1" + momentum_code
    
    content = re.sub(rsi_pattern, replacement, content)
    
    # Salvar arquivo
    with open(tech_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Momentum adicionado ao {tech_file}!")
    return True

def main():
    print("🔧 Corrigindo problemas do sistema de trading...\n")
    
    # 1. Corrigir ML
    print("1. Corrigindo arquivo ML...")
    fix_ml_file()
    
    # 2. Atualizar config
    print("\n2. Atualizando configuração...")
    update_config()
    
    # 3. Adicionar momentum
    print("\n3. Adicionando cálculo do momentum...")
    add_momentum_calculation()
    
    print("\n✅ Correções aplicadas!")
    print("\n📝 Próximos passos:")
    print("1. Execute novamente: trade-system paper")
    print("2. O sistema agora deve executar trades com 45% de confiança")
    print("3. Os warnings de momentum devem parar de aparecer")
    
    # Verificar configuração do Telegram
    print("\n📱 Verificando configuração do Telegram...")
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        print(f"✅ Telegram configurado!")
        print(f"   Bot Token: {bot_token[:20]}...")
        print(f"   Chat ID: {chat_id}")
        print("\n💡 Dica: Certifique-se de ter iniciado uma conversa com seu bot!")
        print("   Acesse: https://t.me/SEU_BOT_USERNAME e envie /start")
    else:
        print("❌ Telegram não configurado no .env")

if __name__ == "__main__":
    main()
