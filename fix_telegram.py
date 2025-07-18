#!/usr/bin/env python3
"""
Configuração completa do Telegram e sistema de trading
"""
import os
import yaml
import requests
from pathlib import Path

print("🔧 CONFIGURAÇÃO COMPLETA DO SISTEMA")
print("=" * 60)

# 1. Criar/Atualizar .env
print("\n1️⃣ Configurando arquivo .env...")
env_content = """# Binance API (adicione suas chaves se tiver)
BINANCE_API_KEY=l9MdCeVSSFznHVYNbQ5SkPcvjIjw7vbYSjogsXHWzdpvAcwGlrjMWDLmpyBVeEAo
BINANCE_API_SECRET=0pyVGlM6OQlELbsdf1xPEkUXYge9CoZOLHgETZTNWtOlfehlOTNpYaNDmBp1pUeI

# Telegram Configuration
TELEGRAM_BOT_TOKEN=8199550294:AAEMIRLicQ167ED7Wz4cc_u2xAjHvAzpVTM
TELEGRAM_CHAT_ID=1025666426
"""

with open(".env", "w", encoding='utf-8') as f:
    f.write(env_content)
print("✅ Arquivo .env criado com credenciais do Telegram!")

# 2. Testar conexão com Telegram
print("\n2️⃣ Testando conexão com Telegram...")
token = "8199550294:AAEMIRLicQ167ED7Wz4cc_u2xAjHvAzpVTM"
chat_id = "1025666426"

url = f"https://api.telegram.org/bot{token}/sendMessage"
test_message = {
    "chat_id": chat_id,
    "text": "🎉 <b>Sistema de Trading Configurado!</b>\n\n"
            "✅ Telegram conectado com sucesso!\n"
            "📊 Você receberá alertas sobre:\n"
            "• Abertura/fechamento de posições\n"
            "• Trades grandes (baleias 🐋)\n"
            "• Análises técnicas importantes\n"
            "• Balanço atualizado\n\n"
            "💡 <i>Modo Paper Trading ativo</i>",
    "parse_mode": "HTML"
}

response = requests.post(url, data=test_message)
if response.status_code == 200:
    print("✅ Mensagem de teste enviada com sucesso!")
    print("📱 Verifique seu Telegram!")
else:
    print(f"❌ Erro ao enviar mensagem: {response.text}")

# 3. Criar configur
