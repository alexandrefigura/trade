#!/usr/bin/env python3
"""
Obtém o Chat ID correto do Telegram
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_updates():
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        print("❌ Token do bot não encontrado no .env")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    
    print("🔍 Buscando atualizações do bot...")
    print(f"Token: {bot_token[:20]}...")
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if not data['ok']:
            print(f"❌ Erro: {data}")
            return
        
        if not data['result']:
            print("❌ Nenhuma mensagem encontrada!")
            print("\n📱 Por favor:")
            print("1. Acesse https://t.me/JacksonTrade_bot")
            print("2. Envie /start")
            print("3. Execute este script novamente")
            return
        
        print(f"\n✅ {len(data['result'])} mensagens encontradas!")
        
        # Pegar o chat_id mais recente
        chat_ids = set()
        for update in data['result']:
            if 'message' in update:
                chat = update['message']['chat']
                chat_id = chat['id']
                chat_ids.add(chat_id)
                
                print(f"\n📨 Mensagem de: {chat.get('first_name', 'Unknown')}")
                print(f"   Chat ID: {chat_id}")
                print(f"   Tipo: {chat['type']}")
                if 'username' in chat:
                    print(f"   Username: @{chat['username']}")
                if 'text' in update['message']:
                    print(f"   Texto: {update['message']['text']}")
        
        if chat_ids:
            print("\n✅ Chat IDs encontrados:", list(chat_ids))
            print(f"\n📝 Atualize seu .env com o chat_id correto:")
            print(f"TELEGRAM_CHAT_ID={list(chat_ids)[0]}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    print("="*50)
    print("OBTER CHAT ID DO TELEGRAM")
    print("="*50)
    get_updates()
