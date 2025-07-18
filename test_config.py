#!/usr/bin/env python3
"""
Testa as configurações do sistema de trading
"""

import os
import sys
import requests
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Carregar variáveis de ambiente
load_dotenv()

def test_binance():
    """Testa conexão com Binance"""
    print("\n🔐 Testando Binance API...")
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Chaves da Binance não encontradas no .env")
        return False
    
    print(f"   API Key: {api_key[:20]}...")
    
    try:
        client = Client(api_key, api_secret)
        
        # Testar conexão
        status = client.get_system_status()
        print(f"✅ Status do sistema: {status}")
        
        # Verificar informações da conta
        account = client.get_account()
        print(f"✅ Conta conectada com sucesso!")
        
        # Verificar balanços
        balances = [b for b in account['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]
        if balances:
            print("\n💰 Balanços encontrados:")
            for balance in balances[:5]:  # Mostrar até 5
                total = float(balance['free']) + float(balance['locked'])
                print(f"   {balance['asset']}: {total:.8f} (livre: {balance['free']}, bloqueado: {balance['locked']})")
        
        # Verificar permissões
        print(f"\n🔑 Permissões da conta:")
        print(f"   Pode negociar: {'✅' if account['canTrade'] else '❌'}")
        print(f"   Pode sacar: {'✅' if account['canWithdraw'] else '❌'}")
        print(f"   Pode depositar: {'✅' if account['canDeposit'] else '❌'}")
        
        return True
        
    except BinanceAPIException as e:
        print(f"❌ Erro na API Binance: {e}")
        if e.code == -2014:
            print("   → Chave API inválida")
        elif e.code == -1021:
            print("   → Timestamp fora de sincronização")
        return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_telegram():
    """Testa conexão com Telegram"""
    print("\n📱 Testando Telegram Bot...")
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Configuração do Telegram não encontrada no .env")
        return False
    
    print(f"   Bot Token: {bot_token[:20]}...")
    print(f"   Chat ID: {chat_id}")
    
    try:
        # Obter informações do bot
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(url)
        data = response.json()
        
        if data['ok']:
            bot_info = data['result']
            print(f"✅ Bot conectado: @{bot_info['username']}")
            print(f"   Nome: {bot_info['first_name']}")
        else:
            print(f"❌ Erro ao conectar bot: {data}")
            return False
        
        # Tentar enviar mensagem de teste
        print("\n📤 Enviando mensagem de teste...")
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        message = "🚀 Teste de configuração do sistema de trading!\n\nSe você está vendo esta mensagem, o Telegram está configurado corretamente."
        
        response = requests.post(url, json={
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        })
        
        data = response.json()
        if data['ok']:
            print("✅ Mensagem enviada com sucesso!")
            print("\n💡 Verifique seu Telegram!")
            return True
        else:
            print(f"❌ Erro ao enviar mensagem: {data}")
            if 'description' in data:
                if 'chat not found' in data['description']:
                    print("\n⚠️ Chat não encontrado! Possíveis soluções:")
                    print("1. Inicie uma conversa com seu bot primeiro")
                    print(f"2. Acesse: https://t.me/{bot_info['username']} e envie /start")
                    print("3. Verifique se o chat_id está correto")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_market_data():
    """Testa recebimento de dados de mercado"""
    print("\n📊 Testando dados de mercado...")
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    try:
        client = Client(api_key, api_secret)
        
        # Obter preço atual
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        print(f"✅ Preço atual BTC/USDT: ${float(ticker['price']):,.2f}")
        
        # Obter informações do mercado
        info = client.get_symbol_info('BTCUSDT')
        print(f"✅ Informações do par BTCUSDT obtidas")
        print(f"   Status: {info['status']}")
        print(f"   Base Asset: {info['baseAsset']}")
        print(f"   Quote Asset: {info['quoteAsset']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def check_python_version():
    """Verifica versão do Python"""
    print(f"\n🐍 Python: {sys.version}")
    if sys.version_info < (3, 7):
        print("⚠️ Python 3.7+ recomendado")
        return False
    return True

def main():
    print("=" * 60)
    print("TESTE DE CONFIGURAÇÃO - SISTEMA DE TRADING")
    print("=" * 60)
    
    results = []
    
    # Verificar Python
    results.append(("Python", check_python_version()))
    
    # Verificar arquivo .env
    if os.path.exists('.env'):
        print("\n✅ Arquivo .env encontrado")
        results.append((".env", True))
    else:
        print("\n❌ Arquivo .env não encontrado!")
        results.append((".env", False))
    
    # Testar Binance
    results.append(("Binance API", test_binance()))
    
    # Testar dados de mercado
    results.append(("Market Data", test_market_data()))
    
    # Testar Telegram
    results.append(("Telegram Bot", test_telegram()))
    
    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ OK" if passed else "❌ FALHOU"
        print(f"{name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 Todas as configurações estão corretas!")
        print("\nPróximos passos:")
        print("1. Execute: python fix_momentum.py")
        print("2. Execute: trade-system paper")
    else:
        print("⚠️ Algumas configurações precisam de atenção")
        print("\nVerifique os erros acima e tente novamente")

if __name__ == "__main__":
    main()
