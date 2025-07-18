@echo off
echo Criando fix_momentum.py...

(
echo #!/usr/bin/env python3
echo """
echo Script para corrigir o problema da feature momentum ausente
echo """
echo.
echo import os
echo import re
echo from dotenv import load_dotenv
echo.
echo # Carregar variáveis de ambiente
echo load_dotenv^(^)
echo.
echo def fix_ml_file^(^):
echo     """Remove a dependência da feature momentum do arquivo ML"""
echo     ml_file = "trade_system/analysis/ml.py"
echo     
echo     if not os.path.exists^(ml_file^):
echo         print^(f"❌ Arquivo {ml_file} não encontrado!"^)
echo         return False
echo     
echo     # Fazer backup
echo     with open^(ml_file, 'r', encoding='utf-8'^) as f:
echo         content = f.read^(^)
echo     
echo     with open^(ml_file + '.backup', 'w', encoding='utf-8'^) as f:
echo         f.write^(content^)
echo     
echo     # Remover referências ao momentum
echo     content = re.sub^(r"if 'momentum' not in features:.*?\n.*?self\.logger\.warning.*?\n", "", content, flags=re.MULTILINE^)
echo     content = re.sub^(r".*'momentum'.*\n", "", content^)
echo     
echo     # Salvar arquivo corrigido
echo     with open^(ml_file, 'w', encoding='utf-8'^) as f:
echo         f.write^(content^)
echo     
echo     print^(f"✅ Arquivo {ml_file} corrigido!"^)
echo     print^(f"📁 Backup salvo em {ml_file}.backup"^)
echo     return True
echo.
echo def update_config^(^):
echo     """Atualiza o config.yaml para permitir trades com menor confiança"""
echo     config_file = "config.yaml"
echo     
echo     if not os.path.exists^(config_file^):
echo         print^(f"❌ Arquivo {config_file} não encontrado!"^)
echo         return False
echo     
echo     with open^(config_file, 'r', encoding='utf-8'^) as f:
echo         content = f.read^(^)
echo     
echo     # Fazer backup
echo     with open^(config_file + '.backup', 'w', encoding='utf-8'^) as f:
echo         f.write^(content^)
echo     
echo     # Atualizar min_confidence
echo     content = re.sub^(r'min_confidence:\s*[\d.]+', 'min_confidence: 0.45', content^)
echo     
echo     # Salvar arquivo
echo     with open^(config_file, 'w', encoding='utf-8'^) as f:
echo         f.write^(content^)
echo     
echo     print^(f"✅ Arquivo {config_file} atualizado!"^)
echo     print^(f"   min_confidence ajustado para 0.45"^)
echo     return True
echo.
echo def add_momentum_calculation^(^):
echo     """Adiciona o cálculo do momentum ao technical.py"""
echo     tech_file = "trade_system/analysis/technical.py"
echo     
echo     if not os.path.exists^(tech_file^):
echo         print^(f"❌ Arquivo {tech_file} não encontrado!"^)
echo         return False
echo     
echo     with open^(tech_file, 'r', encoding='utf-8'^) as f:
echo         content = f.read^(^)
echo     
echo     # Verificar se já tem momentum
echo     if 'momentum' in content:
echo         print^(f"ℹ️ Momentum já existe em {tech_file}"^)
echo         return True
echo     
echo     # Fazer backup
echo     with open^(tech_file + '.backup', 'w', encoding='utf-8'^) as f:
echo         f.write^(content^)
echo     
echo     # Adicionar cálculo do momentum após o RSI
echo     momentum_code = '''
echo         # Momentum ^(taxa de mudança^)
echo         momentum_period = 10
echo         if len^(self.price_buffer^) ^>= momentum_period + 1:
echo             current_price = self.price_buffer[-1]
echo             past_price = self.price_buffer[-^(momentum_period + 1^)]
echo             features['momentum'] = ^(^(current_price - past_price^) / past_price^) * 100
echo         else:
echo             features['momentum'] = 0.0
echo '''
echo     
echo     # Encontrar onde inserir ^(após o cálculo do RSI^)
echo     rsi_pattern = r"^(features\['rsi'\] = rsi^)"
echo     replacement = r"\1" + momentum_code
echo     
echo     content = re.sub^(rsi_pattern, replacement, content^)
echo     
echo     # Salvar arquivo
echo     with open^(tech_file, 'w', encoding='utf-8'^) as f:
echo         f.write^(content^)
echo     
echo     print^(f"✅ Momentum adicionado ao {tech_file}!"^)
echo     return True
echo.
echo def main^(^):
echo     print^("🔧 Corrigindo problemas do sistema de trading...\n"^)
echo     
echo     # 1. Corrigir ML
echo     print^("1. Corrigindo arquivo ML..."^)
echo     fix_ml_file^(^)
echo     
echo     # 2. Atualizar config
echo     print^("\n2. Atualizando configuração..."^)
echo     update_config^(^)
echo     
echo     # 3. Adicionar momentum
echo     print^("\n3. Adicionando cálculo do momentum..."^)
echo     add_momentum_calculation^(^)
echo     
echo     print^("\n✅ Correções aplicadas!"^)
echo     print^("\n📝 Próximos passos:"^)
echo     print^("1. Execute novamente: trade-system paper"^)
echo     print^("2. O sistema agora deve executar trades com 45%% de confiança"^)
echo     print^("3. Os warnings de momentum devem parar de aparecer"^)
echo     
echo     # Verificar configuração do Telegram
echo     print^("\n📱 Verificando configuração do Telegram..."^)
echo     bot_token = os.getenv^('TELEGRAM_BOT_TOKEN'^)
echo     chat_id = os.getenv^('TELEGRAM_CHAT_ID'^)
echo     
echo     if bot_token and chat_id:
echo         print^(f"✅ Telegram configurado!"^)
echo         print^(f"   Bot Token: {bot_token[:20]}..."^)
echo         print^(f"   Chat ID: {chat_id}"^)
echo         print^("\n💡 Dica: Certifique-se de ter iniciado uma conversa com seu bot!"^)
echo         print^("   Acesse: https://t.me/SEU_BOT_USERNAME e envie /start"^)
echo     else:
echo         print^("❌ Telegram não configurado no .env"^)
echo.
echo if __name__ == "__main__":
echo     main^(^)
) > fix_momentum.py

echo.
echo ✅ Arquivo fix_momentum.py criado!
echo.
echo Agora execute:
echo python fix_momentum.py
pause
