#!/usr/bin/env python3
"""
Script para corrigir o erro 'required' is not defined no ML
"""

import os
import re

def fix_ml_required_error():
    """Corrige o erro da variável 'required' não definida"""
    ml_file = "trade_system/analysis/ml.py"
    
    if not os.path.exists(ml_file):
        print(f"❌ Arquivo {ml_file} não encontrado!")
        return False
    
    # Fazer backup
    with open(ml_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Salvar backup com timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{ml_file}.backup_{timestamp}"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📁 Backup salvo em {backup_file}")
    
    # Procurar por "for feat in required:" e substituir
    if "for feat in required:" in content:
        # Definir a lista de features requeridas
        required_features = """
        # Features requeridas para o modelo
        required_features = ['rsi', 'ma_ratio', 'volume_ratio', 'price_change', 'volatility']
        
        # Verificar features ausentes
        missing = []
        for feat in required_features:"""
        
        # Substituir a linha problemática
        content = re.sub(
            r'for feat in required:',
            required_features,
            content
        )
        
        # Também precisamos ajustar a referência para missing
        content = re.sub(
            r'missing\.append\(feat\)',
            '    missing.append(feat)',
            content
        )
    
    # Alternativa: se o código estiver diferente, procurar por padrões mais amplos
    if "name 'required' is not defined" in content or "required" in content:
        # Adicionar definição de required_features no início da função predict
        predict_pattern = r'(def predict\(self, features: Dict\[str, float\]\).*?:\s*\n)'
        replacement = r'\1        required_features = ["rsi", "ma_ratio", "volume_ratio", "price_change", "volatility"]\n'
        content = re.sub(predict_pattern, replacement, content, flags=re.DOTALL)
    
    # Salvar arquivo corrigido
    with open(ml_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Arquivo {ml_file} corrigido!")
    return True

def verify_fix():
    """Verifica se a correção foi aplicada"""
    ml_file = "trade_system/analysis/ml.py"
    
    with open(ml_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "required_features" in content:
        print("✅ Variável required_features encontrada no arquivo!")
        return True
    else:
        print("⚠️ Correção pode não ter sido aplicada completamente")
        return False

def main():
    print("🔧 Corrigindo erro 'required' is not defined...\n")
    
    # Aplicar correção
    if fix_ml_required_error():
        print("\n🔍 Verificando correção...")
        verify_fix()
        
        print("\n✅ Correção aplicada!")
        print("\n📝 Próximos passos:")
        print("1. Execute novamente: trade-system paper")
        print("2. O erro 'required' is not defined deve estar resolvido")
    else:
        print("\n❌ Falha ao aplicar correção")
        print("Verifique o arquivo trade_system/analysis/ml.py manualmente")

if __name__ == "__main__":
    main()
