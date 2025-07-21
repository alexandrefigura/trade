#!/usr/bin/env python3
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
    
    print("\n🚀 Tente executar novamente:")
    print("   python paper_trading_fixed.py")

if __name__ == "__main__":
    apply_patch()
