#!/usr/bin/env python3
"""
Script de limpeza específico para o repositório trade
Remove todos os arquivos desnecessários mantendo apenas o sistema original
"""
import os
import shutil

def create_cleanup_commands():
    """Cria comandos de limpeza para Windows"""
    
    print("🧹 LIMPEZA DO REPOSITÓRIO TRADE")
    print("=" * 70)
    
    # Lista de arquivos para DELETAR (todos os scripts de teste/fix)
    files_to_delete = [
        # Scripts de correção (fix_*.py)
        "fix_all_class_names.py",
        "fix_all_features.py",
        "fix_all_imports.py",
        "fix_all_imports2.py",
        "fix_all_imports_final.py",
        "fix_all_typing.py",
        "fix_analysis_modules.py",
        "fix_cache_import.py",
        "fix_class_names.py",
        "fix_datetime_error.py",
        "fix_env_loading.py",
        "fix_everything.py",
        "fix_from_env.py",
        "fix_import.py",
        "fix_imports_integrated.py",
        "fix_indentation_error.py",
        "fix_ml_error.py",
        "fix_ml_predictor_issue.py",
        "fix_momentum.py",
        "fix_numpy_import.py",
        "fix_technical_class.py",
        "fix_technical_properly.py",
        "fix_telegram.py",
        "fix_trading_system.py",
        "fix_typing_imports.py",
        "fix_websocket_args.py",
        "fix-walrus.patch",
        
        # Scripts de análise/diagnóstico
        "analyze_and_fix.py",
        "check_and_fix_classes.py",
        "diagnose_system.py",
        "diagnostic.py",
        "list_directory.py",
        "test_config.py",
        
        # Scripts de execução alternativos
        "working_paper_trading.py",
        "integrated_paper_trading.py",
        "integrated_trading_clean.py",
        "final_integrated_system.py",
        "final_working_system.py",
        "paper_trading.py",
        "paper_trading_final.py",
        "paper_trading_fixed.py",
        "run_paper_trading.py",
        "run_trading.py",
        "run_with_env.py",
        "run_with_fixes.py",
        "simple_run.py",
        "quick_solution.py",
        
        # Scripts auxiliares
        "apply_patch.py",
        "create_missing_modules.py",
        "create_trading_system_zip.py",
        "enhance_paper_trading.py",
        "final_fix.py",
        "force_more_trades.py",
        "force_paper_trades.py",
        "force_trade.py",
        "generated_fix.py",
        "get_telegram_id.py",
        "monitor.py",
        "monitor_live.py",
        "monitor_trades.py",
        "reset_system.py",
        "add_all_aliases.py",
        "clean_fix_imports.py",
        
        # Arquivos temporários
        "analysis_report.json",
        "file_list.txt",
        "trade_improvements.patch",
        
        # Arquivos zip antigos
        "trading_system_missing_modules_20250717_201611.zip",
        "trading_system_v5.2_20250717_201621.zip",
        
        # Backups de configuração
        "config.yaml.backup",
        "config.yaml.backup_paper",
        
        # Backup de checkpoint antigo
        "backup_checkpoint_20250718_100834.pkl",
    ]
    
    # Criar script batch para Windows
    with open('LIMPAR_REPOSITORIO.bat', 'w', encoding='utf-8') as f:
        f.write('@echo off\n')
        f.write('chcp 65001 > nul\n')  # UTF-8
        f.write('echo.\n')
        f.write('echo ╔══════════════════════════════════════════════════════╗\n')
        f.write('echo ║        LIMPEZA DO REPOSITÓRIO TRADE                  ║\n')
        f.write('echo ╠══════════════════════════════════════════════════════╣\n')
        f.write('echo ║  Este script vai DELETAR permanentemente:            ║\n')
        f.write('echo ║  - Todos os scripts de correção (fix_*.py)          ║\n')
        f.write('echo ║  - Scripts de teste criados                         ║\n')
        f.write('echo ║  - Arquivos temporários                             ║\n')
        f.write('echo ║  - Backups desnecessários                           ║\n')
        f.write('echo ╚══════════════════════════════════════════════════════╝\n')
        f.write('echo.\n')
        f.write('echo ATENÇÃO: Os arquivos serão DELETADOS permanentemente!\n')
        f.write('echo.\n')
        f.write('pause\n')
        f.write('echo.\n')
        f.write('echo Iniciando limpeza...\n')
        f.write('echo.\n')
        
        # Deletar arquivos individuais
        for file in files_to_delete:
            f.write(f'if exist "{file}" (\n')
            f.write(f'    del /f /q "{file}"\n')
            f.write(f'    echo [DELETADO] {file}\n')
            f.write(')\n')
        
        # Deletar todos os backups em trade_system
        f.write('\necho.\n')
        f.write('echo Removendo backups em trade_system...\n')
        f.write('del /f /q "trade_system\\*.backup*" 2>nul\n')
        f.write('del /f /q "trade_system\\analysis\\*.backup*" 2>nul\n')
        
        # Deletar pasta backup_files
        f.write('\necho.\n')
        f.write('echo Removendo pasta backup_files...\n')
        f.write('if exist "backup_files" (\n')
        f.write('    rmdir /s /q "backup_files"\n')
        f.write('    echo [DELETADO] pasta backup_files\n')
        f.write(')\n')
        
        f.write('\necho.\n')
        f.write('echo ══════════════════════════════════════════════════════\n')
        f.write('echo ✅ LIMPEZA CONCLUÍDA!\n')
        f.write('echo ══════════════════════════════════════════════════════\n')
        f.write('echo.\n')
        f.write('echo Próximos passos:\n')
        f.write('echo 1. git add -A\n')
        f.write('echo 2. git commit -m "Limpeza do repositório - removidos arquivos desnecessários"\n')
        f.write('echo 3. git push origin main\n')
        f.write('echo.\n')
        f.write('pause\n')

def show_final_structure():
    """Mostra a estrutura final esperada"""
    print("\n📁 ESTRUTURA FINAL APÓS LIMPEZA:")
    print("=" * 70)
    print("""
TRADE/
├── .env                          # Suas API keys
├── .env.example                  # Exemplo de .env
├── .gitignore                    # Ignorar arquivos
├── config.yaml                   # Configuração do sistema
├── INSTALAMENTO.txt              # Instruções de instalação
├── LICENSE                       # Licença do projeto
├── README.md                     # Documentação
├── requirements.txt              # Dependências Python
├── setup.py                      # Instalação do sistema
├── trade_system/                 # SISTEMA PRINCIPAL
│   ├── __init__.py
│   ├── alerts.py                 # Sistema de alertas
│   ├── backtester.py            # Backtesting
│   ├── cache.py                 # Cache Redis
│   ├── checkpoint.py            # Checkpoints
│   ├── cli.py                   # Interface CLI
│   ├── config.py                # Configurações
│   ├── learning.py              # Machine Learning
│   ├── logging_config.py        # Logs
│   ├── main.py                  # Sistema principal
│   ├── paper_trader.py          # Paper trading
│   ├── rate_limiter.py          # Rate limiting
│   ├── risk.py                  # Gestão de risco
│   ├── signals.py               # Sinais de trading
│   ├── trade_logger.py          # Log de trades
│   ├── utils.py                 # Utilidades
│   ├── validation.py            # Validações
│   ├── websocket_manager.py     # WebSocket
│   └── analysis/                # Análises
│       ├── __init__.py
│       ├── ml.py                # Machine Learning
│       ├── orderbook.py         # Order book
│       └── technical.py         # Análise técnica
├── checkpoints/                 # Checkpoints salvos
├── logs/                        # Logs do sistema
├── models/                      # Modelos ML treinados
│   ├── metrics.json
│   ├── scaler.pkl
│   └── trading_model.pkl
└── .git/                        # Controle de versão
    """)

def main():
    """Função principal"""
    create_cleanup_commands()
    show_final_structure()
    
    print("\n⚠️  IMPORTANTE:")
    print("1. Execute 'LIMPAR_REPOSITORIO.bat' para limpar")
    print("2. Isso vai deletar APENAS localmente")
    print("3. Para remover do GitHub, faça git add/commit/push")
    print("\n📌 Comando principal do sistema após limpeza:")
    print("   trade-system paper")

if __name__ == "__main__":
    main()