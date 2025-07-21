@echo off
chcp 65001 > nul
echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║        LIMPEZA DO REPOSITÓRIO TRADE                  ║
echo ╠══════════════════════════════════════════════════════╣
echo ║  Este script vai DELETAR permanentemente:            ║
echo ║  - Todos os scripts de correção (fix_*.py)          ║
echo ║  - Scripts de teste criados                         ║
echo ║  - Arquivos temporários                             ║
echo ║  - Backups desnecessários                           ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo ATENÇÃO: Os arquivos serão DELETADOS permanentemente!
echo.
pause
echo.
echo Iniciando limpeza...
echo.
if exist "fix_all_class_names.py" (
    del /f /q "fix_all_class_names.py"
    echo [DELETADO] fix_all_class_names.py
)
if exist "fix_all_features.py" (
    del /f /q "fix_all_features.py"
    echo [DELETADO] fix_all_features.py
)
if exist "fix_all_imports.py" (
    del /f /q "fix_all_imports.py"
    echo [DELETADO] fix_all_imports.py
)
if exist "fix_all_imports2.py" (
    del /f /q "fix_all_imports2.py"
    echo [DELETADO] fix_all_imports2.py
)
if exist "fix_all_imports_final.py" (
    del /f /q "fix_all_imports_final.py"
    echo [DELETADO] fix_all_imports_final.py
)
if exist "fix_all_typing.py" (
    del /f /q "fix_all_typing.py"
    echo [DELETADO] fix_all_typing.py
)
if exist "fix_analysis_modules.py" (
    del /f /q "fix_analysis_modules.py"
    echo [DELETADO] fix_analysis_modules.py
)
if exist "fix_cache_import.py" (
    del /f /q "fix_cache_import.py"
    echo [DELETADO] fix_cache_import.py
)
if exist "fix_class_names.py" (
    del /f /q "fix_class_names.py"
    echo [DELETADO] fix_class_names.py
)
if exist "fix_datetime_error.py" (
    del /f /q "fix_datetime_error.py"
    echo [DELETADO] fix_datetime_error.py
)
if exist "fix_env_loading.py" (
    del /f /q "fix_env_loading.py"
    echo [DELETADO] fix_env_loading.py
)
if exist "fix_everything.py" (
    del /f /q "fix_everything.py"
    echo [DELETADO] fix_everything.py
)
if exist "fix_from_env.py" (
    del /f /q "fix_from_env.py"
    echo [DELETADO] fix_from_env.py
)
if exist "fix_import.py" (
    del /f /q "fix_import.py"
    echo [DELETADO] fix_import.py
)
if exist "fix_imports_integrated.py" (
    del /f /q "fix_imports_integrated.py"
    echo [DELETADO] fix_imports_integrated.py
)
if exist "fix_indentation_error.py" (
    del /f /q "fix_indentation_error.py"
    echo [DELETADO] fix_indentation_error.py
)
if exist "fix_ml_error.py" (
    del /f /q "fix_ml_error.py"
    echo [DELETADO] fix_ml_error.py
)
if exist "fix_ml_predictor_issue.py" (
    del /f /q "fix_ml_predictor_issue.py"
    echo [DELETADO] fix_ml_predictor_issue.py
)
if exist "fix_momentum.py" (
    del /f /q "fix_momentum.py"
    echo [DELETADO] fix_momentum.py
)
if exist "fix_numpy_import.py" (
    del /f /q "fix_numpy_import.py"
    echo [DELETADO] fix_numpy_import.py
)
if exist "fix_technical_class.py" (
    del /f /q "fix_technical_class.py"
    echo [DELETADO] fix_technical_class.py
)
if exist "fix_technical_properly.py" (
    del /f /q "fix_technical_properly.py"
    echo [DELETADO] fix_technical_properly.py
)
if exist "fix_telegram.py" (
    del /f /q "fix_telegram.py"
    echo [DELETADO] fix_telegram.py
)
if exist "fix_trading_system.py" (
    del /f /q "fix_trading_system.py"
    echo [DELETADO] fix_trading_system.py
)
if exist "fix_typing_imports.py" (
    del /f /q "fix_typing_imports.py"
    echo [DELETADO] fix_typing_imports.py
)
if exist "fix_websocket_args.py" (
    del /f /q "fix_websocket_args.py"
    echo [DELETADO] fix_websocket_args.py
)
if exist "fix-walrus.patch" (
    del /f /q "fix-walrus.patch"
    echo [DELETADO] fix-walrus.patch
)
if exist "analyze_and_fix.py" (
    del /f /q "analyze_and_fix.py"
    echo [DELETADO] analyze_and_fix.py
)
if exist "check_and_fix_classes.py" (
    del /f /q "check_and_fix_classes.py"
    echo [DELETADO] check_and_fix_classes.py
)
if exist "diagnose_system.py" (
    del /f /q "diagnose_system.py"
    echo [DELETADO] diagnose_system.py
)
if exist "diagnostic.py" (
    del /f /q "diagnostic.py"
    echo [DELETADO] diagnostic.py
)
if exist "list_directory.py" (
    del /f /q "list_directory.py"
    echo [DELETADO] list_directory.py
)
if exist "test_config.py" (
    del /f /q "test_config.py"
    echo [DELETADO] test_config.py
)
if exist "working_paper_trading.py" (
    del /f /q "working_paper_trading.py"
    echo [DELETADO] working_paper_trading.py
)
if exist "integrated_paper_trading.py" (
    del /f /q "integrated_paper_trading.py"
    echo [DELETADO] integrated_paper_trading.py
)
if exist "integrated_trading_clean.py" (
    del /f /q "integrated_trading_clean.py"
    echo [DELETADO] integrated_trading_clean.py
)
if exist "final_integrated_system.py" (
    del /f /q "final_integrated_system.py"
    echo [DELETADO] final_integrated_system.py
)
if exist "final_working_system.py" (
    del /f /q "final_working_system.py"
    echo [DELETADO] final_working_system.py
)
if exist "paper_trading.py" (
    del /f /q "paper_trading.py"
    echo [DELETADO] paper_trading.py
)
if exist "paper_trading_final.py" (
    del /f /q "paper_trading_final.py"
    echo [DELETADO] paper_trading_final.py
)
if exist "paper_trading_fixed.py" (
    del /f /q "paper_trading_fixed.py"
    echo [DELETADO] paper_trading_fixed.py
)
if exist "run_paper_trading.py" (
    del /f /q "run_paper_trading.py"
    echo [DELETADO] run_paper_trading.py
)
if exist "run_trading.py" (
    del /f /q "run_trading.py"
    echo [DELETADO] run_trading.py
)
if exist "run_with_env.py" (
    del /f /q "run_with_env.py"
    echo [DELETADO] run_with_env.py
)
if exist "run_with_fixes.py" (
    del /f /q "run_with_fixes.py"
    echo [DELETADO] run_with_fixes.py
)
if exist "simple_run.py" (
    del /f /q "simple_run.py"
    echo [DELETADO] simple_run.py
)
if exist "quick_solution.py" (
    del /f /q "quick_solution.py"
    echo [DELETADO] quick_solution.py
)
if exist "apply_patch.py" (
    del /f /q "apply_patch.py"
    echo [DELETADO] apply_patch.py
)
if exist "create_missing_modules.py" (
    del /f /q "create_missing_modules.py"
    echo [DELETADO] create_missing_modules.py
)
if exist "create_trading_system_zip.py" (
    del /f /q "create_trading_system_zip.py"
    echo [DELETADO] create_trading_system_zip.py
)
if exist "enhance_paper_trading.py" (
    del /f /q "enhance_paper_trading.py"
    echo [DELETADO] enhance_paper_trading.py
)
if exist "final_fix.py" (
    del /f /q "final_fix.py"
    echo [DELETADO] final_fix.py
)
if exist "force_more_trades.py" (
    del /f /q "force_more_trades.py"
    echo [DELETADO] force_more_trades.py
)
if exist "force_paper_trades.py" (
    del /f /q "force_paper_trades.py"
    echo [DELETADO] force_paper_trades.py
)
if exist "force_trade.py" (
    del /f /q "force_trade.py"
    echo [DELETADO] force_trade.py
)
if exist "generated_fix.py" (
    del /f /q "generated_fix.py"
    echo [DELETADO] generated_fix.py
)
if exist "get_telegram_id.py" (
    del /f /q "get_telegram_id.py"
    echo [DELETADO] get_telegram_id.py
)
if exist "monitor.py" (
    del /f /q "monitor.py"
    echo [DELETADO] monitor.py
)
if exist "monitor_live.py" (
    del /f /q "monitor_live.py"
    echo [DELETADO] monitor_live.py
)
if exist "monitor_trades.py" (
    del /f /q "monitor_trades.py"
    echo [DELETADO] monitor_trades.py
)
if exist "reset_system.py" (
    del /f /q "reset_system.py"
    echo [DELETADO] reset_system.py
)
if exist "add_all_aliases.py" (
    del /f /q "add_all_aliases.py"
    echo [DELETADO] add_all_aliases.py
)
if exist "clean_fix_imports.py" (
    del /f /q "clean_fix_imports.py"
    echo [DELETADO] clean_fix_imports.py
)
if exist "analysis_report.json" (
    del /f /q "analysis_report.json"
    echo [DELETADO] analysis_report.json
)
if exist "file_list.txt" (
    del /f /q "file_list.txt"
    echo [DELETADO] file_list.txt
)
if exist "trade_improvements.patch" (
    del /f /q "trade_improvements.patch"
    echo [DELETADO] trade_improvements.patch
)
if exist "trading_system_missing_modules_20250717_201611.zip" (
    del /f /q "trading_system_missing_modules_20250717_201611.zip"
    echo [DELETADO] trading_system_missing_modules_20250717_201611.zip
)
if exist "trading_system_v5.2_20250717_201621.zip" (
    del /f /q "trading_system_v5.2_20250717_201621.zip"
    echo [DELETADO] trading_system_v5.2_20250717_201621.zip
)
if exist "config.yaml.backup" (
    del /f /q "config.yaml.backup"
    echo [DELETADO] config.yaml.backup
)
if exist "config.yaml.backup_paper" (
    del /f /q "config.yaml.backup_paper"
    echo [DELETADO] config.yaml.backup_paper
)
if exist "backup_checkpoint_20250718_100834.pkl" (
    del /f /q "backup_checkpoint_20250718_100834.pkl"
    echo [DELETADO] backup_checkpoint_20250718_100834.pkl
)

echo.
echo Removendo backups em trade_system...
del /f /q "trade_system\*.backup*" 2>nul
del /f /q "trade_system\analysis\*.backup*" 2>nul

echo.
echo Removendo pasta backup_files...
if exist "backup_files" (
    rmdir /s /q "backup_files"
    echo [DELETADO] pasta backup_files
)

echo.
echo ══════════════════════════════════════════════════════
echo ✅ LIMPEZA CONCLUÍDA!
echo ══════════════════════════════════════════════════════
echo.
echo Próximos passos:
echo 1. git add -A
echo 2. git commit -m "Limpeza do repositório - removidos arquivos desnecessários"
echo 3. git push origin main
echo.
pause
