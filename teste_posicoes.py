"""
Script de teste completo para o sistema de trading
"""
import asyncio
import time
from trade_system.config import get_config
from trade_system.main import TradingSystem
from trade_system.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


async def test_trading_system():
    """Testa o sistema de trading com tempo adequado"""
    
    # Setup
    setup_logging(log_level="INFO")
    
    # Configuração debug
    config = get_config(debug_mode=True)
    config.min_confidence = 0.20  # Apenas 20% de confiança!
    config.force_first_trade = True  # Forçar primeiro trade
    config.main_loop_interval_ms = 500  # Loop mais rápido (0.5s)
    
    logger.info(f"""
🔧 CONFIGURAÇÃO DE TESTE:
- min_confidence: {config.min_confidence}
- force_first_trade: {config.force_first_trade}
- max_position_pct: {config.max_position_pct}
- main_loop_interval_ms: {config.main_loop_interval_ms}
    """)
    
    # Criar sistema
    system = TradingSystem(config, paper_trading=True)
    system.risk_manager.current_balance = 10000
    system.risk_manager.initial_balance = 10000
    
    # Inicializar
    await system.initialize()
    
    # Criar task para executar o sistema
    system_task = asyncio.create_task(system.run())
    
    # Aguardar dados iniciais
    logger.info("⏳ Aguardando sistema estabilizar...")
    await asyncio.sleep(5)
    
    # Monitorar por 2 minutos
    start_time = time.time()
    monitor_duration = 120  # 2 minutos
    
    logger.info(f"📊 Monitorando sistema por {monitor_duration} segundos...")
    
    last_log_time = time.time()
    while time.time() - start_time < monitor_duration:
        # Log a cada 15 segundos
        if time.time() - last_log_time >= 15:
            # Obter dados atuais
            market_data = system.ws_manager.get_latest_data()
            if market_data and len(market_data['prices']) > 0:
                current_price = float(market_data['prices'][-1])
            else:
                current_price = 0
            
            logger.info(f"""
⏱️ Tempo: {int(time.time() - start_time)}s
💰 Balance: ${system.risk_manager.current_balance:,.2f}
📊 Trades: {system.performance_stats['total_trades']}
💵 P&L: ${system.performance_stats['total_pnl']:,.2f}
📈 Preço: ${current_price:,.2f}
🎯 Posição: {'SIM - ' + system.position['side'] if system.position else 'NÃO'}
            """)
            
            last_log_time = time.time()
        
        # Verificar se deve parar
        if not system.is_running:
            logger.warning("Sistema parou inesperadamente!")
            break
        
        await asyncio.sleep(1)
    
    # Parar sistema
    logger.info("⏹️ Parando sistema...")
    system.is_running = False
    
    # Aguardar task terminar
    try:
        await asyncio.wait_for(system_task, timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Timeout ao aguardar sistema parar")
        system_task.cancel()
    
    # Resultados finais
    logger.info(f"""
    
📊 RESULTADOS FINAIS DO TESTE:
════════════════════════════════════════════
💰 Balance Inicial: $10,000.00
💰 Balance Final: ${system.risk_manager.current_balance:,.2f}
📈 Retorno: {(system.risk_manager.current_balance - 10000) / 10000 * 100:+.2f}%
📊 Total de Trades: {system.performance_stats['total_trades']}
✅ Trades Vencedores: {system.performance_stats['winning_trades']}
💵 P&L Total: ${system.performance_stats['total_pnl']:,.2f}
💸 Taxas Totais: ${system.performance_stats['total_fees']:,.2f}
⏱️ Duração: {monitor_duration}s
════════════════════════════════════════════
    """)
    
    # Se não executou nenhum trade, dar dicas
    if system.performance_stats['total_trades'] == 0:
        logger.warning("""
⚠️ NENHUM TRADE EXECUTADO!

Possíveis causas:
1. Mercado muito estável (sem sinais fortes)
2. Ainda precisa de mais tempo para análise
3. Parâmetros ainda muito conservadores

Sugestões:
- Execute por mais tempo (5-10 minutos)
- Reduza min_confidence ainda mais (ex: 0.1)
- Verifique se está recebendo dados (olhe os logs do WebSocket)
        """)


async def test_forced_trade():
    """Testa abertura forçada de posição"""
    setup_logging(log_level="INFO")
    
    config = get_config(debug_mode=True)
    config.min_confidence = 0.01  # Quase zero!
    
    system = TradingSystem(config, paper_trading=True)
    await system.initialize()
    
    # Aguardar dados
    await asyncio.sleep(3)
    
    # Forçar abertura imediata
    market_data = system.ws_manager.get_latest_data()
    if market_data:
        logger.info("🔧 FORÇANDO posição de teste!")
        await system._open_paper_position(market_data, 'BUY', 0.8)
        
        # Aguardar um pouco
        await asyncio.sleep(10)
        
        # Forçar fechamento
        if system.position:
            current_price = float(market_data['prices'][-1])
            await system._close_position(current_price, "Teste forçado")
    
    await system.shutdown()


async def main():
    """Menu principal"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              SISTEMA DE TRADING - TESTES                     ║
╚══════════════════════════════════════════════════════════════╝

Escolha uma opção:

1. Teste Completo (2 minutos)
2. Teste Rápido com Trade Forçado
3. Sair

""")
    
    choice = input("Opção: ").strip()
    
    if choice == "1":
        await test_trading_system()
    elif choice == "2":
        await test_forced_trade()
    else:
        print("Saindo...")


if __name__ == "__main__":
    asyncio.run(main())