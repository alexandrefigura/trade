"""
Sistema de checkpoint e recovery
"""
import os
import json
import pickle
import gzip
from datetime import datetime
from typing import Dict, Optional, Any
from trade_system.logging_config import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Gerenciador de checkpoints para recuperação rápida"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Configurações
        self.checkpoint_interval = 300  # 5 minutos
        self.last_checkpoint = 0
        self.max_checkpoints = 10
        self.use_compression = True
    
    def save_checkpoint(self, state: Dict[str, Any]) -> bool:
        """
        Salva estado do sistema em checkpoint
        
        Args:
            state: Dicionário com estado completo do sistema
            
        Returns:
            True se salvou com sucesso
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.checkpoint_dir, f"checkpoint_{timestamp}.pkl")
            
            # Adicionar metadados
            state['checkpoint_time'] = datetime.now()
            state['checkpoint_version'] = 'v5.2'
            
            # Salvar com ou sem compressão
            if self.use_compression:
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(state, f)
            else:
                with open(filename, 'wb') as f:
                    pickle.dump(state, f)
            
            # Limpar checkpoints antigos
            self._cleanup_old_checkpoints()
            
            logger.info(f"✅ Checkpoint salvo: {filename}")
            
            # Salvar resumo em JSON
            self._save_summary(state, filename)
            
            self.last_checkpoint = datetime.now().timestamp()
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {e}")
            return False
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Carrega checkpoint mais recente
        
        Returns:
            Estado do sistema ou None
        """
        try:
            checkpoint_file = self._get_latest_checkpoint_file()
            
            if not checkpoint_file:
                logger.info("Nenhum checkpoint encontrado")
                return None
            
            # Carregar com ou sem compressão
            if checkpoint_file.endswith('.pkl'):
                try:
                    with gzip.open(checkpoint_file, 'rb') as f:
                        state = pickle.load(f)
                except:
                    with open(checkpoint_file, 'rb') as f:
                        state = pickle.load(f)
            
            logger.info(f"✅ Checkpoint carregado: {checkpoint_file}")
            self._log_checkpoint_info(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint: {e}")
            return None
    
    def should_checkpoint(self) -> bool:
        """Verifica se deve fazer checkpoint"""
        return (datetime.now().timestamp() - self.last_checkpoint) > self.checkpoint_interval
    
    def update_checkpoint_time(self):
        """Atualiza tempo do último checkpoint"""
        self.last_checkpoint = datetime.now().timestamp()
    
    def list_checkpoints(self) -> list:
        """Lista todos os checkpoints disponíveis"""
        files = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("checkpoint_") and f.endswith(".pkl"):
                filepath = os.path.join(self.checkpoint_dir, f)
                summary_file = filepath.replace('.pkl', '_summary.json')
                
                info = {
                    'filename': f,
                    'filepath': filepath,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
                }
                
                # Tentar carregar resumo
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, 'r') as sf:
                            info['summary'] = json.load(sf)
                    except:
                        pass
                
                files.append(info)
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    def delete_checkpoint(self, filename: str) -> bool:
        """
        Deleta um checkpoint específico
        
        Args:
            filename: Nome do arquivo do checkpoint
            
        Returns:
            True se deletou com sucesso
        """
        try:
            filepath = os.path.join(self.checkpoint_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                
                # Remover summary também
                summary_file = filepath.replace('.pkl', '_summary.json')
                if os.path.exists(summary_file):
                    os.remove(summary_file)
                
                logger.info(f"Checkpoint deletado: {filename}")
                return True
            return False
        except Exception as e:
            logger.error(f"Erro ao deletar checkpoint: {e}")
            return False
    
    def _get_latest_checkpoint_file(self) -> Optional[str]:
        """Retorna caminho do checkpoint mais recente"""
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[0]['filepath']
        return None
    
    def _cleanup_old_checkpoints(self):
        """Remove checkpoints antigos, mantendo apenas os N mais recentes"""
        try:
            checkpoints = self.list_checkpoints()
            
            if len(checkpoints) > self.max_checkpoints:
                for checkpoint in checkpoints[self.max_checkpoints:]:
                    self.delete_checkpoint(checkpoint['filename'])
                    
        except Exception as e:
            logger.error(f"Erro ao limpar checkpoints: {e}")
    
    def _save_summary(self, state: Dict, checkpoint_file: str):
        """Salva resumo do checkpoint em JSON"""
        try:
            summary_file = checkpoint_file.replace('.pkl', '_summary.json')
            
            summary = {
                'timestamp': state.get('checkpoint_time', datetime.now()).isoformat(),
                'balance': state.get('balance', 0),
                'total_trades': state.get('performance_stats', {}).get('total_trades', 0),
                'daily_pnl': state.get('daily_pnl', 0),
                'position': bool(state.get('position')),
                'is_paper_trading': state.get('is_paper_trading', True)
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logger.debug(f"Erro ao salvar resumo: {e}")
    
    def _log_checkpoint_info(self, state: Dict):
        """Loga informações do checkpoint carregado"""
        try:
            logger.info(f"   Balanço: ${state.get('balance', 0):,.2f}")
            logger.info(f"   Trades: {state.get('performance_stats', {}).get('total_trades', 0)}")
            logger.info(f"   Tempo: {state.get('checkpoint_time', 'N/A')}")
            
            if state.get('position'):
                pos = state['position']
                logger.info(f"   Posição aberta: {pos.get('side', 'N/A')} @ ${pos.get('entry_price', 0):,.2f}")
        except:
            pass
