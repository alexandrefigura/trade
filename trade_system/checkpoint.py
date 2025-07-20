"""Sistema de checkpoint e recuperação"""
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle
import gzip

class CheckpointManager:
    """Gerencia checkpoints do sistema"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Diretório de checkpoints
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Configurações
        self.interval = config.system.get('checkpoint_interval', 300)
        self.max_checkpoints = 10
        
    async def checkpoint_loop(self):
        """Loop de salvamento periódico"""
        while True:
            await asyncio.sleep(self.interval)
            # O salvamento será chamado pelo sistema principal
    
    async def save_checkpoint(self, data: Dict[str, Any]) -> bool:
        """
        Salva checkpoint do sistema
        
        Args:
            data: Dados para salvar
            
        Returns:
            True se salvou com sucesso
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.checkpoint_dir / f"checkpoint_{timestamp}.json.gz"
            
            # Adicionar metadados
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'version': '2.0',
                'data': data
            }
            
            # Serializar e comprimir
            json_data = json.dumps(checkpoint, default=str)
            compressed = gzip.compress(json_data.encode())
            
            # Salvar
            with open(filename, 'wb') as f:
                f.write(compressed)
            
            self.logger.info(f"💾 Checkpoint salvo: {filename}")
            
            # Limpar checkpoints antigos
            await self._cleanup_old_checkpoints()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar checkpoint: {e}")
            return False
    
    async def load_checkpoint(self, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Carrega checkpoint
        
        Args:
            filename: Nome do arquivo ou None para carregar o mais recente
            
        Returns:
            Dados do checkpoint ou None
        """
        try:
            if filename:
                filepath = self.checkpoint_dir / filename
            else:
                # Buscar checkpoint mais recente
                checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json.gz"))
                if not checkpoints:
                    self.logger.info("Nenhum checkpoint encontrado")
                    return None
                filepath = checkpoints[-1]
            
            # Ler e descomprimir
            with open(filepath, 'rb') as f:
                compressed = f.read()
            
            json_data = gzip.decompress(compressed).decode()
            checkpoint = json.loads(json_data)
            
            self.logger.info(f"✅ Checkpoint carregado: {filepath}")
            
            return checkpoint['data']
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar checkpoint: {e}")
            return None
    
    async def _cleanup_old_checkpoints(self):
        """Remove checkpoints antigos"""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json.gz"))
            
            if len(checkpoints) > self.max_checkpoints:
                for checkpoint in checkpoints[:-self.max_checkpoints]:
                    checkpoint.unlink()
                    self.logger.debug(f"Checkpoint removido: {checkpoint}")
                    
        except Exception as e:
            self.logger.error(f"Erro ao limpar checkpoints: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Lista checkpoints disponíveis"""
        checkpoints = []
        
        for filepath in sorted(self.checkpoint_dir.glob("checkpoint_*.json.gz")):
            stat = filepath.stat()
            checkpoints.append({
                'filename': filepath.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'path': str(filepath)
            })
        
        return checkpoints
