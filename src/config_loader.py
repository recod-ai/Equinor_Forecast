# Arquivo: src/config_loader.py

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Modelos de Dados Pydantic para Validação ---

class BlockConfig(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

class ArchitectureConfig(BaseModel):
    architecture_id: str
    summary: str
    builder: str # "dynamic" ou "legacy"
    blocks: Optional[List[BlockConfig]] = None
    params: Optional[Dict[str, Any]] = None # Para modelos 'legacy'

class HyperparameterProfile(BaseModel):
    hyperparam_id: str
    summary: str
    overrides: Dict[str, Any] = Field(default_factory=dict)

# --- Funções de Carregamento ---

def load_yaml_config(file_path: Path) -> List[Dict]:
    """Carrega e analisa um arquivo de configuração YAML."""
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_experiment_configs(config_dir: str | Path = "experiment_configs"):
    """
    Carrega e valida todas as configurações de arquitetura e hiperparâmetros.
    """
    base_path = Path(__file__).resolve().parent / config_dir
    
    # Carrega e valida arquiteturas
    arch_data = load_yaml_config(base_path / "architectures.yaml")
    architectures = [ArchitectureConfig(**item) for item in arch_data]
    
    # Carrega e valida hiperparâmetros
    hp_data = load_yaml_config(base_path / "hyperparameters.yaml")
    hyperparams = [HyperparameterProfile(**item) for item in hp_data]
    
    print(f"INFO: Carregadas {len(architectures)} arquiteturas e {len(hyperparams)} perfis de hiperparâmetros.")
    
    return {
        "architectures": architectures,
        "hyperparameters": hyperparams
    }

# Exemplo de como usar (para teste)
if __name__ == "__main__":
    try:
        configs = load_experiment_configs()
        # Imprime a primeira arquitetura para verificar
        print("\nExemplo de Arquitetura Carregada:")
        print(configs["architectures"][0].model_dump_json(indent=2))
        
        # Imprime o primeiro perfil de HP para verificar
        print("\nExemplo de Perfil de Hiperparâmetro Carregado:")
        print(configs["hyperparameters"][0].model_dump_json(indent=2))
        
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        print("Certifique-se de que a pasta 'experiment_configs' existe e contém os arquivos YAML.")
    except Exception as e:
        print(f"Ocorreu um erro ao validar as configurações: {e}")