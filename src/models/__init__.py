import logging

# --- Imports das arquiteturas legadas (permanece o mesmo) ---
from .the_golem import create_model as create_model_arch1
from .n_day import create_model as create_model_arch2
from .generic_configurable import create_model as create_model_arch3
from .Seq2Value import create_model as create_model_arch4
from .conv_transform import create_model as create_model_arch5
from .Seq2Context import create_model as create_model_arch6
from .Seq2Fuser import create_model as create_model_arch7
from .Seq2PIN_Trend import create_model as create_model_arch9

# --- Import do NOVO builder e do carregador de configuração ---
from config_loader import load_experiment_configs # Usando '..' para subir um nível
from models.architecture_builder import ArchitectureBuilder

# Carrega as configurações das arquiteturas dinâmicas UMA VEZ
# Isso cria um cache para evitar leituras de disco repetidas
try:
    ALL_CONFIGS = load_experiment_configs()
    DYNAMIC_ARCH_CONFIGS = {
        arch.architecture_id: arch.model_dump()
        for arch in ALL_CONFIGS["architectures"]
        if arch.builder == "dynamic"
    }
except FileNotFoundError:
    logging.warning("Arquivos de configuração de experimento não encontrados. O builder dinâmico não estará disponível.")
    DYNAMIC_ARCH_CONFIGS = {}

# --- Registry de modelos legados (permanece o mesmo) ---
LEGACY_MODEL_REGISTRY = {
    'Golem': create_model_arch1,
    'N_day': create_model_arch2,
    'Generic': create_model_arch3,
    'Seq2Value': create_model_arch4,
    'Conv Trans': create_model_arch5,
    'Seq2Context': create_model_arch6,
    'Seq2Fuser': create_model_arch7,
    'Seq2Trend': create_model_arch9,
    'Seq2PIN': create_model_arch9,
}

def create_model(architecture_name: str, hp_config: dict = None, **kwargs):
    """
    Fábrica de modelos estendida.
    Cria um modelo a partir do registry legado ou dinamicamente a partir de uma configuração.

    Args:
        architecture_name (str): O ID da arquitetura (ex: 'Generic', 'Model1_1xConv').
        hp_config (dict, optional): O perfil de hiperparâmetros a ser aplicado.
        **kwargs: Argumentos para os modelos legados (ex: input_shape, use_lstm).

    Returns:
        Um modelo Keras compilado.
    """
    logging.info(f"Criando modelo para arquitetura: '{architecture_name}'")
    
    # 1. Tenta encontrar no registry de modelos LEGADOS primeiro
    if architecture_name in LEGACY_MODEL_REGISTRY:
        logging.info("Encontrado no registry de modelos legados.")
        create_fn = LEGACY_MODEL_REGISTRY[architecture_name]
        return create_fn(**kwargs)
        
    # 2. Se não encontrou, tenta construir a partir das configurações DINÂMICAS
    elif architecture_name in DYNAMIC_ARCH_CONFIGS:
        logging.info("Construindo modelo dinamicamente a partir da configuração.")
        arch_config = DYNAMIC_ARCH_CONFIGS[architecture_name]
        
        if 'input_shape' not in kwargs:
            raise ValueError("O argumento 'input_shape' é necessário para construir um modelo dinâmico.")
            
        builder = ArchitectureBuilder(arch_config, hp_config)
        return builder.build(kwargs['input_shape'])
        
    # 3. Se não encontrou em nenhum lugar, lança um erro
    else:
        raise ValueError(f"Arquitetura desconhecida: '{architecture_name}'. "
                         f"Não encontrada no registry legado nem nas configurações dinâmicas.")