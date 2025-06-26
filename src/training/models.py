# models/__init__.py
from .models.the_golem import create_model as create_model_arch1
from .models.n_day import create_model as create_model_arch2
from .models.generic_configurable import create_model as create_model_arch3
from .models.transformer_encoder_decoder import create_model as create_model_arch4
# Import additional architectures as needed

from utilities import print_style

# Registry mapping architecture names to their creation functions
MODEL_REGISTRY = {
    'Golem': create_model_arch1,
    'N_day': create_model_arch2,
    'Generic': create_model_arch3,
    'Encoder': create_model_arch4,
}

def create_model(architecture_name, **kwargs):
    """
    Factory function to create models based on architecture name.

    Parameters:
    - architecture_name (str): Key identifying the architecture.
    - **kwargs: Arbitrary keyword arguments for the specific architecture.

    Returns:
    - Compiled Keras model.
    """
    if architecture_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture_name}")
    
    create_fn = MODEL_REGISTRY[architecture_name]
    print_style(f"Architecture: {architecture_name}")
    return create_fn(**kwargs)
