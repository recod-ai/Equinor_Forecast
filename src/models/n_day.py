# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, GRU, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import CosineDecay
# from tensorflow.keras import metrics

# def create_model(
#     input_shape, 
#     future_step=1,               # Determines size of the final Dense layers
#     loss='mean_squared_error',   # Choose 'mean_squared_error' or 'quantile'
#     initial_lr=5e-5,             # Initial learning rate for CosineDecay
#     decay_steps=100*52,          # Number of decay steps for the CosineDecay schedule
#     seed=42                      # Fix seed for reproducibility
# ):
#     """
#     Adapts the GRU2_10 architecture to the Functional API while maintaining the same structure:
#       - 2 stacked GRU layers (each with 128 units)
#       - 10 Dense layers with progressive dimensionality
#       - Optionally adjusts output size for quantile regression
#       - Compiles with Adam + CosineDecay, matching the original approach.

#     Parameters:
#     -----------
#     input_shape : tuple
#         Shape of the input tensor (timesteps, features).
#     future_step : int
#         Number of units for the final Dense layer (e.g., how many steps to predict).
#     loss : str
#         'mean_squared_error' or 'quantile' (requires a custom QuantileLossCalculator).
#     initial_lr : float
#         Initial learning rate for CosineDecay.
#     decay_steps : int
#         Number of steps over which the learning rate decays following a cosine schedule.
#     seed : int
#         Seed for reproducibility.

#     Returns:
#     --------
#     model : tf.keras.Model
#         A compiled Keras Model matching the GRU2_10 architecture.
#     """
    
#     # 1) Set seed for reproducibility
#     # tf.random.set_seed(seed)
    
#     inputs = Input(shape=(input_shape.shape[1], input_shape.shape[2]))
#     x = inputs

#     # 3) Add the first GRU layer (returns sequences)
#     x = GRU(
#         units=128,
#         return_sequences=True,
#         name='GRU_layer_1'
#     )(x)
    
#     # Optionally, add dropout if needed
#     # x = Dropout(0.5, name='Dropout_GRU1')(x)

#     # 4) Add the second GRU layer (does not return sequences)
#     x = GRU(
#         units=128,
#         return_sequences=False,  # Last GRU layer should not return sequences
#         name='GRU_layer_2'
#     )(x)
    
#     # x = Dropout(0.5, name='Dropout_GRU2')(x)

#     # 5) Add Dense layers as per GRU2_10 architecture
#     #    - 2 layers of 128 units
#     x = Dense(128, activation='relu', name='Dense_128_1')(x)
#     x = Dense(128, activation='relu', name='Dense_128_2')(x)
    
#     #    - 3 layers of 64 units
#     x = Dense(64, activation='relu', name='Dense_64_1')(x)
#     x = Dense(64, activation='relu', name='Dense_64_2')(x)
#     x = Dense(64, activation='relu', name='Dense_64_3')(x)
    
#     #    - 3 layers of 32 units
#     x = Dense(32, activation='relu', name='Dense_32_1')(x)
#     x = Dense(32, activation='relu', name='Dense_32_2')(x)
#     x = Dense(32, activation='relu', name='Dense_32_3')(x)
    
#     #    - Final Dense layer with 'future_step' units
#     outputs = Dense(future_step, name='Dense_final_output')(x)

#     # 6) Define the model
#     model = Model(inputs=inputs, outputs=outputs, name='GRU2_10_Functional_API')

#     # 7) Configure optimizer with Cosine Decay
#     # lr_schedule = CosineDecay(
#     #     initial_learning_rate=initial_lr, 
#     #     decay_steps=decay_steps
#     # )
#     optimizer = Adam()

#     # 8) Compile the model
#     model.compile(
#         loss=loss,
#         optimizer=optimizer,
#         metrics=[metrics.mean_absolute_error]
#     )

#     return model


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics

from typing import Any, List, Dict, Tuple, Optional, Union

def create_model_3(
    input_shape, 
    future_step=1,               # Number of units for the final Dense layer in standalone mode
    loss='mean_squared_error',   # 'mean_squared_error' or 'quantile'
    initial_lr=5e-5,             # Initial learning rate (for CosineDecay if used)
    decay_steps=100*52,          # Steps over which learning rate decays (if using a schedule)
    seed=42,                     # Seed for reproducibility (if needed)
    as_branch=True              # If True, return a branch model (no final Dense or compile)
):
    """
    Creates the GRU2_10 architecture adapted for N‑Day forecasting.
    When as_branch=True, the function returns a branch model (i.e. the GRU and Dense layers)
    that outputs an intermediate feature vector. In standalone mode (as_branch=False), it adds
    a final Dense layer to produce `future_step` outputs and compiles the model.
    
    Parameters:
    -----------
    input_shape : tuple
        Input shape (timesteps, features).
    future_step : int
        Number of output units (e.g., forecast steps) for the final Dense layer.
    loss : str
        Loss function to use in standalone mode.
    initial_lr : float
        Initial learning rate (if using a learning rate schedule).
    decay_steps : int
        Decay steps for the learning rate schedule (if used).
    seed : int
        Seed for reproducibility.
    as_branch : bool
        If True, returns a branch model (no final Dense layer, not compiled) for multi‑branch integration.
        If False, returns a full model with a final Dense layer and compiles it.
    
    Returns:
    --------
    model : tf.keras.Model
        If as_branch is True, a branch model with .input and .output that can be merged.
        Otherwise, a complete compiled model.
    """
    
    # Create the input layer using the provided shape (e.g., (timesteps, features))
    inputs = Input(shape=input_shape)
    x = inputs

    # 1) Two stacked GRU layers
    x = GRU(units=128, return_sequences=True)(x)
    x = GRU(units=128, return_sequences=False)(x)
    
    # 2) Series of Dense layers as in the GRU2_10 architecture
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    
    if as_branch:
        # Return the branch (features only); the final prediction will be added later.
        branch_model = Model(inputs=inputs, outputs=x)
        return branch_model
    else:
        # Add a final Dense layer to output the forecast and compile the model.
        outputs = Dense(future_step)(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    
def create_model(
    input_shape: Optional[tuple] = None,
    input_shapes: Optional[List[tuple]] = None
) -> Model:
    """
    Creates a generic model. If input_shapes is provided (list), builds a multi-branch model;
    if a single input_shape is provided, builds a normal (single-input) model.
    
    For multi-branch models, each branch is built by calling create_model_3 in branch mode.
    Then, branch outputs are concatenated and passed through Dense layers for regression.
    """
    if input_shapes is not None:
        if not isinstance(input_shapes, list):
            raise ValueError("input_shapes must be a list of tuples")
        branch_inputs = []
        branch_outputs = []
        for shape in input_shapes:
            branch_model = create_model_3(shape, as_branch=True)
            branch_inputs.append(branch_model.input)
            branch_outputs.append(branch_model.output)
        combined = Concatenate()(branch_outputs) if len(branch_outputs) > 1 else branch_outputs[0]
        outputs = Dense(1, kernel_initializer='he_normal')(combined)
        model = Model(inputs=branch_inputs, outputs=outputs)
        return model

    elif input_shape is not None:
        # Single-input: use create_model_3 in branch mode, then add Dense layers.
        inputs = Input(shape=(input_shape.shape[1], input_shape.shape[2]))  # Ignore batch size
        x = create_model_3((input_shape.shape[1], input_shape.shape[2]))(inputs)
        outputs = Dense(1, kernel_initializer='he_normal')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
    else:
        raise ValueError("Either 'input_shape' or 'input_shapes' must be provided.")
