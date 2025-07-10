from __future__ import annotations
from typing import List, Dict, Tuple, Union, Sequence, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation.evaluation import plot_predictions

import numpy as np
import pandas as pd
import random
import logging

# augment_phys.py
# ---------------------------------------------------------------------------
#  Data-augmentation modular para séries temporais de produção (óleo & gás)
#  - Mantém a “redução de escala” do alvo (multiplicativa) mas de forma
#    fisicamente coerente.
#  - Permite ligar/desligar técnicas por flags.
#  - Funciona direto no pipeline (substitui augment_with_synthetic_samples).
# ---------------------------------------------------------------------------


# ------------------------ helpers de consistência --------------------------
def _ensure_ndarray(x) -> np.ndarray:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    return np.asarray(x)


def _concat(orig: np.ndarray, synt: List[np.ndarray], like):
    out = np.concatenate([orig, *synt], axis=0)
    if isinstance(like, (pd.DataFrame, pd.Series)):
        # reconstrói mesmo tipo/índices – assume reset de índice é ok para treino
        return type(like)(out, columns=getattr(like, "columns", None))
    return out


# ------------------------------ técnicas -----------------------------------
def _scale_decline(
    X: np.ndarray,
    y: np.ndarray,
    feat_idx: Dict[str, int],
    factors: Sequence[float]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Multiplica Q, ΔP (e opcionalmente gas/water) por 1/fator para
    simular estrangulamentos ou regimes pobres de pressão.
    Mantém PI fixo (não escalona).
    """
    synth_X, synth_y = [], []
    q_idx   = feat_idx["BORE_OIL_VOL"]
    adp_idx  = feat_idx.get("AVG_DOWNHOLE_PRESSURE")       # pode não existir em todos os datasets
    gas_idx = feat_idx.get("BORE_GAS_VOL")
    pi_idx = feat_idx.get("PI")

    
    for f in factors:
        X_new = X.copy()
        y_new = y.copy()
        X_new[..., q_idx]   /= f
        y_new       /= f
        if adp_idx is not None:
            X_new[..., adp_idx]  /= f
        if gas_idx is not None:
            X_new[..., gas_idx] /= f   # mantém RGO constante
        if pi_idx is not None:
            X_new[..., pi_idx] /= f
        synth_X.append(X_new); synth_y.append(y_new)
    return synth_X, synth_y


def _jitter_pressure(
    X: np.ndarray,
    feat_idx: Dict[str, int],
    sigma_psi: float = 10.0
) -> np.ndarray:
    """
    Adiciona ruído gaussiano leve às pressões para simular incerteza de sensor.
    """
    pf_idx = feat_idx.get("AVG_WHP_P")
    pr_idx = feat_idx.get("AVG_DOWNHOLE_PRESSURE")
    if pf_idx is None and pr_idx is None:
        return X
    X_new = X.copy()
    noise = np.random.normal(0.0, sigma_psi, size=X_new[..., 0].shape)
    if pf_idx is not None:
        X_new[..., pf_idx] += noise
    if pr_idx is not None:
        X_new[..., pr_idx] += noise
    return X_new


def _warp_decline(
    X: np.ndarray,
    y: np.ndarray,
    feat_idx: Dict[str, int],
    alpha_range=(0.6, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica “declínio tardio’’: a partir de um ponto aleatório,
    multiplica Q e ΔP por α∈(0.6-1.0).  Mantém PI.
    """
    T = X.shape[1]
    cut = random.randint(T // 3, T - 1)      # não muito cedo
    α = random.uniform(*alpha_range)

    X_new, y_new = X.copy(), y.copy()
    q_idx  = feat_idx["BORE_OIL_VOL"]
    dp_idx = feat_idx.get("delta_P")

    X_new[:, cut:, q_idx]  *= α
    y_new[:, cut:]         *= α
    if dp_idx is not None:
        X_new[:, cut:, dp_idx] *= α
    return X_new, y_new


def _block_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Re-amostra janelas inteiras (block bootstrap).  Preserva correlação
    intrajanel a preço quase zero de implementação.
    """
    idx = np.random.randint(0, X.shape[0], size=n_samples)
    return [X[idx]], [y[idx]]


# -------------------------- função principal --------------------------------
def augment_phys(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series,  np.ndarray],
    *,
    feature_indices: Dict[str, int],
    scale_factors: Sequence[float] = (1.5, 2, 3, 5),
    use_scale: bool = True,
    use_jitter: bool = True,
    use_decline_warp: bool = True,
    n_bootstrap: int = 0,
    sigma_pressure: float = 10.0,
    random_state: int | None = None
) -> Tuple[Union[pd.DataFrame, np.ndarray],
           Union[pd.Series,   np.ndarray]]:
    """
    Augmenta dados físicos de forma incremental.  Cada técnica pode ser
    ligada/desligada por flags.

    Parameters
    ----------
    X_train , y_train : tipos iguais aos do pipeline (np.ndarray ou DataFrame)
                        • X window shape esperado: (N, T, F)
                        • y shape              : (N, T_out)
    feature_indices   : mapeia nome da feature → índice na dimensão F
                        {'BORE_OIL_VOL': 0, 'delta_P': 2, ...}
    use_* flags       : ativa técnicas individuais
    n_bootstrap       : nº extra de janelas geradas por bootstrap
    sigma_pressure    : desvio (psi) para jitter de pressões
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    X = _ensure_ndarray(X_train)
    y = _ensure_ndarray(y_train)

    synth_X, synth_y = [], []          # listas de novos exemplos

    # ---- 1. escala multiplicativa (declínio/estrangulamento) --------------
    if use_scale and scale_factors:
        sx, sy = _scale_decline(X, y, feature_indices, scale_factors)
        synth_X.extend(sx); synth_y.extend(sy)

    # ---- 2. jitter em pressão ---------------------------------------------
    if use_jitter:
        X_jit = _jitter_pressure(X, feature_indices, sigma_pressure)
        synth_X.append(X_jit); synth_y.append(y.copy())

    # ---- 3. warp de declínio ----------------------------------------------
    if use_decline_warp:
        Xw, yw = _warp_decline(X, y, feature_indices)
        synth_X.append(Xw); synth_y.append(yw)

    # ---- 4. block bootstrap ------------------------------------------------
    if n_bootstrap > 0:
        bx, by = _block_bootstrap(X, y, n_bootstrap)
        synth_X.extend(bx); synth_y.extend(by)

    # ---------------- concatena e devolve no mesmo tipo --------------------
    X_aug = _concat(X, synth_X, X_train)
    y_aug = _concat(y, synth_y, y_train)
    return X_aug, y_aug



def augment_with_synthetic_samples(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    scales: List[float] = [1.5, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19]
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Augments the training data by scaling the amplitude of features and target,
    working generically with both pandas objects and NumPy arrays.

    Parameters:
      X_train: Original training features (pd.DataFrame or np.ndarray).
      y_train: Original target values (pd.Series or np.ndarray).
      scales: List of scale factors to use.

    Returns:
      A tuple (augmented_X_train, augmented_y_train)
    """
    
    # Lista para armazenar os dados originais e os escalados
    X_synthetic = [X_train]
    y_synthetic = [y_train]
    
    # Aplica a escala para cada fator e acumula os resultados
    for scale in scales:
        X_scaled = X_train / scale
        y_scaled = y_train / scale
        X_synthetic.append(X_scaled)
        y_synthetic.append(y_scaled)
    
    # Concatenando os dados de acordo com o tipo
    if isinstance(X_train, pd.DataFrame):
        X_train_aug = pd.concat(X_synthetic, axis=0).reset_index(drop=True)
    elif isinstance(X_train, np.ndarray):
        X_train_aug = np.concatenate(X_synthetic, axis=0)
    else:
        raise TypeError(f"Tipo não suportado para X_train: {type(X_train)}")
    
    if isinstance(y_train, pd.Series):
        y_train_aug = pd.concat(y_synthetic, axis=0).reset_index(drop=True)
    elif isinstance(y_train, np.ndarray):
        y_train_aug = np.concatenate(y_synthetic, axis=0)
    else:
        raise TypeError(f"Tipo não suportado para y_train: {type(y_train)}")
    
    
    return X_train_aug, y_train_aug



import numpy as np
import pickle
from typing import Tuple

def create_internal_validation_set_from_disk(
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    metadata_path: str,
    val_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cria um conjunto de treino e validação interna a partir de dados aumentados
    e metadados salvos em disco, garantindo a separação cronológica dentro de cada bloco.

    Args:
        X_aug: O array completo de features aumentado.
        y_aug: O array completo de rótulos aumentado.
        metadata_path: Caminho para o arquivo .pkl contendo os metadados dos blocos.
        val_frac: Fração de cada bloco a ser usada para validação (ex: 0.1 para 10%).

    Returns:
        Uma tupla contendo (X_train_fit, y_train_fit, X_val_internal, y_val_internal).
    """
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        end_indices = metadata["end_indices"]
    except (FileNotFoundError, IOError) as e:
        raise RuntimeError(f"Não foi possível carregar o arquivo de metadados de '{metadata_path}'. "
                         f"Certifique-se de que a função de aumento foi executada com a flag "
                         f"'save_metadata_path'. Erro original: {e}")

    # Listas para coletar as partes de cada novo conjunto
    X_train_fit_parts = []
    y_train_fit_parts = []
    X_val_internal_parts = []
    y_val_internal_parts = []

    start_index = 0
    for end_index in end_indices:
        # Pega o bloco de dados atual (original ou sintético)
        X_block = X_aug[start_index:end_index]
        y_block = y_aug[start_index:end_index]

        # Calcula o ponto de divisão dentro do bloco atual
        n_block_samples = len(y_block)
        if n_block_samples == 0:
            continue

        split_point = int(n_block_samples * (1 - val_frac))

        # Adiciona os primeiros (1 - val_frac)% ao conjunto de treino
        X_train_fit_parts.append(X_block[:split_point])
        y_train_fit_parts.append(y_block[:split_point])
        
        # Adiciona os últimos val_frac% ao conjunto de validação
        X_val_internal_parts.append(X_block[split_point:])
        y_val_internal_parts.append(y_block[split_point:])
        
        # Atualiza o índice de início para o próximo bloco
        start_index = end_index

    # Concatena todas as partes de treino e validação
    # Lida com o caso em que uma das listas pode estar vazia (se val_frac=0 ou 1)
    X_train_fit = np.concatenate(X_train_fit_parts, axis=0) if X_train_fit_parts else np.array([])
    y_train_fit = np.concatenate(y_train_fit_parts, axis=0) if y_train_fit_parts else np.array([])
    X_val_internal = np.concatenate(X_val_internal_parts, axis=0) if X_val_internal_parts else np.array([])
    y_val_internal = np.concatenate(y_val_internal_parts, axis=0) if y_val_internal_parts else np.array([])

    print(f"Divisão interna criada: "
          f"Treino_fit: {X_train_fit.shape[0]} amostras, "
          f"Val_internal: {X_val_internal.shape[0]} amostras.")

    return X_train_fit, y_train_fit, X_val_internal, y_val_internal

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any


def inspect_array_integrity(
    arr: np.ndarray, name: str
) -> Dict[str, Any]:
    """
    Inspects a numpy array for NaNs and Infs, returning counts and percentages.

    Args:
        arr: numpy array to inspect.
        name: label for the array.

    Returns:
        A dict with keys: name, total_elements, nan_count, nan_pct, posinf_count,
        neginf_count, inf_count, inf_pct.
    """
    total = arr.size
    nan_count = int(np.isnan(arr).sum())
    posinf_count = int(np.isposinf(arr).sum())
    neginf_count = int(np.isneginf(arr).sum())
    inf_count = posinf_count + neginf_count
    return {
        'dataset': name,
        'total_elements': total,
        'nan_count': nan_count,
        'nan_pct': nan_count / total * 100,
        'posinf_count': posinf_count,
        'neginf_count': neginf_count,
        'inf_count': inf_count,
        'inf_pct': inf_count / total * 100
    }


def create_integrity_report(
    X_before: np.ndarray,
    y_before: np.ndarray,
    X_after: np.ndarray,
    y_after: np.ndarray,
    metadata: Optional[Dict[str, List[int]]] = None,
    display: bool = True
) -> pd.DataFrame:
    """
    Gera um relatório de integridade dos dados antes e depois da data augmentation,
    identificando NaNs e Infs, e opcionalmente detalhando por blocos com metadados.

    Args:
        X_before: dados originais (X)
        y_before: rótulos originais (y)
        X_after: dados aumentados (X)
        y_after: rótulos aumentados (y)
        metadata: dicionário com 'end_indices' listando os limites cumulativos de cada bloco
        display: se True, imprime o DataFrame estilizado no notebook.

    Returns:
        DataFrame com relatório agregado (e por bloco, se metadata informado).
    """
    reports = []
    # global before/after
    reports.append(inspect_array_integrity(X_before, 'X_before'))
    reports.append(inspect_array_integrity(y_before, 'y_before'))
    reports.append(inspect_array_integrity(X_after, 'X_after'))
    reports.append(inspect_array_integrity(y_after, 'y_after'))

    df = pd.DataFrame(reports)

    # se tiver metadados, detalhar por bloco
    if metadata and 'end_indices' in metadata:
        ends = metadata['end_indices']
        starts = [0] + ends[:-1]
        for i, (s, e) in enumerate(zip(starts, ends)):
            X_block = X_after[s:e]
            y_block = y_after[s:e]
            reports_block = inspect_array_integrity(X_block, f'X_block_{i}')
            reports_block.update({'block': i})
            reports.append(reports_block)
            reports_block = inspect_array_integrity(y_block, f'y_block_{i}')
            reports_block.update({'block': i})
            reports.append(reports_block)
        df = pd.DataFrame(reports)
    
    # organizar colunas
    cols = ['dataset', 'block', 'total_elements', 'nan_count', 'nan_pct',
            'posinf_count', 'neginf_count', 'inf_count', 'inf_pct']
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    if display:
        try:
            from IPython.display import display as _disp
            _disp(df.style.format({
                'nan_pct': '{:.2f}%',
                'inf_pct': '{:.2f}%'
            }))
        except ImportError:
            print(df)
    return df



def augment_with_synthetic_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scales: List[float] = [1.5, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    sample_frac_per_scale: float = 0.95,
    random_state: int = 42,
    save_metadata_path: Optional[str] = "Meta_validation"  # Parâmetro opcional para salvar metadados
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aumenta os dados com versões escaladas, preservando a ordem cronológica,
    e opcionalmente salva os metadados dos blocos em disco.

    Args:
        X_train: Dados de treino originais (ordenados cronologicamente).
        y_train: Rótulos de treino originais.
        scales: Lista de fatores de escala para gerar dados sintéticos.
        sample_frac_per_scale: Fração de amostras a serem selecionadas de cada bloco.
        random_state: Semente para reprodutibilidade.
        save_metadata_path: Se fornecido, salva os índices de fim de bloco neste caminho.

    Returns:
        Um novo conjunto de treino e rótulos aumentados.
    """
    
    rng = np.random.RandomState(random_state)

    # Coleta as partes dos dados aumentados
    X_augmented_list = [X_train]
    y_augmented_list = [y_train]

    num_original_samples = X_train.shape[0]

    for scale in scales:
        # Gera o bloco sintético completo
        X_scaled = X_train / scale
        y_scaled = y_train / scale

        # Determina quantos índices selecionar
        num_to_sample = int(num_original_samples * sample_frac_per_scale)
        if num_to_sample == 0:
            continue # Pula se a fração for muito pequena e resultar em 0 amostras

        # 1. Seleciona `k` índices aleatórios do bloco sintético.
        chosen_indices = rng.choice(num_original_samples, size=num_to_sample, replace=False)

        # 2. ORDENA os índices selecionados para preservar a ordem cronológica.
        sorted_indices = np.sort(chosen_indices)

        # 3. Usa os índices ordenados para criar a subamostra.
        X_sub_ordered = X_scaled[sorted_indices]
        y_sub_ordered = y_scaled[sorted_indices]

        X_augmented_list.append(X_sub_ordered)
        y_augmented_list.append(y_sub_ordered)
        
    # --- Lógica de Metadados ---
    # É executada mesmo se não for salvar, pois é leve.
    block_sizes = [len(part) for part in X_augmented_list]
    end_indices = np.cumsum(block_sizes).tolist()
    
    # Salva os metadados se um caminho for fornecido
    if save_metadata_path:
        metadata = {"end_indices": end_indices}
        try:
            with open(save_metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        except IOError as e:
            print(f"Erro ao salvar metadados em {save_metadata_path}: {e}")

    # --- Concatenação Final ---
    X_final = np.concatenate(X_augmented_list, axis=0)
    y_final = np.concatenate(y_augmented_list, axis=0)

    return X_final, y_final

def create_synthetic_samples(X_train, y_train, scales=[2, 3, 5]):
    """
    Generate synthetic samples by scaling down the amplitude of X_train and y_train.
    
    Args:
        X_train (DataFrame): Original feature set.
        y_train (Series or array): Original target values.
        scales (list): List of scales to use for amplitude reduction.
        
    Returns:
        X_train_augmented (DataFrame): Original and synthetic feature set.
        y_train_augmented (Series): Original and synthetic target values.
    """
    # Initialize lists to hold the augmented data
    X_synthetic = [X_train]
    y_synthetic = [y_train]

    # Loop over each scale to create synthetic data
    for scale in scales:
        # Scale down X_train and y_train by the scale factor
        X_scaled = X_train / scale
        y_scaled = y_train / scale

        # Append the scaled data to the synthetic data lists
        X_synthetic.append(X_scaled)
        y_synthetic.append(y_scaled)

    # Concatenate original and synthetic data
    X_train_augmented = pd.concat(X_synthetic, axis=0).reset_index(drop=True)
    y_train_augmented = pd.concat(y_synthetic, axis=0).reset_index(drop=True)

    return X_train_augmented, y_train_augmented


def split_time_series(
    df: pd.DataFrame,
    test_size: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame em treino e teste, preservando ordem temporal.
    """
    n_test   = int(len(df) * test_size)
    df_train = df.iloc[:-n_test]
    df_test  = df.iloc[-n_test:]
    return df_train, df_test