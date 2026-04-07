#src/data/data_loading.py
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from data.data_preparation import  apply_kalman_filter
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict, Union
from evaluation.evaluation import plot_time_series
from forecast_pipeline.plotting import plot_by_well_advanced
import logging

from forecast_pipeline.config import kpa2psi, psi2kpa, bar2psi, m3d2stbd, m3d2scfd
from forecast_pipeline.config import VARIABLE_MAPPING, CANON_FEATURES, _UNISIM_IV_MAP, INITIAL_PRESSURE

# --- Unified logging helpers (plug-and-play) -----------------------------------
from forecast_pipeline.logging_utils import get_logger, phase

def _make_log(component: str, **ctx):
    """
    Create a contextual logger for data loading & feature engineering steps.
    Examples:
      _make_log("data.loading", dataset="VOLVE", well="15/9-F-11")
      _make_log("data.features", dataset="UNISIM-IV", well="P13")
    """
    return get_logger(component, context=ctx)




def normalize_unisim_iv(df: pd.DataFrame) -> pd.DataFrame:
    # 1) renomeia quem existe
    df = df.rename(columns=_UNISIM_IV_MAP)
    
    # 2) insere colunas que o CANON exige mas não vieram
    for col in CANON_FEATURES:
        if col not in df.columns:
            # se for CE, preencha com NaN; se for AVG_WHP_P, você pode copiar de AVG_DOWNHOLE_PRESSURE ou NaN
            df[col] = 0
    
    # 3) reordena estritamente na ordem canônica
    df = df[CANON_FEATURES]
    return df


class BaseDataLoader:
    def __init__(
        self,
        data_path: str,
        wells: Optional[List[str]] = None,
        serie_name: Optional[str] = None,
        cum_sum: bool = False,
        remove_zeros: bool = False,
        add_physical_features: bool = False
    ):
        self.data_path = data_path
        self.wells = wells
        self.serie_name = serie_name
        self.cum_sum = cum_sum
        self.remove_zeros = remove_zeros
        self.add_physical_features = add_physical_features

    def load(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        If multiple wells are provided, returns {well: DataFrame}.
        Otherwise, returns a single DataFrame.
        """
        log = _make_log("data.loading", dataset=self.__class__.__name__.replace("DataLoader", ""))
        
        # A FASE EXTERNA ("PACOTE") COMEÇA AQUI
        with phase(log, "Load data"):
            if self.wells:
                log.info("Starting multi-well load: %d wells.", len(self.wells))
                out = {}
                for w in self.wells:
                    # A fase interna por poço é gerenciada pelo _load_well
                    df_well = self._load_well(w)
                    out[w] = df_well
                    log.info("Loaded well '%s' with final shape %s.", w, getattr(df_well, "shape", "N/A"))
                log.info("Finished multi-well load.")
                return out
            else:
                log.info("Starting single-well (or generic) load.")
                df = self._load_well(None)
                log.info("Loaded dataset with final shape %s.", getattr(df, "shape", "N/A"))
                return df

    def _load_well(self, well: Optional[str]) -> pd.DataFrame:
        raise NotImplementedError("The _load_well() method must be implemented by subclasses.")

    def _add_time_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 'Tempo_Inicio_Prod' starting at 1 as the first column."""
        log = _make_log("data.loading", dataset=self.__class__.__name__)
        time_steps = np.arange(1, len(df) + 1)
        df.insert(0, "Tempo_Inicio_Prod", time_steps)
        log.debug("Inserted 'Tempo_Inicio_Prod' (len=%d).", len(time_steps))
        return df

class VolveDataLoader(BaseDataLoader):
    def _load_well(self, well: Optional[str]) -> pd.DataFrame:
        log = _make_log("data.loading", dataset="VOLVE", well=well or "N/A")
        with phase(log, "Prepare VOLVE well dataframe"):
            df_prepared = load_and_prepare_data(self.data_path, well)
            log.info("Prepared raw VOLVE data: shape %s.", getattr(df_prepared, "shape", None))
            df_features = engineer_features(df_prepared, cum_sum=self.cum_sum, well=well)
            df_features = treat_dataframe(df_features, df_features.columns)
            log.info("Engineered + treated VOLVE features: shape %s.", getattr(df_features, "shape", None))
            return df_features


class OpsdDataLoader(BaseDataLoader):
    def _load_well(self, well: Optional[str]) -> pd.DataFrame:
        df = load_df_opsd(self.data_path)
        if df is None:
            return pd.DataFrame()
        df = preprocess_data(df)
        df = df[20:]
   
        # if self.remove_zeros:
        #     df = df[df[self.serie_name] != 0].dropna(subset=[self.serie_name])

        print('df[self.serie_name].mean()', df[self.serie_name].mean())
        df = df[::24]  # Seleciona amostras a cada 48 linhas
        df[f"GB_GBN_{well}_generation_tax"] = df[self.serie_name]
        if self.cum_sum:
            df[self.serie_name] = df[self.serie_name].cumsum()
        
        df = df[[col for col in df.columns if col != self.serie_name] + [self.serie_name]]
        plot_time_series(df[f"GB_GBN_{well}_generation_tax"], 'GB_GBN', f"{well} Generation")
        df = self._add_time_column(df)
        df = treat_dataframe(df, [f"GB_GBN_{well}_generation_tax", f"Tempo_Inicio_Prod", self.serie_name])
        df = df[10:]
        plot_time_series(df[self.serie_name], 'GB_GBN', f"{well} Generation Cumulative")

        # plot_by_well_advanced(
        #     df,
        #     columns=[self.serie_name],
        #     well=well,     # <-- aparece no título, se for fornecido
        # )

        return df


class OpsdDataLoader(BaseDataLoader):
    def _load_well(self, well: Optional[str]) -> pd.DataFrame:
        log = _make_log("data.loading", dataset="OPSD", well=well or "N/A")
        with phase(log, "Prepare OPSD dataframe"):
            df = load_df_opsd(self.data_path)
            if df is None:
                log.warning("load_df_opsd returned None; returning empty DataFrame.")
                return pd.DataFrame()
            df = preprocess_data(df)
            df = df[20:]
            # replace print with log
            log.info("Series mean for '%s' before resample: %.6f", self.serie_name, df[self.serie_name].mean())
            df = df[::24]
            df[f"GB_GBN_{well}_generation_tax"] = df[self.serie_name]
            if self.cum_sum:
                df[self.serie_name] = df[self.serie_name].cumsum()
            df = df[[col for col in df.columns if col != self.serie_name] + [self.serie_name]]
            plot_time_series(df[f"GB_GBN_{well}_generation_tax"], "GB_GBN", f"{well} Generation")
            df = self._add_time_column(df)
            df = treat_dataframe(df, [f"GB_GBN_{well}_generation_tax", "Tempo_Inicio_Prod", self.serie_name])
            df = df[10:]
            plot_time_series(df[self.serie_name], "GB_GBN", f"{well} Generation Cumulative")
            log.info("OPSD final shape %s.", df.shape)
            
            # plot_by_well_advanced(
            #     df,
            #     columns=[self.serie_name],
            #     well=well,     # <-- aparece no título, se for fornecido
            # )
            return df


def log_invalid_values(df: pd.DataFrame, threshold: float = 1e30):
    """
    Varre o DataFrame em busca de inf, NaN ou valores > threshold,
    e emite um logging.info para cada ocorrência.
    Se encontrar algo, lança ValueError para interromper o pipeline.
    """
    arr = df.to_numpy()
    # máscaras de invalidez
    mask_inf = ~np.isfinite(arr)
    mask_big = np.abs(arr) > threshold

    if not (mask_inf.any() or mask_big.any()):
        logging.info("Nenhum valor inválido detectado em DataFrame.")
        return

    # reporta infinitos/NaN
    for (r, c) in zip(*np.where(mask_inf)):
        logging.info(f"Inf/NaN em linha {r}, coluna '{df.columns[c]}': {arr[r, c]}")
    # reporta valores muito grandes
    for (r, c) in zip(*np.where(mask_big)):
        logging.info(f"Valor muito grande em linha {r}, coluna '{df.columns[c]}': {arr[r, c]}")

    raise ValueError("Abortando: valores inválidos detectados no DataFrame.")

class Unisim_IV_DataLoader(BaseDataLoader):
    """
    Reads UNISIM-IV CSVs named by well, applies feature engineering and normalization.
    """
    def _load_well(self, well: Optional[str]) -> pd.DataFrame:
        log = _make_log("data.loading", dataset="UNISIM-IV", well=well or "N/A")

        # 1) Resolve the template path
        path_template = str(self.data_path)
        if "{well}" in path_template:
            csv_file = Path(path_template.format(well=well))
        else:
            base = Path(self.data_path)
            csv_file = (base / f"Well_{well}_UNISIM-IV.csv") if base.is_dir() else base

        log.info("Load from %s", csv_file)

        if not csv_file.exists():
            log.warning("CSV file not found: %s", csv_file)
            raise FileNotFoundError(f"File not found: {csv_file}")

        try:
            with phase(log, "Read & validate CSV"):
                df = pd.read_csv(csv_file)
                log.info("Read CSV ok: %d rows, %d cols.", df.shape[0], df.shape[1])
                if "Day" not in df.columns:
                    log.error("'Day' column missing. Available: %s", df.columns.tolist())
                    raise KeyError(f"'Day' column not found in {csv_file.name}")
                df = df.rename(columns={"Day": "Tempo_Inicio_Prod"})
                log.debug("Renamed 'Day' -> 'Tempo_Inicio_Prod'.")

            with phase(log, "Feature engineering (UNISIM-IV)"):
                df = engineer_features_unisim(
                    df,
                    p_reservoir_kpa=INITIAL_PRESSURE["UNISIM-IV-2024"],
                    well=well,
                )
                log.info("Added UNISIM-IV features (delta_P, PI, etc.).")

            with phase(log, "Normalize to canonical schema"):
                df = normalize_unisim_iv(df)
                log.info("Normalized to canonical feature set with %d columns.", df.shape[1])

            log_invalid_values(df)  # will raise if invalid; we want that

            log.info("UNISIM-IV final shape %s.", df.shape)

            # plot_by_well_advanced(
            #     df,
            #     columns=[self.serie_name],
            #     well=well,     
            # )
            
            return df

        except Exception:
            log.error("Error processing UNISIM-IV CSV.", exc_info=True)
            raise


class DataSource:
    def __init__(self, config: dict):
        self.name = config['name']
        self.wells = config.get('wells', None)
        self.model_path = config.get('model_path')
        self.features = config.get('features')
        self.target_column = config.get('target_column')
        self.filter_postprocess = config.get('filter_postprocess')
        self.load_params = config.get('load_params', {})
        # Tenta obter o nome da série a partir da própria configuração ou dos parâmetros de load
        self.serie_name = config.get('serie_name', self.load_params.get('serie_name'))

    def get_loader(self) -> BaseDataLoader:
        loader_kwargs = {
            "data_path": self.load_params.get("data_path"),
            "wells": self.wells,
            "serie_name": self.serie_name,
            "cum_sum": self.load_params.get("cum_sum", False),
            "remove_zeros": self.load_params.get("remove_zeros", False),
            "add_physical_features": self.load_params.get("add_physical_features", False),
        }
        return data_loader_factory(self.name, **loader_kwargs)
    
    
    

def data_loader_factory(source: str, **kwargs) -> BaseDataLoader:
    source = source.lower()
    if source == "volve":
        return VolveDataLoader(**kwargs)
    elif source == "unisim":
        return UnisimDataLoader(**kwargs)
    elif source == "opsd":
        return OpsdDataLoader(**kwargs)
    elif source == "unisim_iv":
        return Unisim_IV_DataLoader(**kwargs)
    else:
        raise ValueError(f"Fonte de dados '{source}' não reconhecida.")
        
        
        

def load_data(file_path, delimiter="\t"):
    """
    Loads data from a CSV file into a DataFrame, handling possible parsing issues.
    """
    log = _make_log("data.loading", dataset="UNISIM")
    try:
        data = pd.read_csv(file_path, delimiter=delimiter)
    except FileNotFoundError:
        log.error("The specified file '%s' was not found.", file_path)
        raise
    except pd.errors.ParserError:
        log.error("Failed to parse file '%s'. Check the delimiter and file format.", file_path)
        raise

    expected_columns = ["WELL", "DAY", "QLOB", "QWOB", "QOOB", "QGOB", "PWFO"]
    if not all(col in data.columns for col in expected_columns):
        first_col = data.columns[0]
        if data[first_col].astype(str).str.contains("\t").any():
            split_data = data[first_col].astype(str).str.split("\t", expand=True)
            split_data.columns = expected_columns
            data = split_data
            log.warning("Fixed malformed UNISIM file by splitting the first column with tabs.")
        else:
            raise ValueError("Data format is incorrect and cannot be parsed.")

    numeric_columns = ["DAY", "QLOB", "QWOB", "QOOB", "QGOB", "PWFO"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    log.info("Loaded UNISIM raw data with shape %s.", data.shape)
    return data



def filter_data_by_well(df, well_name):
    log = _make_log("data.loading", dataset="UNISIM", well=well_name or "N/A")
    if "WELL" not in df.columns:
        raise KeyError("The DataFrame does not contain a 'WELL' column.")
    filtered_df = df[df["WELL"] == well_name].reset_index(drop=True)
    if filtered_df.empty:
        log.error("No data found for well '%s'.", well_name)
        raise ValueError(f"No data found for well '{well_name}'.")
    log.info("Filtered to well '%s': shape %s.", well_name, filtered_df.shape)
    return filtered_df



def treat_dataframe(df, columns):
    """
    Select columns, report NaNs/inf, then impute NaNs with mean.
    """
    log = _make_log("data.dataframe")
    df = df[columns]

    cols_with_nans = [col for col in df.columns if df[col].isna().any()]
    def _count_nans(c): return int(df[c].isna().sum())
    def _count_infs(c): return int(np.isinf(df[c]).sum())

    if cols_with_nans:
        pre = {c: {"nan": _count_nans(c), "inf": _count_infs(c)} for c in cols_with_nans}
        log.info("NaNs/Inf before imputation: %s", pre)

    imputer = SimpleImputer(strategy="mean")
    df_imputed = imputer.fit_transform(df)
    df = pd.DataFrame(df_imputed, columns=columns)

    if cols_with_nans:
        post = {c: {"nan": _count_nans(c), "inf": _count_infs(c)} for c in cols_with_nans}
        log.info("NaNs/Inf after imputation: %s", post)

    return df




def preprocess_data(df):
    """
    Pré-processa o DataFrame para lidar com dados faltantes ou repetidos.

    Parâmetros:
        df (pd.DataFrame): O DataFrame a ser pré-processado.

    Retorna:
        pd.DataFrame: O DataFrame pré-processado.
    """
    # Remover índices duplicados
    df = df[~df.index.duplicated(keep='first')]
    # Preencher valores faltantes usando preenchimento para frente
    df = df.fillna(method='ffill')
    return df


def load_df_opsd(file_path):
    log = _make_log("data.loading", dataset="OPSD")
    try:
        df = pd.read_csv(file_path, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
        log.info("Loaded OPSD CSV: shape %s.", df.shape)
        return df
    except Exception as e:
        log.error("Failed to load OPSD data: %s", e)
        return None


                                                 
def causal_impute(series: pd.Series, limit: int = 3) -> pd.Series:
    """
    Forward-fill com limite + interpolação estritamente para trás.
    Não usa informação futura (causal).
    """
    s = (series
         .ffill(limit=limit)      # preenche até N valores à frente
         .interpolate(method="linear", limit_direction="forward")  # só pontos passados
    )
    return s


import pandas as pd
import numpy as np
from typing import List, Optional

def replace_physical_outliers(
    df: pd.DataFrame,
    columns_to_check: List[str],
    iqr_multiplier: float = 1.5,
    rolling_window_size: int = 7,
    log: Optional[object] = None # Aceita um objeto logger
) -> pd.DataFrame:
    """
    Identifica e substitui outliers em um DataFrame com base em uma média rolante causal.

    A substituição ocorre em duas etapas para cada coluna especificada:
    1. Identifica outliers como valores negativos ou valores extremos superiores
       (definidos pelo método IQR: valor > Q3 + multiplier * IQR).
    2. Substitui cada outlier identificado pela média dos 'rolling_window_size'
       valores anteriores válidos (média causal).

    Args:
        df: O DataFrame de entrada.
        columns_to_check: Lista de nomes de colunas para verificar.
        iqr_multiplier: Multiplicador para o intervalo IQR. Padrão é 1.5.
        rolling_window_size: Tamanho da janela para a média rolante. Padrão é 7.
        log: Um objeto logger opcional para registrar as substituições.

    Returns:
        Um novo DataFrame com os outliers substituídos.
    """
    df_processed = df.copy()
    total_replacements = 0

    if len(df_processed) == 0:
        return df_processed

    for col in columns_to_check:
        if col not in df_processed.columns:
            if log:
                log.warning(f"Coluna '{col}' para checagem de outlier não encontrada no DataFrame.")
            continue

        # --- Etapa 1: Identificar todos os outliers de uma vez ---
        # Valores negativos
        negative_mask = df_processed[col] < 0

        # Outliers extremos (IQR)
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + iqr_multiplier * IQR
        
        # Evita problemas se Q3 ou IQR forem NaN (coluna constante)
        if pd.notna(upper_bound):
            extreme_outlier_mask = df_processed[col] > upper_bound
        else:
            extreme_outlier_mask = pd.Series(False, index=df_processed.index)

        # Máscara combinada de todos os outliers para a coluna
        total_outlier_mask = negative_mask | extreme_outlier_mask
        num_outliers = total_outlier_mask.sum()

        if num_outliers == 0:
            continue

        if log:
            log.info(f"Encontrados {num_outliers} outliers na coluna '{col}' para substituição.")
        
        # --- Etapa 2: Substituir usando média rolante ---
        # Cria uma série de médias rolantes causais (usando apenas dados passados)
        # min_periods=1 garante que tenhamos um valor mesmo no início da série
        causal_rolling_mean = df_processed[col].rolling(
            window=rolling_window_size, min_periods=1
        ).mean()

        # A função 'where' mantém o valor original onde a condição é Verdadeira
        # e substitui pelo valor do segundo argumento onde é Falsa.
        # Condição: ~total_outlier_mask (ou seja, "onde NÃO é um outlier")
        df_processed[col] = df_processed[col].where(~total_outlier_mask, causal_rolling_mean)
        
        # Pode haver NaNs no início se os primeiros valores forem outliers.
        # Usamos backfill para preenchê-los com o primeiro valor válido subsequente.
        df_processed[col] = df_processed[col].bfill()

        total_replacements += num_outliers

    if log:
        log.info(f"Substituição de outliers concluída. Total de valores substituídos: {total_replacements}.")

    return df_processed


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter



def _causal_smooth_series(
    series: pd.Series,
    window_length: int,
    polyorder: int
) -> pd.Series:
    """
    Aplica um filtro Savitzky-Golay de forma causal (sem olhar para o futuro).

    Usa uma janela deslizante para garantir que a suavização em cada ponto
    use apenas dados passados e presentes.
    """
    # Garante que a janela seja ímpar e válida para o filtro
    if window_length % 2 == 0:
        window_length += 1
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    # A função a ser aplicada em cada janela deslizante
    def savgol_on_window(window):
        # O filtro precisa de um número mínimo de pontos para rodar
        if len(window) < window_length:
            return np.nan # Retorna NaN se a janela não estiver cheia
        # Aplica o filtro na janela e retorna APENAS o último ponto suavizado
        return savgol_filter(window, window_length, polyorder)[-1]

    # Aplica a função na janela deslizante
    # min_periods garante que a função só rode quando a janela estiver cheia
    smoothed_series = series.rolling(
        window=window_length, 
        min_periods=window_length
    ).apply(savgol_on_window, raw=True)
    
    # Para o início da série (onde tínhamos NaN), preenchemos com os valores originais
    # Isso evita perder o início da produção, que é um período crítico.
    smoothed_series.fillna(series, inplace=True)
    
    return smoothed_series


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth_and_plot_production(
    df: pd.DataFrame,
    column_name: str,
    window_length: int = 51,
    polyorder: int = 3
) -> pd.Series:
    """
    Suaviza uma série temporal de produção ruidosa usando um filtro Savitzky-Golay na curva acumulada.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados de produção.
        column_name (str): O nome da coluna com os dados de produção (ex: 'BORE_OIL_VOL').
        window_length (int): O tamanho da janela para o filtro Savgol. Deve ser um número ímpar.
                             Quanto maior, mais suave será a curva.
        polyorder (int): A ordem do polinômio usado no filtro. Deve ser menor que window_length.
                         Valores comuns são 2 ou 3.

    Returns:
        pd.Series: Uma nova série pandas com a produção suavizada.
    """
    # --- 1. Validação dos Parâmetros ---
    if column_name not in df.columns:
        raise ValueError(f"A coluna '{column_name}' não foi encontrada no DataFrame.")
    
    if window_length % 2 == 0:
        window_length += 1
        print(f"Aviso: window_length deve ser ímpar. Ajustando para {window_length}.")
        
    if polyorder >= window_length:
        raise ValueError("polyorder deve ser menor que window_length.")

    # --- 2. Cálculos Principais ---
    # Série de produção original (ruidosa)
    p_original = df[column_name]
    
    # Série acumulada original (baseada nos dados ruidosos)
    c_original = p_original.cumsum()
    
    # Aplica o filtro Savitzky-Golay na curva acumulada para suavizá-la
    c_suave = savgol_filter(c_original, window_length, polyorder)
    
    # Diferencia a curva acumulada suave para reconstruir a produção diária ideal
    # A primeira produção é igual à primeira acumulada.
    p_suave = np.diff(c_suave, prepend=c_suave[0])
    
    # Converte o resultado de volta para uma Série pandas com o índice correto
    p_suave_series = pd.Series(p_suave, index=p_original.index, name=f"{column_name}_smoothed")

    # --- 3. Plotagem dos Resultados ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Gráfico 1: Comparação da Produção Diária
    ax1.plot(p_original.index, p_original, 'o', markersize=3, alpha=0.5, label='Produção Original (Ruidosa)')
    ax1.plot(p_suave_series.index, p_suave_series, 'r-', linewidth=2.5, label='Produção Suavizada (Decaimento Ideal)')
    ax1.set_title('Comparativo da Produção Diária/Mensal', fontsize=14)
    ax1.set_ylabel('Volume de Óleo')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Gráfico 2: Verificação da Correspondência Acumulada
    ax2.plot(c_original.index, c_original, label='Acumulada Original (dos dados ruidosos)', linewidth=2)
    ax2.plot(c_original.index, c_suave, 'r--', label='Acumulada Suavizada (base para reconstrução)', linewidth=2.5)
    ax2.set_title('Verificação da Correspondência na Curva Acumulada', fontsize=14)
    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('Volume de Óleo Acumulado')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return p_suave_series


def engineer_features(
    df: pd.DataFrame,
    cum_sum: bool = False,
    well: str = None
) -> pd.DataFrame:
    log = _make_log("data.features", dataset="VOLVE", well=well or "N/A")

    # Sanity checks on units
    if df["AVG_DOWNHOLE_PRESSURE"].median() > 1000:
        log.warning("Downhole pressure seems to be already in psi (expected bar).")
    if df["AVG_WHP_P"].median() > 1000:
        log.warning("Wellhead pressure seems to be already in psi (expected bar).")
    if df["BORE_OIL_VOL"].median() > 10000:
        log.warning("Oil volume seems already stb/d (expected m³/d).")
    if df["BORE_GAS_VOL"].median() > 1000000:
        log.warning("Gas volume seems already scf/d (expected m³/d).")

    df = df.copy()

    if not cum_sum:
        df["BORE_OIL_VOL"]  = df["BORE_OIL_VOL"].apply(m3d2stbd)
        df["BORE_GAS_VOL"]  = df["BORE_GAS_VOL"].apply(m3d2scfd)
        df["BORE_WAT_VOL"]  = df["BORE_WAT_VOL"].apply(m3d2stbd)
        df["AVG_DOWNHOLE_PRESSURE"] = df["AVG_DOWNHOLE_PRESSURE"].apply(bar2psi)
        df["AVG_WHP_P"]             = df["AVG_WHP_P"].apply(bar2psi)
        log.info("Applied unit conversions (bar→psi, m³/d→stb|scf/d).")


    # Replace zeros in downhole pressure with mean+noise (causal-safe)
    col = "AVG_DOWNHOLE_PRESSURE"
    rng = np.random.default_rng(42)
    mean_val = df.loc[df[col] > 0, col].mean()
    mask_zero = df[col] == 0
    noise = rng.normal(0, mean_val * 0.01, size=mask_zero.sum())
    df.loc[mask_zero, col] = mean_val + noise
    if mask_zero.any():
        log.info("Replaced %d zeros in '%s' with mean±noise.", int(mask_zero.sum()), col)

    # if not cum_sum:
    #     # Defina aqui as colunas e seus parâmetros de suavização
    #     smoothing_config = {
    #         "BORE_OIL_VOL": {"window_length": 30, "polyorder": 2},
    #         "BORE_GAS_VOL": {"window_length": 30, "polyorder": 2},
    #         "AVG_DOWNHOLE_PRESSURE": {"window_length": 15, "polyorder": 2},
    #         "AVG_WHP_P": {"window_length": 15, "polyorder": 2},
    #     }
        
    #     log.info("Applying causal smoothing to specified features...")
    #     for col, params in smoothing_config.items():
    #         if col in df.columns:
    #             df[col] = _causal_smooth_series(
    #                 series=df[col],
    #                 window_length=params["window_length"],
    #                 polyorder=params["polyorder"]
    #             )
    #             log.info(f" > Feature '{col}' has been smoothed.")

    # ΔP, PI, time, decline, choke efficiency
    df["delta_P"] = df["AVG_DOWNHOLE_PRESSURE"] - df["AVG_WHP_P"]
    df["delta_P"] = causal_impute(df["delta_P"]).replace(0, 1e-6)
    df["PI"] = df["BORE_OIL_VOL"] / df["delta_P"]
    df["Tempo_Inicio_Prod"] = np.arange(len(df), dtype=int) + 1

    q = df["BORE_OIL_VOL"]
    prev = q.shift(1).where(q.shift(1) > 0, q.rolling(10, 1).mean().shift(1))
    prev = prev.fillna(1e-6).clip(lower=1e-6)
    df["Taxa_Declinio"] = -np.log((q / prev).clip(lower=1e-6))

    df["CE"] = df["BORE_OIL_VOL"] / df["AVG_CHOKE_SIZE_P"].replace(0, np.nan)

    if cum_sum:
        df["BORE_OIL_VOL"]         = df["BORE_OIL_VOL"].cumsum()
        df["BORE_GAS_VOL"]         = df["BORE_GAS_VOL"].cumsum()
        df["BORE_WI_VOL_15_9_F_4"] = df["BORE_WI_VOL_15_9_F_4"].cumsum()
        keep = ["Tempo_Inicio_Prod", "BORE_GAS_VOL", "ON_STREAM_HRS", "BORE_OIL_VOL"]
        df = df[keep].copy()
        log.info("Generated cumulative series; columns kept: %s.", keep)


    # plot_by_well_advanced(
    #     df,
    #     columns=["BORE_OIL_VOL", "AVG_DOWNHOLE_PRESSURE", "PI", "BORE_GAS_VOL"],
    #     well=well,     # <-- aparece no título, se for fornecido
    # )

    return df


def engineer_features_unisim(
    df: pd.DataFrame,
    *, p_reservoir_kpa: float,
    decline_window: int = 10,
    eps: float = 1e-6,
    well=None
) -> pd.DataFrame:
    log = _make_log("data.features", dataset="UNISIM-IV", well=well or "N/A")

    if df["Well Bottom-hole Pressure"].median() < 1000:
        log.info("'Well Bottom-hole Pressure' appears already in psi; expected kPa.")
    if df["Oil Rate SC"].median() > 10000:
        log.info("'Oil Rate SC' appears already in stb/d; expected m³/d.")

    df = df.copy().loc[df["Oil Rate SC"] > 0]
    df["Oil Rate SC"] = df["Oil Rate SC"].apply(m3d2stbd)
    df["Well Bottom-hole Pressure"] = kpa2psi(df["Well Bottom-hole Pressure"])

    df["delta_P"] = kpa2psi(p_reservoir_kpa) - df["Well Bottom-hole Pressure"]
    df["delta_P"] = causal_impute(df["delta_P"]).replace(0, eps)
    df["PI"] = df["Oil Rate SC"] / df["delta_P"].replace(0, np.nan)

    df["Tempo_Inicio_Prod"] = np.arange(len(df), dtype=int) + 1
    q = df["Oil Rate SC"]
    prev = q.shift(1).where(q.shift(1) > 0, q.rolling(decline_window, 1).mean().shift(1))
    prev = prev.fillna(eps).clip(lower=eps)
    df["Taxa_Declinio"] = -np.log((q / prev).clip(lower=eps))

    log.info("UNISIM-IV engineered features ready: shape %s.", df.shape)
    # plot_by_well_advanced(
    #     df,
    #     columns=["Oil Rate SC", "delta_P", "PI", "Gas Rate SC"],
    #     well=well,     # <-- aparece no título, se for fornecido
    # )
    return df.iloc[1:].reset_index(drop=True)


def load_and_prepare_data(file_path: str, well: str) -> pd.DataFrame:
    """
    Carrega os dados do CSV, filtra para o poço especificado e para o poço especial '15/9-F-4',
    realiza o merge com base na data de produção e preenche valores faltantes.

    Args:
        file_path (str): Caminho para o arquivo CSV.
        well (str): Nome do poço principal para filtrar os dados.

    Returns:
        pd.DataFrame: DataFrame preparado com os dados filtrados e alinhados por data.
    """
    # Carrega o CSV em um DataFrame
    df = pd.read_csv(file_path, engine='python', decimal=",")

    well = well.replace('/F', '-F')
    # Filtra para o poço principal
    df_principal = df.loc[df['NPD_WELL_BORE_NAME'] == well].copy()

    # Filtra para o poço especial '15/9-F-4' e seleciona 'DATEPRD' e 'BORE_WI_VOL'
    df_injection = df.loc[df['NPD_WELL_BORE_NAME'] == '15/9-F-4', ['DATEPRD', 'BORE_WI_VOL']].copy()

    # Realiza o merge com base em 'DATEPRD', mantendo todas as datas do poço principal
    df_merged = pd.merge(
        df_principal,
        df_injection,
        on='DATEPRD',
        how='left',
        suffixes=('', '_injection')
    )

    # Substitui valores NaN em 'BORE_WI_VOL_injection' por 0
    df_merged['BORE_WI_VOL_injection'] = df_merged['BORE_WI_VOL_injection'].fillna(0)

    # Renomeia a coluna para um nome mais descritivo
    df_merged.rename(columns={'BORE_WI_VOL_injection': 'BORE_WI_VOL_15_9_F_4'}, inplace=True)

    # Garante que 'BORE_WI_VOL_15_9_F_4' seja numérico
    df_merged['BORE_WI_VOL_15_9_F_4'] = pd.to_numeric(
        df_merged['BORE_WI_VOL_15_9_F_4'], errors='coerce'
    ).fillna(0)

    # Seleciona as colunas de interesse
    columns_to_keep = [
        'ON_STREAM_HRS', 'BORE_WAT_VOL', 'BORE_WI_VOL_15_9_F_4',
        'AVG_DOWNHOLE_PRESSURE', 'AVG_WHP_P',
        'AVG_CHOKE_SIZE_P', 'BORE_OIL_VOL', 'BORE_GAS_VOL'
    ]

    df_filtered = df_merged[columns_to_keep].copy()

    # Remove linhas onde 'BORE_OIL_VOL' é zero
    df_filtered = df_filtered[df_filtered['BORE_OIL_VOL'] != 0].reset_index(drop=True)

    # plot_by_well_advanced(
    #     df,
    #     columns=["BORE_OIL_VOL", "AVG_DOWNHOLE_PRESSURE", "BORE_GAS_VOL"],
    #     well=well,     # <-- aparece no título, se for fornecido
    # )

    return df_filtered





        






