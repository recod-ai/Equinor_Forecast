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
        Se houver múltiplos poços, itera sobre eles e retorna um dicionário:
            {nome_do_poço: DataFrame}
        Caso contrário, retorna um único DataFrame.
        """
        if self.wells:
            return {well: self._load_well(well) for well in self.wells}
        else:
            return self._load_well(None)

    def _load_well(self, well: Optional[str]) -> pd.DataFrame:
        raise NotImplementedError("O método _load_well() deve ser implementado na subclasse.")


    def _add_time_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona a coluna 'Tempo_Inicio_Prod' iniciada em 1 e a coloca como primeira coluna."""
        # Generate a 1-based sequence of integers
        time_steps = np.arange(1, len(df) + 1)
        # Insert it at position 0
        df.insert(0, 'Tempo_Inicio_Prod', time_steps)
        return df




class VolveDataLoader(BaseDataLoader):
    def _load_well(self, well: Optional[str]) -> pd.DataFrame:
        # Carrega e prepara os dados para o poço
        df_prepared = load_and_prepare_data(self.data_path, well)
        df_features = engineer_features(
            df_prepared,
            cum_sum=self.cum_sum,
            well=well
        )
        df_features = treat_dataframe(df_features, df_features.columns)
        
        return df_features


class UnisimDataLoader(BaseDataLoader):
    def _load_well(self, well: Optional[str]) -> pd.DataFrame:
        df = load_data(self.data_path)
        if df is None:
            return pd.DataFrame()
        # df = preprocess_data(df)
        df = filter_data_by_well(df, well)
        
        # Mapeia o nome da série para a coluna correta
        variable_column = self.serie_name
        
        if self.remove_zeros:
            df = df[df[variable_column] != 0].dropna(subset=[variable_column])
        
        # Plota a série original
        # plot_time_series(df[variable_column], self.serie_name, well)
        
        if self.cum_sum:
            for variable in ['QLOB', 'QWOB', 'QOOB', 'QGOB', 'PWFO']:
                df[variable] = df[variable].cumsum()
            plot_time_series(df[variable_column], f"Cumulative {self.serie_name}", well)
        
        df = treat_dataframe(df, ['QOOB'])
        df = df[[col for col in df.columns if col != variable_column] + [variable_column]]
        df = self._add_time_column(df)

        # print(df.head())
        # --------------------------- PLOT COM UNIDADES ---------------------------
        # plot_by_well_advanced(
        #     df,
        #     columns=['QLOB', 'QWOB', 'QOOB', 'QGOB', 'PWFO'],
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
    Reads UNISIM-IV CSVs named by well, e.g. Well_P13_UNISIM-IV.csv,
    applies feature engineering and normalization.
    """
    def _load_well(self, well: Optional[str]) -> pd.DataFrame:
        # 1) Resolve the template path
        path_template = str(self.data_path)
        if "{well}" in path_template:
            csv_file = Path(path_template.format(well=well))
        else:
            base = Path(self.data_path)
            if base.is_dir():
                csv_file = base / f"Well_{well}_UNISIM-IV.csv"
            else:
                csv_file = base

        logging.info("Attempting to load UNISIM-IV data for well '%s' from %s", well, csv_file)

        # 2) Check existence
        if not csv_file.exists():
            logging.warning("CSV file not found for well '%s': %s", well, csv_file)
            raise FileNotFoundError(f"File not found: {csv_file}")

        try:
            # 3) Read CSV
            df = pd.read_csv(csv_file)
            logging.info("Successfully read CSV: %d rows, %d columns", df.shape[0], df.shape[1])

            # 4) Validate 'Day' column
            if "Day" not in df.columns:
                logging.error("'Day' column missing in %s. Available columns: %s",
                             csv_file.name, df.columns.tolist())
                raise KeyError(f"'Day' column not found in {csv_file.name}")

            # 5) Rename and feature-engineer
            df = df.rename(columns={"Day": "Tempo_Inicio_Prod"})
            logging.info("Renamed 'Day' to 'Tempo_Inicio_Prod'")

            df = engineer_features_unisim(
                df,
                p_reservoir_kpa=INITIAL_PRESSURE["UNISIM-IV-2024"],
                well = well
            )
            logging.info("Added UNISIM-IV features (delta_P, PI, etc.)")

            df = normalize_unisim_iv(df)
            logging.info("Normalized DataFrame to canonical feature set")
            log_invalid_values(df)

            # plot_by_well_advanced(
            #         df,
            #         columns=[self.serie_name],
            #         well=well,     
            #     )

            return df

        except Exception:
            logging.exception("Error processing UNISIM-IV CSV for well '%s'", well)
            raise






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
        
        
        

def load_data(file_path, delimiter='\t'):
    """
    Loads data from a CSV file into a DataFrame, handling possible parsing issues.

    Parameters:
        file_path (str): The path to the CSV file.
        delimiter (str): The delimiter used in the CSV file (default is tab).

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the data cannot be properly parsed.
    """
    try:
        # Attempt to read the CSV file with the specified delimiter
        data = pd.read_csv(file_path, delimiter=delimiter)
    except FileNotFoundError:
        print(f"Error: The specified file '{file_path}' was not found.")
        raise
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the file '{file_path}'. Check the delimiter and file format.")
        raise

    # Define the expected columns
    expected_columns = ['WELL', 'DAY', 'QLOB', 'QWOB', 'QOOB', 'QGOB', 'PWFO']

    # Check if the data was parsed correctly
    if not all(col in data.columns for col in expected_columns):
        # Attempt to split the first column if data is incorrectly parsed
        first_col = data.columns[0]
        if data[first_col].str.contains('\t').any():
            # Split the first column into multiple columns
            split_data = data[first_col].str.split('\t', expand=True)
            split_data.columns = expected_columns
            data = split_data
        else:
            raise ValueError("Data format is incorrect and cannot be parsed.")

    # Convert columns to appropriate data types
    numeric_columns = ['DAY', 'QLOB', 'QWOB', 'QOOB', 'QGOB', 'PWFO']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return data


def filter_data_by_well(df, well_name):
    """
    Filters the DataFrame to include data only for the specified well.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        well_name (str): The name of the well to filter by.

    Returns:
        pd.DataFrame: The filtered DataFrame containing data for the specified well.

    Raises:
        ValueError: If the specified well name is not found in the DataFrame.
    """
    # Check if the 'WELL' column exists
    if 'WELL' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'WELL' column.")

    # Filter the DataFrame for the specified well
    filtered_df = df[df['WELL'] == well_name].reset_index(drop=True)

    if filtered_df.empty:
        raise ValueError(f"No data found for well '{well_name}'.")

    return filtered_df


def treat_dataframe(df, columns):
    """
    Processes a DataFrame by selecting specific columns, counting NaNs and infinite values, 
    and imputing NaNs with column mean values.

    Args:
        df (pandas.DataFrame): The DataFrame to process.

    Returns:
        pandas.DataFrame: The processed DataFrame with NaN values imputed.
    """
    
    # Select specific columns
    df = df[columns]
    
    # Identify columns with NaN values
    cols_with_nans = [col for col in df.columns if df[col].isna().any()]

    # Function to count NaNs in a column
    def count_nans_per_col(df, col_name):
        return df[col_name].isna().sum()

    # Function to count infinite values in a column
    def count_infs_per_col(df, col_name):
        return np.isinf(df[col_name]).sum()

    # Print the count of NaNs and infinite values before imputation
    print('NaNs and infinite values before imputation')
    for col in cols_with_nans:
        print(f'Col: {col} NaN: {count_nans_per_col(df, col)}')
        print(f'Col: {col} InF: {count_infs_per_col(df, col)}')
    
    # Instantiate SimpleImputer with strategy to fill NaN values with mean
    imputer = SimpleImputer(strategy='mean')

    # Fit the imputer to your data and transform it
    df_imputed = imputer.fit_transform(df)

    # Transform the numpy array back into a DataFrame
    df = pd.DataFrame(df_imputed, columns=columns)
    
    # Print the count of NaNs and infinite values after imputation
    print('NaNs and infinite values after imputation')
    for col in cols_with_nans:
        print(f'Col: {col} NaN: {count_nans_per_col(df, col)}')
        print(f'Col: {col} InF: {count_infs_per_col(df, col)}')

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
    """
    Carrega dados do arquivo CSV.

    Parâmetros:
        file_path (str): O caminho para o arquivo CSV.

    Retorna:
        pd.DataFrame: O DataFrame carregado, ou None se o carregamento falhar.
    """
    import pandas as pd
    try:
        df = pd.read_csv(file_path, parse_dates=['utc_timestamp'], index_col='utc_timestamp')
        return df
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None
                                                  
def engineer_features(
    df: pd.DataFrame,
    cum_sum: bool = False,
    well: str = None
) -> pd.DataFrame:
    """
    Feature engineering para dados do campo Volve
    – converte unidades (bar → psi, m³/d → stb/d),
    – calcula ΔP em psi, PI coerente (stb d⁻¹ psi⁻¹),
    – acrescenta Tempo_Inicio_Prod e Taxa_Declinio.

    Retorna o DataFrame completo ou, se `cum_sum=True`,
    apenas as colunas acumuladas que o modelo de curva total usa.
    """

    # ----------------------------- [SANITY CHECKS DE UNIDADE] -----------------------------

    # Verifica se as pressões já estão em psi (valores típicos > 3000 psi)
    if df["AVG_DOWNHOLE_PRESSURE"].median() > 1000:
        logging.warning("Pressure appears to already be in psi. Expected: bar.")

    if df["AVG_WHP_P"].median() > 1000:
        logging.warning("Wellhead pressure appears to already be in psi. Expected: bar.")

    # Verifica se o volume de óleo já está em stb/d
    if df["BORE_OIL_VOL"].median() > 10000:
        logging.warning("Oil volume already appears to be in stb/d. Expected: m³/d.")

    # Verifica se o volume de gás já está em scf/d
    if df["BORE_GAS_VOL"].median() > 1000000:
        logging.warning("Gas volume already appears to be in stb/d. Expected: m³/d.")

    df = df.copy()

    # df["BORE_GAS_VOL"]  = np.log1p(df["BORE_GAS_VOL"])

    if not cum_sum:
        # --------------------------------------------------- 1. Conversões
        df["BORE_OIL_VOL"]  = df["BORE_OIL_VOL"].apply(m3d2stbd)      # stb/d
        df["BORE_GAS_VOL"]  = df["BORE_GAS_VOL"].apply(m3d2scfd)      # scf/d
        df["BORE_WAT_VOL"]  = df["BORE_WAT_VOL"].apply(m3d2stbd)
        df["AVG_DOWNHOLE_PRESSURE"] = df["AVG_DOWNHOLE_PRESSURE"].apply(bar2psi)  # psi
        df["AVG_WHP_P"]            = df["AVG_WHP_P"].apply(bar2psi)

    # --------------------------------------------------- 2. ΔP e PI
    df["delta_P"] = df["AVG_DOWNHOLE_PRESSURE"] - df["AVG_WHP_P"]
    df["delta_P"] = causal_impute(df["delta_P"]).replace(0, 1e-6)

    df["PI"] = df["BORE_OIL_VOL"] / df["delta_P"]       # stb/d / psi

    # --------------------------------------------------- 3. Tempo & Declínio
    df["Tempo_Inicio_Prod"] = np.arange(len(df), dtype=int) + 1

    q     = df["BORE_OIL_VOL"]
    prev  = q.shift(1).where(q.shift(1) > 0,
                             q.rolling(10, 1).mean().shift(1))
    prev  = prev.fillna(1e-6).clip(lower=1e-6)
    df["Taxa_Declinio"] = -np.log((q / prev).clip(lower=1e-6))

    # --------------------------------------------------- 4. Eficiência de choke
    df["CE"] = df["BORE_OIL_VOL"] / df["AVG_CHOKE_SIZE_P"].replace(0, np.nan)

    # --------------------------------------------------- 5. Acumulados opcionais
    if cum_sum:
        df["BORE_OIL_VOL"]          = df["BORE_OIL_VOL"].cumsum()
        df["BORE_GAS_VOL"]          = df["BORE_GAS_VOL"].cumsum()
        df["BORE_WI_VOL_15_9_F_4"]  = df["BORE_WI_VOL_15_9_F_4"].cumsum()

        keep = ["Tempo_Inicio_Prod", "BORE_GAS_VOL", "ON_STREAM_HRS", "BORE_OIL_VOL"]
        df = df[keep].copy()

    # --------------------------- PLOT COM UNIDADES ---------------------------
    # plot_by_well_advanced(
    #     df,
    #     columns=["BORE_OIL_VOL", "AVG_DOWNHOLE_PRESSURE", "PI"],
    #     well=well,     # <-- aparece no título, se for fornecido
    # )

    return df


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


def engineer_features_unisim(
    df: pd.DataFrame,
    *, p_reservoir_kpa: float,
    decline_window: int = 10,
    eps: float = 1e-6,
    well = None
) -> pd.DataFrame:

        # --- Verificações de sanidade / unidade de entrada ---
    # 1. Checar se já está em psi (valores tipicamente entre 500 e 6000 psi)
    if df["Well Bottom-hole Pressure"].median() < 1000:
        logging.info("Pressão em 'Well Bottom-hole Pressure' parece já estar em psi. A função espera kPa.")

    # 2. Checar se taxa de óleo já está em stb/d (valores geralmente altos)
    if df["Oil Rate SC"].median() > 10000:
        logging.info("Taxa de óleo em 'Oil Rate SC' parece já estar em stb/d. A função espera m³/d.")

    
    df = df.copy().loc[df["Oil Rate SC"] > 0]

    # 1) conversões coerentes ------------------------------------------
    df["Gas Rate SC"]  = np.log1p(df["Gas Rate SC"])
    df["Oil Rate SC"]  = df["Oil Rate SC"].apply(m3d2stbd)

    # *** CONVERTA EM-PLACE a coluna ORIGINAL ***
    df["Well Bottom-hole Pressure"] = kpa2psi(df["Well Bottom-hole Pressure"])
    # (não crie outra coluna!)

    # ΔP em psi ----------------------------------------------------------
    df["delta_P"] = kpa2psi(p_reservoir_kpa) - df["Well Bottom-hole Pressure"]
    df["delta_P"] = causal_impute(df["delta_P"]).replace(0, eps)

    df["PI"] = df["Oil Rate SC"] / df["delta_P"].replace(0, np.nan)

    # tempo & declínio
    df["Tempo_Inicio_Prod"] = np.arange(len(df), dtype=int) + 1
    q     = df["Oil Rate SC"]
    prev  = q.shift(1).where(q.shift(1) > 0,
                             q.rolling(decline_window, 1).mean().shift(1))
    prev  = prev.fillna(eps).clip(lower=eps)
    df["Taxa_Declinio"] = -np.log((q / prev).clip(lower=eps))

    # --------------------------- PLOT COM UNIDADES ---------------------------
    # plot_by_well_advanced(
    #     df,
    #     columns=["Oil Rate SC", "Well Bottom-hole Pressure", "PI"],
    #     well=well,     # <-- aparece no título, se for fornecido
    # )

    # descarta a primeira linha (prev artificial)
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

    return df_filtered
        






