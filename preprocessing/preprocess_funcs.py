import pandas as pd
import numpy as np
from scipy.stats import zscore
from itertools import combinations


def remove_unused(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Remove colunas que nao serao usadas.
    
    :param df: Dataframe alvo da transformacao
    :type df: pd.DataFrame
    :return: Dataframe transformado
    :rtype: pd.DataFrame
    """
    df_cp = df.copy()
    df_cp = df.drop(cols, axis=1)
    return df_cp
    

def angle_to_coord(row):
    alpha = row['alpha']
    delta = row['delta']
    a = np.cos(alpha) * np.cos(delta)
    b = np.sin(alpha) * np.cos(delta)
    c = np.sin(delta)
    return a, b, c


def spherical_to_castesian(df: pd.DataFrame, labels: list) -> pd.DataFrame:
    """
    Converte grandezas esfericas em grandezas de um plano cartesiano.
    
    :param df: Dataframe alvo da transformacao
    :type df: pd.DataFrame
    :return: Dataframe transformado
    :rtype: pd.DataFrame
    """
    
    df_unified = df.copy()
    df_unified['coords'] = df.apply(angle_to_coord, axis=1)
    df_split = df_unified.copy()
    df_split[labels] = pd.DataFrame(df_unified['coords'].tolist(), index=df.index)
    df_split = df_split.drop(['coords'], axis=1)
    return df_split


def remove_outliers(df: pd.DataFrame, features: list, threshold: int = 3) -> pd.DataFrame:
    """
    Remove outliers de um conjunto definido de features caso seu z-score seja superior a algum limiar.
    
    :param df: Dataframe alvo da transformacao
    :type df: pd.DataFrame
    :param features: 
    :return: Dataframe transformado
    :rtype: pd.DataFrame
    """
    df_cp = df.copy()     
    for feature in features:
        outliers = None
        z_scores = zscore(df[feature])
        outliers = df[(z_scores > threshold) | (z_scores < -threshold)]
        rem_index = outliers.index
        df_cp = df.drop(rem_index, axis=0)
    return df_cp


def one_hot_encoding(df: pd.DataFrame, categorical_cols: list, drop_first=False) -> pd.DataFrame:
    """
    Executa o One Hot Encoding em cada coluna fornecida de um DataFrame.

    :param df: DataFrame alvo das transformacoes
    :df type: pd.DataFrame
    :param cols: Lista de colunas que devem ser transformadas
    :cols type: list
    :return: DataFrame transformado
    :rtype: pd.Dataframe
    """

    missing_cols = [col for col in categorical_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas categóricas ausentes no DataFrame: {missing_cols}")
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df_encoded


def save_transformed_data(df: pd.DataFrame, transformations: dict, filename: str):
    """
    
    """

    transformed_df = df.copy()
    for transformation, params in transformations:
        transformed_df = transformation(transformed_df, **params)
    transformed_df.to_csv(filename, index=True)
    print(f"Arquivo salvo: {filename}")



def preprocess_data(df: pd.DataFrame, transformations: dict, filename: str):
    """
    
    """
    
    transformation_names = list(transformations.keys())
    for r in range(1, len(transformation_names) + 1):
        for combo in combinations(transformation_names, r):
            filename_combo = f"{filename}_{'_'.join(combo)}.csv"
            transformation_funcs = [(transformations[name][0], transformations[name][1]) for name in combo]
            save_transformed_data(df, transformation_funcs, filename_combo)