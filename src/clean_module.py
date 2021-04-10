import os
import numpy as np


def try_create_folder(path="images"):
    """
    Intentar crear carpeta
    Parameters
    ----------
    path : string
        direccion.
    """
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)


def drop_spaces_data(df):
    """
    sacar los espacios de columnas que podrián venir interferidas
    Parameters
    ----------
    df : dataframe
        input data
    column : string
        string sin espacios en sus columnas
    Returns
    -------
    """
    for column in df.columns:
        try:
            df[column] = df[column].str.lstrip()
            df[column] = df[column].str.rstrip()
        except Exception as e:
            print(e)
            pass
    return df


def make_empty_identifiable(value):
    """
    Parameters
    ----------
    value : int, string, etc
        valor con el que se trabaja.
    Returns
    -------
    nans en los vacios.
    """
    if value == "":
        output = np.nan
    else:
        output = value
    return output


def replace_empty_nans(df):
    """
    Parameters
    ----------
    df : int, string, etc
        valor con el que se trabaja.
    Returns
    -------
    nans en los vacios.
    """
    for col in df.columns:
        print("buscando vacios en:", col, "...")
        df[col] = df[col].apply(lambda x: make_empty_identifiable(x))
    return df


def convert_df_float(df):
    """
    Pasa por las columnas tratando de convertirlas a float64
    Parameters
    ----------
    df : dataframe
        df de trabajo.
    Returns
    -------
    df : dataframe
        df con las columnas númericas en float.
    """
    for col in df.columns:
        try:
            df[col] = df[col].apply(float)
        except Exception as e:
            print(e)
    df.reset_index(drop=True, inplace=True)
    return df
