import numpy as np


def add_lagged_variables(df, columns, nr_of_lags=1):
    """
    Agregar variables pasadas del dataframe con el que se esta trabajando
    Parameters
    ----------
    df_input : df
        dataframe a operar.
    nr_of_lags : int
        número de frames de la evolución que quirees ir hacia atrás.
    columns : list
        lista de columas a las cuales agregar variables lagged.
    Returns
    -------
    df : df
        dataframe con las variables agregadas.
    """
    for i in range(1, nr_of_lags+1):
        for col in columns:
            lagged_column = col + f'_lagged_{i}'
            df[lagged_column] = df[col].shift(i)
    return df


def log_features(df, columns):
    """
    Agregar logaritmo de las variables
    Parameters
    ----------
    df : dataframe
        dataframe a operar.
    nr_of_lags : int
        número de frames de la evolución que quirees ir hacia atrás.
    columns : list
        lista de columas a las cuales agregar variables lagged.
    Returns
    -------
    df : df
        dataframe con las variables agregadas con logaritmo.
    """
    for col in columns:
        log_column = 'log_' + col
        df[log_column] = df[col].apply(lambda x: np.log(x))
    return df


def log_lagged_variables(df, columns, nr_of_lags=1):
    """
    Agregar variables aplicando logaritmo en ellas
    Parameters
    ----------
    df : dataframe
        dataframe a operar.
    nr_of_lags : int
        número de frames de la evolución que quirees ir hacia atrás.
    columns : list
        lista de columas a las cuales agregar variables lagged.
    Returns
    -------
    df : df
        dataframe con las variables agregadas con logaritmo.
    """
    for i in range(1, nr_of_lags+1):
        for col in columns:
            lagged_column = col + f'_log_lagged_{i}'
            df[lagged_column] = df[col].shift(i)
            df[lagged_column] = df[lagged_column].apply(lambda x: np.log(x))
    return df


def log_transform(number):
    """
    Aplicar logaritmo en las variables
    Parameters
    ----------
    number : float
        aplicar logaritmos.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.log(number)


def downcast_dtypes(df):
    """
    Función super util para bajar la cantidad de operaciones flotante que
    se van a realizar en el proceso de entrenamiento de la red
    Parameters
    ----------
    df : dataframe
        df a disminuir la cantidad de operaciones flotantes.
    Returns
    -------
    df : dataframe
        dataframe con los tipos int16 y float32 como formato número
    """
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def lowwer_rename(df):
    """
    Renombrar nombres de las columnas
    Parameters
    ----------
    df : dataframe
        dataframe.
    Returns
    -------
    df : dataframe
        dataframe con los nombres en minusculas.
    """
    for col in df.columns:
        new_col = col.lower()
        print("Renombrando columna:", col, "-->", new_col)
        df.rename(columns={col: new_col}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def selection_by_correlation(dataset, threshold=0.5):
    """
    Selecciona solo una de las columnas altamente correlacionadas y elimina
    la otra

    Parameters
    ----------
    dataset : dataframe
        dataset sin la variable objectivo.
    threshold : float
        modulo del valor threshold de correlación pearson.
    Returns
    -------
    dataset : dataframe
        dataset con las columnas eliminadas.

    """
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            cond1 = (corr_matrix.iloc[i, j] >= threshold)
            if cond1 and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]
    dataset = dataset.reset_index(drop=True)
    return dataset


def nn_preparation(df, target_col):
    """
    Hacer la preparación de la red neuronal
    Parameters
    ----------
    df : datarame
        dataframe con todas las variables.
    target_col : string or list
        nombre de la/as columna/as target/s.
    Returns
    -------
    x : array
        x en numpy.
    y : array
        target en numpy.
    """
    if len(target_col) == 1:
        y = df[[target_col]]
    else:
        y = df[target_col]
    x = df.drop(columns=target_col)
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y
