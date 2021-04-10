import pandas as pd
from src.preprocessing_module import (lowwer_rename, log_features,
                                      add_lagged_variables,
                                      log_lagged_variables,
                                      selection_by_correlation)
from src.clean_module import try_create_folder

path = "data/"
df = pd.read_csv(path+"data.csv")
df = lowwer_rename(df)

# variable objetivo
target_cols = ["co", "nox"]

# columnas a extraer carácteristicas
columns = list(df.columns)
columns.remove("year")
for col in target_cols:
    columns.remove(col)

# agregar los log fatures
df = log_features(df, columns)

# agregar variables retrasadas
for i in range(1, 13):
    print("Agregando variable lag:", i, "...")
    df = add_lagged_variables(df, columns, i)
    print("Agregando variable logaritmica lag:", i, "...")
    df = log_lagged_variables(df, columns, i)

# limpiar los nulos
df.dropna(inplace=True)

# agregar un id a la fila para recuperar después
# df.reset_index(drop=False, inplace=True)
# df.rename(columns={"index": "id"}, inplace=True)

# features
features = list(df.columns)
features.remove("year")
for col in target_cols:
    features.remove(col)


# fechas

date = df[["year"]]
date.reset_index(drop=True, inplace=True)
# crear folder featured
try_create_folder(path+"featured")

# eliminar por correlación las varibles y guardar los dataset para probarlos
for thres in range(10, 19):
    thres = thres / 20
    print("featured dataset para un threshold de:", thres)
    # seleccionar solo las carácteristicas
    dataset = df[features]
    # eliminar por correlación
    dataset = selection_by_correlation(dataset, threshold=thres)
    dataset = pd.concat([dataset, date], axis=1)
    print("shape dataset:", dataset.shape)
    # guardar el dataset eliminado por corr
    thres = str(thres).replace(".", "_")
    dataset.to_csv(path+"featured/"+f"featured_{thres}.csv", index=False)


# guardar el featured
df.to_csv(path+"featured/featured_data.csv", index=False)
