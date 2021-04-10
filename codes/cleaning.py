import pandas as pd
from src.clean_module import (drop_spaces_data, replace_empty_nans,
                              convert_df_float)

path = "data/"
df = pd.read_csv(path+"raw_data.csv")

# sacar espacios en blanco
df = drop_spaces_data(df)
df = replace_empty_nans(df)
df = convert_df_float(df)

# guardar data limpia
df.to_csv(path+"data.csv", index=False)
