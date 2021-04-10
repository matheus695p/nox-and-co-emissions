import pandas as pd
from src.open_module import ls

path = "data/"
files = ls(path)

data = pd.DataFrame()
for file in files:
    if "gt" in file:
        print("Leyendo", path+file)
        year = int(''.join(filter(str.isdigit, file)))
        datai = pd.read_csv(path+file)
        datai["YEAR"] = str(year)
        data = pd.concat([data, datai], axis=0)

# guardar data
data.reset_index(drop=True, inplace=True)
data.to_csv(path+"raw_data.csv", index=False)
