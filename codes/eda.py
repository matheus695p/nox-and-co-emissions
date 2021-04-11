import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.config_cnn import arguments_parser
from src.visualizations import (pairplot_sns, pairgrid_plot, violinplot,
                                kernel_density_estimation)
warnings.filterwarnings("ignore")
sns.set(font_scale=1.5)
plt.style.use('dark_background')

# variables
args = arguments_parser()
path = "data/featured/"
file = "featured_data.csv"

# lectura del dataframe
df = pd.read_csv(path+file)
df["year"] = df["year"].apply(int)

# features
feature_cols = ['at', 'ap', 'ah', 'afdp', 'gtep', 'tit', 'tat', 'tey', 'cdp']
# target col
target_cols = ["co", "nox"]

# datos de training
train_df = df[(df["year"] >= 2011) & (df["year"] <= 2013)]
train_df.drop(columns="year", inplace=True)

# datos de testing
test_df = df[df["year"] >= 2014]
test_df.drop(columns="year", inplace=True)

# solo los targets
targets_train = train_df[target_cols]
targets_test = test_df[target_cols]

# solo los features
feature_train = train_df[feature_cols]
feature_test = test_df[feature_cols]

# PAIRPLOTS
# pairplot targets
pairplot_sns(targets_train, name="Entrenamiento")
pairplot_sns(targets_test, name="Testeo")

# pairplot features
pairplot_sns(feature_train, name="Train")
pairplot_sns(feature_test, name="Test")


# PAIRGRIDS
# pairgrids targets
pairgrid_plot(targets_train, name="Entrenamiento")
pairgrid_plot(targets_test, name="Testeo")

# DISTRIBUCIONES targets
for col in target_cols:
    print("Distribuciones de la columna:", col)
    # Violin plot de la distribución
    violinplot(targets_train, col, name="Entrenamiento")
    violinplot(targets_test, col, name="Testeo")
    # Kernel estimación
    kernel_density_estimation(targets_train, col, name="Entrenamiento")
    kernel_density_estimation(targets_test, col, name="Testeo")

# DISTRIBUCIONES features
for col in feature_cols:
    print("Distribuciones de la columna:", col)
    # Violin plot de la distribución
    # violinplot(feature_train, col, name="Entrenamiento")
    # violinplot(feature_test, col, name="Testeo")
    # Kernel estimación
    kernel_density_estimation(feature_train, col, name="Entrenamiento")
    kernel_density_estimation(feature_test, col, name="Testeo")
