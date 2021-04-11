import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.config_cnn import arguments_parser
from src.visualizations import (training_history, plot_multiple_xy_results)
from src.clean_module import try_create_folder
from src.evaluation_module import (get_model_summary, mae_evaluation)
from src.utils import get_prime_factors, input_shape
warnings.filterwarnings("ignore")

# variables
args = arguments_parser()
path = "data/featured/"
file = "featured_data.csv"

# indice de la arquitectura probada
indice = str(2)
# directorio de resultados
folder_results = "cnn_architectures"

# leear datos
df = pd.read_csv(path+file)
df["year"] = df["year"].apply(int)
# target col
target_cols = ["co", "nox"]

# datos de training
train_df = df[(df["year"] >= 2011) & (df["year"] <= 2013)]
train_df.drop(columns="year", inplace=True)
y_train = train_df[target_cols].to_numpy()
train_df = train_df.drop(columns=target_cols).to_numpy()

# datos de testing
test_df = df[df["year"] >= 2014]
test_df.drop(columns="year", inplace=True)
y_test = test_df[target_cols].to_numpy()
test_df = test_df.drop(columns=target_cols).to_numpy()

# numero de feautures
n_features = int(train_df.shape[1])

# encuntra la factorización prima de los valores
prime_factorization = get_prime_factors(n_features)

# encuentra el tamaño de la matriz para usar convoluciones 2D
shape = input_shape(prime_factorization,
                    natural_shape=train_df.shape[1])

# Normalizar datos X
sc = MinMaxScaler(feature_range=(0, 1))
# training
x_train = sc.fit_transform(train_df)
# testing
x_test = sc.transform(test_df)

# reshape en formato de matriz
x_train = np.reshape(x_train, (-1, shape[0], shape[1], 1))
x_test = np.reshape(x_test, (-1, shape[0], shape[1], 1))

# modelo
# cnn = tf.keras.Sequential()
# cnn.add(tf.keras.layers.Conv2D(32, input_shape=x_train.shape[1:],
#                                kernel_size=(10, 10), padding="same",
#                                activation="relu"))
# cnn.add(tf.keras.layers.Dropout(0.2))
# # cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
# cnn.add((tf.keras.layers.Conv2D(64, (5, 5), padding="same",
#                                 activation="relu")))
# cnn.add(tf.keras.layers.Dropout(0.2))
# cnn.add((tf.keras.layers.Conv2D(128, (3, 3), padding="same",
#                                 activation="relu")))
# cnn.add(tf.keras.layers.Dropout(0.2))
# # cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3)))
# cnn.add(tf.keras.layers.Flatten())
# cnn.add(tf.keras.layers.Dense(1024, activation='relu'))
# cnn.add(tf.keras.layers.BatchNormalization())
# cnn.add(tf.keras.layers.Dropout(0.2))
# cnn.add(tf.keras.layers.Dense(512, activation='relu'))
# cnn.add(tf.keras.layers.BatchNormalization())
# cnn.add(tf.keras.layers.Dense(y_train.shape[1], activation='linear'))
# cnn.summary()

cnn = tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv2D(16, input_shape=x_train.shape[1:],
                               kernel_size=(10, 10), padding="same",
                               activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.2))
# cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
cnn.add((tf.keras.layers.Conv2D(32, (5, 5), padding="same",
                                activation="relu")))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add((tf.keras.layers.Conv2D(64, (5, 5), padding="same",
                                activation="relu")))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add((tf.keras.layers.Conv2D(128, (5, 5), padding="same",
                                activation="relu")))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add((tf.keras.layers.Conv2D(256, (5, 5), padding="same",
                                activation="relu")))
cnn.add(tf.keras.layers.Dropout(0.2))

# cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3)))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(2048, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(1024, activation='relu'))
cnn.add(tf.keras.layers.Dense(y_train.shape[1], activation='linear'))
cnn.summary()


# compilar modelo
cnn.compile(loss='mean_squared_error',
            optimizer=args.optimizer)
# llamar callbacks de early stopping
tf.keras.callbacks.Callback()

# condición de parada
stop_condition = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=args.patience, verbose=1,
    min_delta=args.min_delta, restore_best_weights=True)

# bajar el learning_rate durante la optimización
learning_rate_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=args.lr_factor,
    patience=args.lr_patience,
    verbose=1,
    mode="auto",
    cooldown=0,
    min_lr=args.lr_min)

# cuales son los callbacks que se usaran
callbacks = [stop_condition, learning_rate_schedule]
# entrenar la red
history = cnn.fit(x_train, y_train,
                  validation_split=args.validation_size,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  shuffle=False,
                  verbose=1,
                  callbacks=callbacks)
# ver resutados de entrenamiento
training_history(history, model_name="cnn", filename="cnn")

# hacer predicciones en el test
predictions = cnn.predict(x_test)

# gráficar los resultados
folder_name = file.replace(".csv", "").replace("featured", "cnn_corr")
# visualizaciones
plot_multiple_xy_results(predictions, y_test, target_cols, indice,
                         folder_name=folder_name)
# arquitectura usada en string
architecture = get_model_summary(cnn)
print(architecture)

# dataframe de salida
output = mae_evaluation(predictions, y_test, target_cols, cnn,
                        folder_name=folder_results,
                        filename=f"{indice}-{folder_name}")
print(output)

path_nn = "results/models"
try_create_folder(path_nn)
cnn.save(path_nn+f"/{indice}_model_cnn.h5")
