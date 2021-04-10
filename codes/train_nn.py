import warnings
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.config import arguments_parser
from src.visualizations import training_history
from src.preprocessing_module import nn_preparation
from src.visualizations import plot_xy_results
warnings.filterwarnings("ignore")

# parser de argumentoss
args = arguments_parser()
path = "data/featured/"
df = pd.read_csv(path+"featured_data.csv")
df["year"] = df["year"].apply(int)
# datos de training
train_df = df[(df["year"] >= 2011) & (df["year"] <= 2013)]
train_df.drop(columns="year", inplace=True)

# datos de testing
test_df = df[df["year"] >= 2014]
test_df.drop(columns="year", inplace=True)

# target col
target_col = "tey"

# separar los conjuntos de datos
x_train, y_train = nn_preparation(train_df, target_col)
x_test, y_test = nn_preparation(test_df, target_col)

# Normalizar datos
sc = MinMaxScaler(feature_range=(0, 1))
# training
train_df = sc.fit_transform(train_df)
# testing
test_df = sc.transform(test_df)

# modelo
nn = tf.keras.Sequential()
nn.add(tf.keras.layers.Dense(
    2048, input_dim=x_train.shape[1], activation='relu'))
nn.add(tf.keras.layers.BatchNormalization())
nn.add(tf.keras.layers.Dropout(0.3))
nn.add(tf.keras.layers.Dense(1024, activation='relu'))
nn.add(tf.keras.layers.BatchNormalization())
nn.add(tf.keras.layers.Dropout(0.3))
nn.add(tf.keras.layers.Dense(512, activation='relu'))
nn.add(tf.keras.layers.BatchNormalization())
nn.add(tf.keras.layers.Dense(1, activation='linear'))
# arquitectura usada
nn.summary()

nn.compile(loss='mean_squared_error',
           optimizer=args.optimizer)
# llamar callbacks de early stopping
tf.keras.callbacks.Callback()
stop_condition = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  mode='min',
                                                  patience=args.patience,
                                                  verbose=1,
                                                  min_delta=args.min_delta,
                                                  restore_best_weights=True)


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
history = nn.fit(x_train, y_train,
                 validation_split=args.validation_size,
                 batch_size=args.batch_size,
                 epochs=args.epochs,
                 shuffle=False,
                 verbose=1,
                 callbacks=callbacks)

# ver resutados de entrenamiento
training_history(history, model_name="NN", filename="NN")


# hacer predicciones en el test
predictions = nn.predict(x_test)

# gráficar los resultados
plot_xy_results(predictions, y_test)
