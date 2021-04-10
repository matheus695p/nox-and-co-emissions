import warnings
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.config_nn import arguments_parser
from src.open_module import ls
from src.clean_module import try_create_folder
from src.visualizations import (training_history, plot_multiple_xy_results)
from src.preprocessing_module import nn_preparation
from src.evaluation_module import (get_model_summary, mae_evaluation)
warnings.filterwarnings("ignore")

# parser de argumentos
args = arguments_parser()
path = "data/featured/"
files = ls(path)

# indice de la arquitectura probada
indice = str(4)
# directorio de resultados
folder_results = "nn_architectures"

for file in files:
    print("Ejecutando para el dataset:", file)
    df = pd.read_csv(path+file)
    df["year"] = df["year"].apply(int)
    # datos de training
    train_df = df[(df["year"] >= 2011) & (df["year"] <= 2013)]
    train_df.drop(columns="year", inplace=True)

    # datos de testing
    test_df = df[df["year"] >= 2014]
    test_df.drop(columns="year", inplace=True)

    # target col
    target_cols = ["co", "nox"]

    # separar los conjuntos de datos
    x_train, y_train = nn_preparation(train_df, target_cols)
    x_test, y_test = nn_preparation(test_df, target_cols)

    # Normalizar datos X
    sc = MinMaxScaler(feature_range=(0, 1))
    # training
    x_train = sc.fit_transform(x_train)
    # testing
    x_test = sc.transform(x_test)

    # Normalizar datos Y
    # scy = MinMaxScaler(feature_range=(0, 1))
    # # training
    # y_train = scy.fit_transform(y_train)
    # # testing
    # y_test = scy.transform(y_test)

    # modelo
    nn = tf.keras.Sequential()

    # index 4
    nn.add(tf.keras.layers.Dense(
        2048, input_dim=x_train.shape[1], activation='relu'))
    nn.add(tf.keras.layers.BatchNormalization())
    nn.add(tf.keras.layers.Dropout(0.3))
    nn.add(tf.keras.layers.Dense(1024, activation='relu'))
    nn.add(tf.keras.layers.BatchNormalization())
    nn.add(tf.keras.layers.Dropout(0.3))
    nn.add(tf.keras.layers.Dense(1024, activation='relu'))
    nn.add(tf.keras.layers.BatchNormalization())
    nn.add(tf.keras.layers.Dropout(0.3))
    nn.add(tf.keras.layers.Dense(1024, activation='relu'))
    nn.add(tf.keras.layers.BatchNormalization())
    nn.add(tf.keras.layers.Dense(y_train.shape[1], activation='linear'))

    # 3 index
    # nn.add(tf.keras.layers.Dense(
    #     2048, input_dim=x_train.shape[1], activation='relu'))
    # nn.add(tf.keras.layers.BatchNormalization())
    # nn.add(tf.keras.layers.Dropout(0.2))
    # nn.add(tf.keras.layers.Dense(512, activation='relu'))
    # nn.add(tf.keras.layers.BatchNormalization())
    # nn.add(tf.keras.layers.Dense(y_train.shape[1], activation='linear'))

    # index 5
    # nn.add(tf.keras.layers.Dense(
    #     4096, input_dim=x_train.shape[1], activation='relu'))
    # nn.add(tf.keras.layers.BatchNormalization())
    # nn.add(tf.keras.layers.Dropout(0.2))
    # nn.add(tf.keras.layers.Dense(y_train.shape[1], activation='linear'))

    # arquitectura usada
    nn.summary()

    nn.compile(loss='mean_squared_error',
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

    # desnormalizar
    # predictions = scy.inverse_transform(predictions)
    # y_test = scy.inverse_transform(y_test)

    # gráficar los resultados
    folder_name = file.replace(".csv", "").replace("featured", "nn_corr")
    # visualizaciones
    plot_multiple_xy_results(predictions, y_test, target_cols, indice,
                             folder_name=folder_name)
    # arquitectura usada en string
    architecture = get_model_summary(nn)
    print(architecture)

    # dataframe de salida
    output = mae_evaluation(predictions, y_test, target_cols, nn,
                            folder_name=folder_results,
                            filename=f"{indice}-{folder_name}")
    print(output)
    # guardar la arquitectura
    path_nn = "results/models"
    try_create_folder(path_nn)
    nn.save(path_nn+f"/{indice}_model.h5")
