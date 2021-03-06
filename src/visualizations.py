import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from src.clean_module import try_create_folder
sns.set(font_scale=1.5)
plt.style.use('dark_background')


def plot_instance_training(history, epocas_hacia_atras, model_name,
                           filename):
    """
    Sacar el historial de entrenamiento de epocas en partivular
    Parameters
    ----------
    history : object
        DESCRIPTION.
    epocas_hacia_atras : int
        epocas hacia atrás que queremos ver en el entrenamiento.
    model_name : string
        nombre del modelo.
    filename : string
        nombre del archivo.
    Returns
    -------
    bool
        gráficas de lo ocurrido durante el entrenamiento.
    """
    plt.style.use('dark_background')
    letter_size = 20
    # Hist training
    largo = len(history.history['loss'])
    x_labels = np.arange(largo-epocas_hacia_atras, largo)
    x_labels = list(x_labels)
    # Funciones de costo
    loss_training = history.history['loss'][-epocas_hacia_atras:]
    loss_validation = history.history['val_loss'][-epocas_hacia_atras:]
    # Figura
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax.plot(x_labels, loss_training, 'gold', linewidth=2)
    ax.plot(x_labels, loss_validation, 'r', linewidth=2)
    ax.set_xlabel('Epocas', fontname="Arial", fontsize=letter_size-5)
    ax.set_ylabel('Función de costos', fontname="Arial",
                  fontsize=letter_size-5)
    ax.set_title(f"{model_name}", fontname="Arial", fontsize=letter_size)
    ax.legend(['Entrenamiento', 'Validación'], loc='upper left',
              prop={'size': letter_size-5})
    # Tamaño de los ejes
    for tick in ax.get_xticklabels():
        tick.set_fontsize(letter_size-5)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(letter_size-5)
    plt.show()
    return fig


def training_history(history, model_name="NN", filename="NN"):
    """
    Según el historial de entrenamiento que hubo plotear el historial
    hacía atrás de las variables
    Parameters
    ----------
    history : list
        lista con errores de validación y training.
    model_name : string, optional
        nombre del modelo. The default is "Celdas LSTM".
    filename : string, optional
        nombre del archivo. The default is "LSTM".
    Returns
    -------
    None.
    """
    size_training = len(history.history['val_loss'])
    fig = plot_instance_training(history, size_training, model_name,
                                 filename + "_ultimas:" +
                                 str(size_training) + "epocas")

    fig = plot_instance_training(history, int(1.5 * size_training / 2),
                                 model_name,
                                 filename + "_ultimas:" +
                                 str(1.5 * size_training / 2) + "epocas")
    # guardar el resultado de entrenamiento de la lstm
    print(os.getcwd())
    try_create_folder("results")
    fig.savefig(f"results/{model_name}_training.png")

    fig = plot_instance_training(history, int(size_training / 2),
                                 model_name,
                                 filename + "_ultimas:" + str(
                                     size_training / 2) + "epocas")

    fig = plot_instance_training(history, int(size_training / 3), model_name,
                                 filename + "_ultimas:" +
                                 str(size_training / 3) + "epocas")
    fig = plot_instance_training(history, int(size_training / 4), model_name,
                                 filename + "_ultimas:" + str(
                                     size_training / 4) + "epocas")
    print(fig)


def plot_sequence(predictions, real, fechas, indice, folder_name="nn"):
    """
    Plot sequence de la secuecnia
    Parameters
    ----------
    predictions : array
        predicciones.
    real : array
        valores reales.
    fechas : array
        array de fechas.
    indice : TYPE
        indice de la columna.
    Returns
    -------
    plot de prediciones vs real.
    """
    letter_size = 20
    new_fechas = []
    for fecha in fechas:
        fecha = fecha[0:10]
        new_fechas.append(fecha)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, figsize=(20, 12))
    ax.plot(new_fechas, real, 'gold', linewidth=2)
    ax.plot(new_fechas, predictions, 'orangered', linewidth=2)
    ax.set_xlabel('Tiempo', fontname="Arial", fontsize=letter_size)
    ax.set_ylabel('Predicción vs Real', fontname="Arial",
                  fontsize=letter_size+2)
    ax.set_title(f"Predicciones vs real {str(indice)}",
                 fontname="Arial", fontsize=letter_size+10)
    ax.legend(['real', 'predicción'], loc='upper left',
              prop={'size': letter_size+5})
    # Tamaño de los ejes
    for tick in ax.get_xticklabels():
        tick.set_fontsize(letter_size)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(letter_size)
    try_create_folder(f"results/{folder_name}")
    plt.xticks(rotation=75)
    plt.show()
    fig.savefig(f"results/{folder_name}/{indice}_results.png")


def plot_multiple_xy_results(predictions, y_test, target_cols, ind,
                             folder_name="nn"):
    """
    Plotea lo resutltados de la red, cuando es con más de un input
    Parameters
    ----------
    predictions : array
        predicciones.
    real : array
        valores reales.
    names : list
        nombre de las columans target.
    folder_name : string, optional
        directorio donde se dejaran las carpetas. The default is "nn".
    Returns
    -------
    Plot.
    """
    try_create_folder("results")
    try_create_folder(f"results/{folder_name}")
    for i in range(predictions.shape[1]):
        print("Resultados:", target_cols[i], "...")
        predi = predictions[:, i]
        reali = y_test[:, i]
        namei = target_cols[i]
        plot_xy_results(predi, reali, index=ind, name=namei,
                        folder_name=folder_name)


def plot_xy_results(predictions, real, index=str(1), name="col",
                    folder_name="nn"):
    """
    Plot sequence de la secuecnia
    Parameters
    ----------
    predictions : array
        predicciones.
    real : array
        valores reales.
    fechas : array
        array de fechas.
    indice : TYPE
        indice de la columna.
    Returns
    -------
    plot de prediciones vs real.
    """
    plt.style.use('dark_background')
    letter_size = 20
    mae = np.abs(predictions - real).mean()
    mae = round(mae, 4)
    # caracteristicas del dataset
    mean = round(real.mean(), 1)
    std = round(real.std(), 1)
    fig, ax = plt.subplots(1, figsize=(22, 12))
    plt.scatter(real, real, color='green')
    plt.scatter(real, predictions, color='orangered')
    titulo = f"Predicciones {name} --> error: {str(mae)}" + "\n" +\
        f"Caracteristicas: promedio: {mean}, desv: {std}"
    plt.title(titulo, fontsize=30)
    plt.xlabel('Real', fontsize=30)
    plt.ylabel(f'Predicción {folder_name}', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(['real', 'predicción'], loc='upper left',
               prop={'size': letter_size+5})
    # plt.ylim(0, 4600)
    # plt.xlim(0, 4600)
    plt.show()
    path = f"results/{folder_name}/{index}-{name}.png"
    fig.savefig(path)


def pairplot_sns(df, name="Entrenamiento"):
    """
    Pairplot del dataframe insertado

    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".

    Returns
    -------
    Pairplot.

    """
    length = len(list(df.columns))
    sns.set(font_scale=0.75 * length)
    plt.style.use('dark_background')
    pplot = sns.pairplot(df)
    pplot.fig.set_figwidth(7.5*length)
    pplot.fig.set_figheight(7.5*length)
    pplot.set(title=name)


def pairgrid_plot(df, name="Entrenamiento"):
    """
    Pair grid del dataframe insertado
    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".

    Returns
    -------
    Pairgrid plot.

    """
    length = len(list(df.columns))
    if length <= 2:
        sns.set(font_scale=0.75 * length)
        plt.style.use('dark_background')
        g = sns.PairGrid(df)
        g.map_upper(sns.histplot)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        g.fig.set_figwidth(7.5*length)
        g.fig.set_figheight(7.5*length)
        g.set(title=name)
    else:
        sns.set(font_scale=15)
        plt.style.use('dark_background')
        g = sns.PairGrid(df)
        g.map_upper(sns.histplot)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        g.fig.set_figwidth(length/2)
        g.fig.set_figheight(length/2)
        g.set(title=name)


def violinplot(df, col, name="Entrenamiento"):
    """
    Violin plot de la columna de un dataframe

    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    col : string
        nombre de la columna a realizar el violinplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".

    Returns
    -------
    Violin plot de la columna.

    """
    rcParams['figure.figsize'] = 15, 15
    g = sns.violinplot(y=col, data=df)
    g.set_yticklabels(g.get_yticks(), size=30)
    g.axes.set_title(name, fontsize=40)
    g.set_ylabel(col, fontsize=40)
    plt.show()


def kernel_density_estimation(df, col, name="Entrenamiento", bw_adjust=0.1):
    """
    Estimación de la densidad a través de un kernel

    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    col : string
        nombre de la columna a realizar el violinplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".
    bw_adjust : float, optional
        Ajuste de la distribución. The default is 0.1.
    Returns
    -------
    Estimación de la distribución de la columna.

    """
    sns.set(font_scale=1.5)
    plt.style.use('dark_background')
    pplot = sns.displot(df, x=col, kind="kde", bw_adjust=bw_adjust)
    pplot.fig.set_figwidth(10)
    pplot.fig.set_figheight(8)
    pplot.set(title=name)
