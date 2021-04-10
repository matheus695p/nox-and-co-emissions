import io
import os
import numpy as np
import pandas as pd
from keras.utils.vis_utils import plot_model
from src.clean_module import try_create_folder
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


def mae_evaluation(predictions, real, names, nn,
                   folder_name="nn_architectures", filename="algo"):
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
    names : model
        tf model.
    folder_name : string, optional
        directorio donde se dejaran las carpetas. The default is "nn".
    Returns
    -------
    Plot.
    """
    try_create_folder("results")
    try_create_folder(f"results/{folder_name}")

    output = []
    for i in range(predictions.shape[1]):
        print("Resultados:", names[i], "...")
        name = names[i]
        predi = predictions[:, i]
        reali = real[:, i]
        mae = np.abs(predi - reali).mean()
        mae = round(mae, 4)
        output.append([name, mae, filename])
    output = pd.DataFrame(output, columns=["variable", "mae", "architecture"])
    output.to_csv(f"results/{folder_name}/{filename}.csv", index=False)
    plot_model(nn, to_file=f"results/{folder_name}/{filename}.png",
               show_shapes=True,
               show_layer_names=True)

    return output


def get_model_summary(model):
    """
    Pasa el summary del modelo a un string para guardar el resultado
    Parameters
    ----------
    model : tf model
        modelo de tensorflow.
    Returns
    -------
    summary_string : string
        summary del modelo en formato de string.
    """
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string
