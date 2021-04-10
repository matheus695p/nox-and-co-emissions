import os


def ls(path="."):
    """
    listar archivos en un directorio
    Parameters
    ----------
    path : string, optional
        DESCRIPTION. The default is ".".
    Returns
    -------
    files : list
        lista de archivos.
    """
    files = os.listdir(path)
    return files
