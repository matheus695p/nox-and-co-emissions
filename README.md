![Build Status](https://www.repostatus.org/badges/latest/active.svg)

#  NOX-and-CO-emissions

Esta en desarrollo, este repo, quedan partes por implementar ...

### Documentación

Documentación de los modulos con sphinx 

```sh
build/html/index.html
```


## Instalar las librerías necesarias para trabajar con deepAR en gluonts
```sh
$ git clone https://github.com/matheus695p/nox-and-co-emissions.git
$ cd nox-and-co-emissions
$ pip install -r requirements.txt
```

tree del proyecto

```sh
│   .gitignore
│   README.md
│   requirements.txt
│
├───codes
│       cleaning.py
│       open.py
│       preprocessing.py
│       train_cnn.py
│       train_nn.py
│
├───data
│   │   data.csv
│   │   gt_2011.csv
│   │   gt_2012.csv
│   │   gt_2013.csv
│   │   gt_2014.csv
│   │   gt_2015.csv
│   │   raw_data.csv
│   │
│   └───featured
│           featured_0_5.csv
│           featured_0_55.csv
│           featured_0_6.csv
│           featured_0_65.csv
│           featured_0_7.csv
│           featured_0_75.csv
│           featured_0_8.csv
│           featured_0_85.csv
│           featured_0_9.csv
│           featured_data.csv
│
├───documents
│       environmental pollution prediction.pdf
│       predicting-co-and-nox-emissions-from-gas-turbines-novel-data-and-2019.pdf
│
├───results
│   │   resultados.png
└───src
        clean_module.py
        config_cnn.py
        config_nn.py
        evaluation_module.py
        open_module.py
        preprocessing_module.py
        utils.py
        visualizations.py
        __init__.py

```

# Caso a resolver:

Los sistemas de monitoreo predictivo de emisiones (PEMS) son herramientas importantes para la validación y respaldo de costosos sistemas de monitoreo continuo de emisiones utilizados en centrales eléctricas basadas en turbinas de gas. Su implementación se basa en la disponibilidad de datos apropiados y ecológicamente válidos (superintendencia de medio ambiente debe estar de acuerdo). En este repositorio, usamos un conjunto de datos PEMS

* [![url a los datos(https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set)

Los cuales contienen 5 años de información de una turbina de gas para de CO y NOx. Analizamos los datos utilizando deep learning para presentar información útil sobre las predicciones de emisiones. Además, presentamos un procedimiento experimental de referencia para la comparabilidad de trabajos futuros.

## Diccionario de datos

Features:
* Variable (Abbr.) Unit Min Max Mean
* Ambient temperature (AT) °C
* Ambient pressure (AP) mbar 
* Ambient humidity (AH) (%) 
*  Air filter difference pressure (AFDP) mbar 
* Gas turbine exhaust pressure (GTEP) mbar 
* Turbine inlet temperature (TIT) °C 
*  Turbine after temperature (TAT) °C
* Compressor discharge pressure (CDP) mbar 
* Turbine energy yield (TEY) MWH 

Targets:
* Carbon monoxide (CO) mg/m3 
* Nitrogen oxides (NOx) mg/m3 

## Interpretación física

<p align="center">
  <img src="./images/diagrama1.png">
</p>



# Análisis

La idea con la que trataré el problema será:

* clean data: valores faltantes, vacios, 
* preprocessing: ordenar y dejar listo para etapa de feature engineering
* feature engineering: lag values, log(x), log(lag(x)), analisis de correlación por pearson, eliminando columnas que estén altamente correlacionadas 
* neural nets: redes dense, diferentes arquitecturas en los dataset tratados para distintos valores threshold de correlación entre columnas
* conv nets: redes convolucionales para hacer el end-to-end de la selección de caracteristicas

## Feature Engineering:

Feeature engineering:
* lag values: dado que estos corresponden a datos de series de tiempo, utilizamos los valores pasados en la predicción actual (ejemplo: temperatura t depende de t-1 (transferencia de calor))
* log(x): aplicamos logaritmo en las columnas de forma de estandarizar los valores
* log(lag(x)): aplicamos logaritmo en las columnas laggeadas de forma de estandarizar los valores
* analisis de correlación por pearson: esto permite crear múltiples datasets en los que las redes son evaluadas, eliminando columnas que estén altamente correlacionadas de acuerdo a un threshold de correlación entre 0.5 y 0.95


## NN implementación


## CNN implementación











