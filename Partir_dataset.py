# Comandos magicos para actualiar las librerias de las dependencias
#%load_ext autoreload
#$%autoreload 2

import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import numpy as np
import os  

from libreria.utileria import Discriminar_archivo, Copia_hiper, Limpiar_directorio
##

if __name__ == "__main__":
    DIR_BASE        = Path.home() / "Maestria" / "Primer_semestre" / "PDS" / "Proyecto_Filtros" 
    DATA_PATH       = DIR_BASE / "dataset" 
    PATH_train      = DIR_BASE / "dataset_separado" / "train"
    PATH_test       = DIR_BASE / "dataset_separado" / "test"

    REGISTROS       = list(DATA_PATH.iterdir())
    TRAIN_PERCENT   = 0.8
    TEST_PERCENT    = 0.2

## Antes de todo eliminamos los archivos que ya existen en los directorios de train y test
    Limpiar_directorio(PATH_train)
    Limpiar_directorio(PATH_test)

## Revolvemos los registros usando la misma semilla para reproducibilidad
    np.random.seed(0)
    np.random.shuffle(REGISTROS)

## Crea un arreglo con los registros ordenados por tipo de archivo. por ejemplo: Los archivos 'cuatro' tienen el indice '4'
    registros_ordenados = [[],[],[],[],[],[],[],[],[],[]] 
    for reg in REGISTROS:
        registros_ordenados[Discriminar_archivo(reg)].append(reg)

## Separamos el dataset en train y test de forma estratificada ( Misma proporción de cada clase en test y train )
    for i in range(10):
        clase_registros     = registros_ordenados[i]

        n_train = int(len(clase_registros) * TRAIN_PERCENT)
        n_test = len(clase_registros) - n_train

        reg_train = clase_registros[:n_train]
        reg_test  = clase_registros[ n_train: ]

        for reg in reg_train:
            Copia_hiper(reg, PATH_train)

        for reg in reg_test:
            Copia_hiper(reg, PATH_test)
    
