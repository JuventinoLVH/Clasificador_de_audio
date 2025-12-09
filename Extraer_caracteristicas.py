import matplotlib.pyplot as plt
from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm
import os  

from libreria.preprocesamiento import Preprocesamiento
from libreria.utileria import Limpiar_directorio, Leer_archivo, Guardar_caracteristicas
from libreria.Caracteristicas_MFCC import Matriz_MFCC
from libreria.Caracteristicas_Modelo_coclear import Matriz_Modelo_coclear
from libreria.Caracteristicas_Mel_Coclea import Matriz_Composicion
from libreria.Caracteristicas_Gammatone import Matriz_Gammatone

## Crea y testea un clasificador K-NN usando las caracteristicas MFCC
def main():
    ## Constantes
    DIR_BASE            = Path.home() / "Maestria" / "Primer_semestre" / "PDS" / "Proyecto_Filtros" 
    PATH_Dataset        = DIR_BASE / "dataset" 

    CAR_DEST_MFCC       = DIR_BASE / "Caracteristicas_extraidas" / "MFCC"
    CAR_DEST_COCLEA     = DIR_BASE / "Caracteristicas_extraidas" / "Modelo_coclear"
    CAR_DEST_COMP       = DIR_BASE / "Caracteristicas_extraidas" / "Composicion"
    CAR_DEST_GAMMATON   = DIR_BASE / "Caracteristicas_extraidas" / "Gammatone"

    REGISTROS           = list(PATH_Dataset.iterdir())
    HILOS_PARALELOS     = 6 # XXX : Cambiar dependiendo de los hilos de la computadora


    ## Función para paralelizar la extracción de caracteristicas
    def Procesar_archivo(registro):
        tiempo, señal, fs       = Leer_archivo(registro)
        tiempo_seg, segmentos   = Preprocesamiento(tiempo, señal)

        parametros = {
            "n_coef" : 36,
            "fs" : fs,
            "fs_i" : 100
        }

        # Ejemplo de como se pude testear un solo segmento
        #car_coclea              = Matriz_Modelo_coclear([segmentos[2]], **parametros, verbose = True)
        #return

        car_MFCC                = Matriz_MFCC(segmentos, **parametros)
        car_coclea              = Matriz_Modelo_coclear(segmentos, **parametros)
        car_Gammatone           = Matriz_Gammatone(segmentos, **parametros)
        car_comp                = Matriz_Composicion(segmentos, **parametros)

        Guardar_caracteristicas(registro, car_MFCC, CAR_DEST_MFCC)
        Guardar_caracteristicas(registro, car_coclea, CAR_DEST_COCLEA)
        Guardar_caracteristicas(registro, car_Gammatone, CAR_DEST_GAMMATON)
        Guardar_caracteristicas(registro, car_comp, CAR_DEST_COMP)

    
    ## Se guardan las caracteristicas MFCC de todos los archivos
    Limpiar_directorio(CAR_DEST_MFCC)
    Limpiar_directorio(CAR_DEST_COCLEA)
    Limpiar_directorio(CAR_DEST_COMP)
    Limpiar_directorio(CAR_DEST_GAMMATON)

    Parallel(n_jobs = HILOS_PARALELOS)(delayed(Procesar_archivo)(registro) for registro in tqdm(REGISTROS))

##  
if __name__ == "__main__":
    main()

