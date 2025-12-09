"""
    Guarda las caracteristicas MFCC de todos los archivos. 
"""
import matplotlib.pyplot as plt
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
import os  

from libreria.preprocesamiento import Preprocesamiento
from libreria.utileria import Leer_archivo, Discriminar_archivo, Moda


## Extrae las caracteristicas MFCC de todos los archivos y los guarda en un archivo .npy
def main( verbose = False ):
    ## Constantes
    DIR_BASE            = Path.home() / "Maestria" / "Primer_semestre" / "PDS" / "Proyecto_Filtros" 

    PATH_train          = DIR_BASE / "dataset_separado" / "train"
    PATH_test           = DIR_BASE / "dataset_separado" / "test"

    CAR_DEST_MFCC       = DIR_BASE / "Caracteristicas_extraidas" / "MFCC"
    CAR_DEST_COCLEA     = DIR_BASE / "Caracteristicas_extraidas" / "Modelo_coclear"
    CAR_DEST_GAMMATON   = DIR_BASE / "Caracteristicas_extraidas" / "Gammatone"
    CAR_DEST_COMP       = DIR_BASE / "Caracteristicas_extraidas" / "Composicion"

    CAR_DEST_CSV        = DIR_BASE / "Caracteristicas_extraidas" 

    REGISTROS_train     = list(PATH_train.iterdir())
    REGISTROS_test      = list(PATH_test.iterdir())
    REGISTROS           = REGISTROS_train + REGISTROS_test

    ## Se cargan las caracteristicas y etiquetas de las caracteristicas
    etiquetas_archivos      = Obtener_etiquetas(REGISTROS)
    directorios_car         = [
        CAR_DEST_MFCC, 
        CAR_DEST_COCLEA, 
        CAR_DEST_GAMMATON, 
        CAR_DEST_COMP
    ]
    caract_archivos         = Cargar_archivos(REGISTROS, directorios_car)


    ## La matriz que representa al archivo se convierte en un vector ( segmetnos x caracteristicas ) -> (caracteristicas)
    # Arreglo vacio con 4 arreglos vacios dentro
    caract_segmentos        = [ [] for _,_ in enumerate(directorios_car) ]

    # Por cada uno de los archivos
    for i,archivo in enumerate(caract_archivos):
        etiqueta_archivo    = etiquetas_archivos[i]
        nombre_archivo      = REGISTROS[i].name
        cant_segmentos      = len(archivo[0])

        # Por cada uno de los segmentos del archivo actual
        for i_seg in range(cant_segmentos):
            nombre_seg      = f"{nombre_archivo}_{i_seg}"

            # Para cada una de las caracteristicas
            for tipo_car,car in enumerate(directorios_car):
                segmento    = []   
                for caracteristica in archivo[tipo_car][i_seg]:
                    segmento.append(caracteristica)
                segmento.append(nombre_seg)
                segmento.append(etiqueta_archivo)

                # Se agrega el segmento de la caracteristica correspondiente
                caract_segmentos[tipo_car].append(segmento)

    ## El arreglo de caracteristicas se convierte en un dataframe
    df_caract_segmentos     = [ pd.DataFrame(caract) for caract in caract_segmentos ]
    num_car                 = len(caract_archivos[0][0][0])

    columnas_archivos       = [f"car_{i}" for i in range( num_car ) ]
    columnas_archivos.append("nombre_segmento")
    columnas_archivos.append("etiqueta")

    for df in df_caract_segmentos:
        df.columns = columnas_archivos

## Se guardan las caracteristicas en un dataframe
    print("Guardando en csv")
    for tipo_car,car in enumerate(tqdm(directorios_car)):
        df      = df_caract_segmentos[tipo_car]
        
        nombre  = car.name + ".csv"
        df.to_csv( CAR_DEST_CSV / nombre, index = False)
##

'''
    Funciones privadas necesarias para la clasificacion 
'''
## Funciones privadas

def Cargar_archivos(archivos,directorios_car):
    carac_archivos      = []

    print("Cargando archivos")
    for archivo in tqdm(archivos):
        registro        = archivo.name.split(".")[0]
        registro_car    = registro + ".npy"

        
        caracteristicas = []
        for directorio in directorios_car:
            archivo_car = np.load(directorio / registro_car)
            caracteristicas.append(archivo_car)

        carac_archivos.append(caracteristicas)

    return carac_archivos


def Obtener_etiquetas(REGISTROS):
    etiquetas = []
    for reg in REGISTROS:
        etiquetas.append(Discriminar_archivo(reg))
    return etiquetas


## La función main solo se ejecuta si el script es ejecutado directamente
if __name__ == "__main__":
    main(verbose = False)

