import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans

from tqdm import tqdm
import pandas as pd
import numpy as np
import os  

from libreria.preprocesamiento import Preprocesamiento
from libreria.utileria import Leer_archivo, Discriminar_archivo, Moda

## Extrae las caracteristicas MFCC de todos los archivos y los guarda en un archivo .npy
def main( verbose = False ):
    ## Constantes
    DIR_BASE        = Path.home() / "Maestria" / "Primer_semestre" / "PDS" / "Proyecto_Filtros" 

    PATH_train      = DIR_BASE / "dataset_separado" / "train"
    PATH_test       = DIR_BASE / "dataset_separado" / "test"

    CAR_DEST_CSV    = DIR_BASE / "Caracteristicas_extraidas" 

    REGISTROS_train = list(PATH_train.iterdir())
    REGISTROS_test  = list(PATH_test.iterdir())

    DIRECTORIOS_CAR     = [
        'MFCC', 
        'Modelo_coclear', 
        'Gammatone', 
        'Composicion'
    ]
    ## Se guardan los nombres de los archivos de train y test
    train_etiquetas     = [ Discriminar_archivo(reg) for reg in REGISTROS_train]
    test_etiquetas      = [ Discriminar_archivo(reg) for reg in REGISTROS_test]

    Nombre_train        = [ reg.name for reg in REGISTROS_train]
    Nombre_test         = [ reg.name for reg in REGISTROS_test]

    ## Se cargan las caracteristicas y etiquetas de las caracteristicas
    for tipo_car, nombre_car in enumerate(DIRECTORIOS_CAR):
        print(f"Caracteristicas: {nombre_car}")
        # Se abre el archivo con las caracteristicas correspondientes
        csv_dir         = CAR_DEST_CSV / str(nombre_car + ".csv")
        caract          = pd.read_csv(csv_dir)

        # Para el entrenamiento solo se toman las caracteristicas del conjunto de entrenamiento
        segmentos_train = []

        for segmento in caract.nombre_segmento: 
            archivo_seg = segmento.split("_")[0]
            if archivo_seg in Nombre_train:
                segmentos_train.append(True)
            else :
                segmentos_train.append(False)

    
        # Se eliminan las columnas de etiquetas y nombre de segmento para quedarse solo con las caracteristicas
        train_x         = caract[segmentos_train]
        train_x         = caract.drop(columns = ['nombre_segmento'])
        train_x         = train_x.drop(columns = ['etiqueta'])

        
    ## Se cargan las caracteristicas y etiquetas de las caracteristicas

        # Entrenamiento del clasificador K-Means 
        model_KM        = KMeans( 10, random_state = 42 )
        model_KM.fit(train_x)

        # Predicciones de los archivos de entrenamiento
        predic_archivo  = [ Predecir_archivo(archivo, model_KM, caract) for archivo in REGISTROS_train]

        # Diccionario que mapea el cluster a la etiqueta
        cluster_2_label = Imprimir_modas_de_clusters(train_etiquetas, predic_archivo)

        # Si verbose es verdadero, se imprimen las metricas del modelo tambine para los archivos de entrenamiento
        if verbose:
            Evaluar_metricas_del_modelo(REGISTROS_train, train_etiquetas, model_KM, cluster_2_label, caract)
        Evaluar_metricas_del_modelo(REGISTROS_test, test_etiquetas, model_KM, cluster_2_label, caract)




'''
    Funciones privadas necesarias para la clasificacion 
'''
## Funciones privadas

def Predecir_archivo(archivo, predictor, segmentos): 
    segmentos_arhivo        = segmentos[
        segmentos['nombre_segmento'].str.contains(archivo.name)
    ]
    segmentos_arhivo        = segmentos_arhivo.drop(columns = ['nombre_segmento'])
    segmentos_arhivo        = segmentos_arhivo.drop(columns = ['etiqueta'])

    clasificacion_segmentos = predictor.predict(segmentos_arhivo)
    return int(Moda( list(clasificacion_segmentos) ))


def Imprimir_modas_de_clusters( etiquetas, predicciones ):
    mapeo = {}
    for i in range(10):
        moda = Moda(Revisar_etiquetas_cluster(i, etiquetas, predicciones))
        print(f"Cluster {i} : {moda}")
        mapeo[i] = moda

    return mapeo


def Revisar_etiquetas_cluster(cluster_enfoque, etiquetas_real, etiquetas_predict):
    etiquetas_en_cluster = []
    for i in range(len(etiquetas_predict)):
        if(etiquetas_predict[i] == cluster_enfoque):
            etiquetas_en_cluster.append(etiquetas_real[i])

    return etiquetas_en_cluster


def Evaluar_metricas_del_modelo(archivos, etiquetas, modelo , cluster_2_etiqueta, segmentos):
    prediccion_archivos  = [ Predecir_archivo(archivo, modelo, segmentos) for archivo in archivos ]
    predicciones = [cluster_2_etiqueta[i] for i in prediccion_archivos]

    cm = confusion_matrix(etiquetas, predicciones)

    print(classification_report(etiquetas, predicciones))
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cluster_2_etiqueta.keys())
    disp.plot()
    plt.show()

    return


## La función main solo se ejecuta si el script es ejecutado directamente
if __name__ == "__main__":
    main(verbose = False)
