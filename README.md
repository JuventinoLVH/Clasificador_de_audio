# Proyecto : Clasificación de números
 
    El proyecto extrae las características de la voz usando diferentes filtros (Composición, Gammatone, MFCC y
    Modelo coclear). 
    Las características son utilizadas para entrenar un modelo de clustering (K-means) y comparar el desempeño de
    los diferentes filtros.
 
    Adicionalmente, se compara el proceso de extracion de características usando las funciones echas en la carpeta
    de libreria con las funciones de la librería librosa.
 

# Estructura del proyecto
 
    La estructura del proyecto está pensada para mostrar todo el proceso de extracción de características.
    Iniciando desde el dataset original. El orden en el que se debe ejecutar los scripts es el siguiente:
    1. Partir_dataset.py
    2. Extraer_caracteristicas.py
    3. Convertir_npy_2_csv.py
    4. Entrenar_clusters.py
 
    Adicionalmente, se puede ejecutar el script Comparacion_de_filtros.py para comparar el desempeño de los diferentes
    filtros y ver que tanto error existe entre la implementación propia y la de la librería librosa. 
 
    NOTA : Es necesario que al ejecutar el programa tenga derechos de escritura en la carpeta del proyecto para que
    pueda crear las carpetas y archivos necesarios. Ademas hay que editar las variables de configuración para
    indicar las dirección del dataset original, las carpetas de salida, los parámetros de entrenamiento, etc.
 
    NOTA : La carpeta de dataset_separado se crea al ejecutar el script Partir_dataset.py. 

.
├── Caracteristicas_extraidas               <- Se almacenan las características extraídas dependiendo del filtro
│   ├── Composicion                         <- Características de cada registro usando el filtro compuesto.
│   ├── Composicion.csv                     <- Archivo CSV con las caracteristicas de las ventanas de cada registro
│   ├── Gammatone                           <- Identico, pero usando el filtro Gammatone
│   ├── Gammatone.csv
│   ├── MFCC                                <- Idem, usando MFCC     
│   ├── MFCC.csv
│   ├── Modelo_coclear                      <- Idem, usando el modelo coclear
│   └── Modelo_coclear.csv
├── dataset_separado                        <- Dataset separado en carpetas de entrenamiento y prueba
│   ├── test
│   └── train
├── dataset_sonidos                         <- Dataset original con los sonidos de números 
├── libreria
│   ├── Caracteristicas_Gammatone.py        <- Características usando filtro Gammatone
│   ├── Caracteristicas_Mel_Coclea.py       <- Idem, usando filtro compuesto entre Mel y coclear
│   ├── Caracteristicas_MFCC.py             <- Idem, usando MFCC
│   ├── Caracteristicas_Modelo_coclear.py   <- Idem, usando Modelo coclear
│   ├── preprocesamiento.py                 <- Preprocesamiento estandar
│   └── utileria.py                         <- Utilidades varias
├── Comparacion_de_filtros.py               <- Script para comparar el desempeño de los diferentes filtros 
├── Convertir_npy_2_csv.py                  <- Tercer archivo, Convierte los archivos .npy a .csv
├── Entrenar_clusters.py                    <- Cuarto archivo, Entrena los modelos de clustering
├── Extraer_caracteristicas.py              <- Segundo archivo, Extrae las características de los sonidos
├── Partir_dataset.py                       <- Primer archivo, Parte el dataset en train y test 
├── README.md
├── Reporte_de_ejecuciones.pdf
└── requerimientos.txt

14 directories, 1048 files
