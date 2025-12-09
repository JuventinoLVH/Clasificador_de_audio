from libreria.preprocesamiento import Preprocesamiento
import numpy as np
import pathlib
import struct
import wave
import os


#from libreria.utileria import Banco_filtros_no_eq, Energia_salida, Leer_archivo, FFT
#from libreria.utileria import Limpiar_directorio, Leer_archivo, Guardar_caracteristicas

"""
    Utileria para el manejo de señales
"""
def FFT(señal, fs):
    tam_señal   = len(señal)
    fft         = np.fft.fft(señal)
    fft_p       = np.abs(fft)

    seg_mit     = tam_señal // 2 if tam_señal % 2 == 0 else (tam_señal + 1) // 2
    fs_mit      = fs/2
    bins        = np.linspace(0, int(fs_mit), seg_mit)

    return bins, fft_p[:seg_mit], fs_mit


def Banco_filtros_no_eq(frecuencias, n_segmentos, fs):
    def no_equidistante(f_a,f_c,f_s,k):
        if (k <= f_c):
            return max(0, (k - f_a)/(f_c - f_a) )
        else:
            return max(0, (k - f_s)/(f_c - f_s) ) 

    return Banco_filtros(frecuencias, n_segmentos, fs, no_equidistante)


def Banco_filtros(f_centrales, num_segmentos, fs, funcion_construccion):
    banco       = []  
    bin2_freq   = lambda k : (k*fs) / num_segmentos

    for i in range (1, len(f_centrales) - 1):
        f_c     = f_centrales[i]
        f_a     = f_centrales[i - 1]
        f_s     = f_centrales[i + 1]

        b       = np.zeros(num_segmentos)
        for k in range(num_segmentos):
            f_bin   = bin2_freq(k)
            b[k]    = funcion_construccion(f_a, f_c, f_s, f_bin)

        banco.append(b)

    return banco


def TDC(banco_señales, tam_señal, K):
    coeficientes = np.zeros(K)
   
    for n in range(tam_señal):

        u_n = 1/np.sqrt(K) if n == 0 else np.sqrt(2/K)
        for k,señal in enumerate(banco_señales):
            coeficientes[n] += u_n * señal * np.cos( ( (2*k + 1)*n*np.pi ) / ( 2 * K) )

    return coeficientes

def Energia_salida(señal, H):
    n_filtros   = len(H)
    Y           = np.zeros(n_filtros)
    s_p         = np.pow( señal, 2 )

    for i,H_m in enumerate(H):
        Y[i] = np.sum(s_p * H_m)

    return Y

"""
    Utileria para ayudar en el clasificador
"""

def Moda(lista):
    return max(set(lista), key = lista.count)


"""
    Utileria para el manejo de archivos
"""


def Leer_archivo(archivo):
    # Se lee del archivo .wav el tiempo, la señal y la frecuencia de muestreo
    archivo = os.path.abspath(archivo)
    with wave.open(archivo, 'rb') as file:
        num_muestras = file.getnframes()
        señal = file.readframes(num_muestras)
        señal = np.array(struct.unpack('<' + 'h' * (num_muestras), señal))
        fs = file.getframerate()
        tiempo = np.linspace(0, num_muestras / fs, num_muestras )


    return  tiempo, señal, fs


def Discriminar_archivo(archivo):
    diccionario = {
        "cero" : 0,
        "uno" : 1,
        "dos" : 2,
        "tres" : 3,
        "cuatro" : 4,
        "cinco" : 5,
        "seis" : 6,
        "siete" : 7,
        "ocho" : 8,
        "nueve" : 9
    }
    nombre = archivo.name.split(".")[0]

    for key in diccionario:
        if key in nombre:
            return diccionario[key]

    return -1


def Limpiar_directorio(directorio):
    for archivo in directorio.iterdir():
        os.remove(archivo)


def Guardar_caracteristicas(registro, caracteristicas, directorio):
    nombre = registro.name.split(".")[0] + ".npy"
    np.save(directorio / nombre, caracteristicas)


def Copia_hiper(archivo, directorio):
    nombre = archivo.name
    os.symlink(archivo, directorio / nombre)



