"""
    Módulo que implementa la extracción de características usando el modelo coclear del laboratorio de procesamiento digital de señales 
    La función principal es 'Matriz_Modelo_coclear'
"""
import matplotlib.pyplot as plt
from numpy import ndarray 
import numpy as np

from libreria.utileria import Banco_filtros_no_eq,  Energia_salida , FFT, TDC


# Extrae las características caracteristicas usando el modelo de la coclea resultado del laboratorio
def Matriz_Modelo_coclear(segmentos : ndarray[ndarray], n_coef : int, fs : int, fs_i : int, verbose = 0) -> ndarray[ndarray]:
    matriz_coclea = []

    for segmento in segmentos:
        matriz_coclea.append(Caracteristicas_Coclear(segmento, n_coef, fs, fs_i, verbose))
    return np.array(matriz_coclea)


# Función principal. Extracción de características para un segmento
def Caracteristicas_Coclear(segmento, n_coef : int, fs : int, fs_i : int, verbose = False):
    bins, espectro, fs      = FFT(segmento, fs)
    tam_espectro            = len(espectro)

    frec_centrales          = Centros_Coclea(fs_i, fs, n_coef)
    banco_filtros_coclea    = Banco_filtros_no_eq(frec_centrales, tam_espectro, fs )

    Y                       = Energia_salida(espectro, banco_filtros_coclea)

    # Los primeros centros estan muy cerca y con 128 puntos hay una ventana que tiene todos los valores en 0. Esto se arregla cuando aumenta el numero de segmentos, pero para que no de error con el logaritmo se implemento el siguiente código
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = (Y[i-1] + Y[i+1]) / 2

    S                       = np.log(np.abs(Y))


    caracteristicas_coclea  = TDC(S, len(S) , n_coef)

    if(verbose):
        _plot_resumen(segmento,espectro,banco_filtros_coclea,Y,S,caracteristicas_coclea, bins)

    return caracteristicas_coclea


# Genera centros del banco de filtro formando puntos equidistantes en la escala de Mel
def Centros_Coclea(fs_l : int, fs_h : int, n_coef : int) -> list[float]:
    PI  = np.pi
    E   = np.exp
    Ln  = np.log
    SQRT= np.sqrt

    def Hz_to_M(Hz):
        return Ln( (Hz) / (2.003 * 10**4) ) / -141.2

    def M_to_Hz(M):
        return 2.003 * (10**4) * E(-141.2 * (M) ) 
        #return 2.003 * (10**4) * E(-141.2 * (M) ) + 160

    B       = Hz_to_M  
    B_1     = M_to_Hz

    Delta_B = B(fs_h) - B(fs_l)
    B_fl    = B(fs_l)

    puntos  = np.arange(0,n_coef+2)
    coef    = [  B_1(B_fl + punto * (Delta_B / (n_coef + 1)))  for punto in puntos]

    return coef


# Grafica a modo de resumen
def _plot_resumen(señal,espectro,banco,Y,S,vector_Gammatone, bins):
    plt.figure()

    plt.subplot(611)
    plt.plot(señal)

    plt.subplot(612)
    plt.plot(bins,espectro)

    plt.subplot(613)
    for H_m in banco:
        plt.plot(bins,H_m)

    plt.subplot(614)
    plt.plot(Y)

    plt.subplot(615)
    plt.plot(S)

    plt.subplot(616)
    plt.plot(vector_Gammatone)

    plt.show()

