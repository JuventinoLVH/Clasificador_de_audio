"""
    Módulo que implementa la extracción de características MFCC
    La función principal es 'Matriz_MFCC'
"""
import matplotlib.pyplot as plt
from numpy import ndarray 
import numpy as np

from libreria.utileria import Banco_filtros_no_eq,  Energia_salida , FFT, TDC


# Extrae las características MFCC de un conjunto de segmentos
def Matriz_MFCC(segmentos : ndarray[ndarray], n_coef : int, fs : int, fs_i : int, verbose = 0) -> ndarray[ndarray]:
    matriz_MFCC = []

    for segmento in segmentos:
        matriz_MFCC.append(Caracteristicas_MFCC(segmento, n_coef, fs, fs_i, verbose))
    return np.array(matriz_MFCC)


# Función principal. Extracción de características para un segmento
def Caracteristicas_MFCC(segmento, n_coef : int, fs : int, fs_i : int, verbose = False):
    bins, espectro, fs      = FFT(segmento, fs)
    tam_espectro            = len(espectro)

    frec_centrales          = Centros_MFCC(fs_i, fs, n_coef)
    banco_filtros_MFCC      = Banco_filtros_no_eq(frec_centrales, tam_espectro, fs )

    Y                       = Energia_salida(espectro, banco_filtros_MFCC)
    S                       = np.log(Y)

    caracteristicas_mfcc    = TDC(S, len(S) , n_coef)

    if(verbose):
        _plot_resumen(segmento,espectro,banco_filtros_MFCC,Y,S,caracteristicas_mfcc, bins)

    return caracteristicas_mfcc


# Genera centros del banco de filtro formando puntos equidistantes en la escala de Mel
def Centros_MFCC(fs_l : int, fs_h : int, n_coef : int) -> list[float]:
    def Hz_to_Mel(hz):
        return 1125 * np.log(1 + hz / 700)

    def Mel_to_hs(ml):
        return 700 * (np.exp(ml / 1125) - 1)

    B       = Hz_to_Mel
    B_1     = Mel_to_hs

    Delta_B = B(fs_h) - B(fs_l)
    B_fl    = B(fs_l)

    puntos  = np.arange(0,n_coef+2)

    coef    = [ B_1( B_fl + punto * (Delta_B / (n_coef + 1)) ) for punto in puntos]
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
        plt.plot(H_m)

    plt.subplot(614)
    plt.plot(Y)

    plt.subplot(615)
    plt.plot(S)

    plt.subplot(616)
    plt.plot(vector_Gammatone)

    plt.show()

