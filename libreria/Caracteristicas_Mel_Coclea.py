"""
    Módulo que implementa la extracción de características Comp
    La función principal es 'Matriz_Comp'
"""
import matplotlib.pyplot as plt
from numpy import ndarray 
import numpy as np

from libreria.utileria import Banco_filtros_no_eq,  Energia_salida , FFT, TDC


# Extrae las características de un conjunto de segmentos
def Matriz_Composicion(segmentos : ndarray[ndarray], n_coef : int, fs : int, fs_i : int, verbose = 0) -> ndarray[ndarray]: 
    matriz_Comp = []

    for segmento in segmentos:
        matriz_Comp.append(Caracteristicas_Comp(segmento, n_coef, fs, fs_i, verbose))
    return np.array(matriz_Comp)


# Función principal. Extracción de características para un segmento
def Caracteristicas_Comp(segmento, n_coef : int, fs : int, fs_i : int, verbose = False):
    bins, espectro, fs      = FFT(segmento, fs)
    tam_espectro            = len(espectro)

    frec_centrales_m        = Centros_Mel(fs_i, fs, n_coef)
    frec_centrales_c        = Centros_mel2coc(frec_centrales_m)
    frec_centrales_hz       = Coclea2Hz(frec_centrales_c)

    banco_filtros_Comp      = Banco_filtros_no_eq(frec_centrales_hz, tam_espectro, fs )

    Y                       = Energia_salida(espectro, banco_filtros_Comp)
    S                       = np.log(Y)

    caracteristicas_comp    = TDC(S, len(S) , n_coef)

    if(verbose):
        _plot_resumen(segmento,espectro,banco_filtros_Comp,Y,S,caracteristicas_comp, bins)

    return caracteristicas_comp


# Genera centros equidistantes en la escala de Mel
def Centros_Mel(fs_l : int, fs_h : int, n_coef : int) -> list[float]:
    def Hz_to_Mel(hz):
        return 1125 * np.log(1 + hz / 700)
        #return np.log( (hz) / (2.003 * 10**4) ) / -141.2

    B       = Hz_to_Mel

    Delta_B = B(fs_h) - B(fs_l)
    B_fl    = B(fs_l)

    puntos  = np.arange(0,n_coef+2)

    coef    = [ B_fl + punto * (Delta_B / (n_coef + 1)) for punto in puntos]
    return coef


# Genera centros del banco de filtro formandos por llevar la escala de mel a la escala de la coclea
def Centros_Composicion(fs_l : int, fs_h : int, n_coef : int) -> list[float]:
    f_centrales_mel = Centros_Mel(fs_l, fs_h, n_coef)
    f_centrales_coc = Centros_mel2coc(f_centrales_mel)
    f_centrales_hz  = Coclea2Hz(f_centrales_coc)

    return f_centrales_hz


# Convierte de Mel a la escala de la coclea
def Centros_mel2coc(centros_mel):
    a           = 2.003 * (10**4)
    b           = -1.4142
    ln          = np.log
    E           = np.exp
    mel2cocl    = lambda f_mel : ( ln( E( f_mel / 1125 ) - 1 ) + ln(700/a) )/b


    return [mel2cocl(f_mel) for f_mel in centros_mel]


# Convierte de la escala de la coclea a Hz
def Coclea2Hz(f_coclea):
    E           = np.exp
    Ln          = np.log

    coclea2Hz   = lambda cm: 2.003 * (10**4) * E(-141.2 * (cm/100) ) 
    return [coclea2Hz(f_coclea) for f_coclea in f_coclea]


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


## Código de prueba
def __main__():
    ## Prueba de la función Centros_MFCC
    mel_f       = Centros_Mel(100, 10000, 38)
    cocl_f      = Centros_mel2coc(mel_f)

    print(mel_f)
    print(cocl_f)

    centros = [Coclea2Hz(f) for f in cocl_f]

    print(centros)
    plt.plot(centros)

if __name__ == "__main__":
    __main__()
