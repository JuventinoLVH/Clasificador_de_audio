import matplotlib.pyplot as plt
from numpy import ndarray 
import numpy as np

from libreria.utileria import Banco_filtros_no_eq,  Energia_salida , FFT, TDC




def Matriz_Gammatone(segmentos : ndarray[ndarray], n_coef : int, fs : int, fs_i : int, verbose = 0) -> ndarray[ndarray]:
    matriz_Gammatone = []

    for segmento in segmentos:
        matriz_Gammatone.append(Caracteristicas_Gammatone(segmento, n_coef, fs, fs_i, verbose))
    return np.array(matriz_Gammatone)


def Caracteristicas_Gammatone(segmento, n_coef : int, fs : int, fs_i : int, verbose = False):
    señal_banco_gammaton    = Aplicar_banco_gammaton(segmento, fs, n_coef, fs_i)
    señal_banco_gammaton    = señal_banco_gammaton[::-1] 

    len_s                   = len(segmento)
    mit_s                   = len_s//2 if len_s%2 == 0 else (len_s+1)//2

    espectro_banco          = [ np.abs(np.fft.fft(señal)[:mit_s]) for señal in señal_banco_gammaton]
    energia_banco           = [ np.sum(np.pow( señal, 2 )) for señal in espectro_banco ]
    log_energia             = np.log(energia_banco)

    caracteristicas_gamma  = TDC(log_energia, len(log_energia) , n_coef)

    if(verbose):
        _plot_resumen(segmento,señal_banco_gammaton,espectro_banco,energia_banco,log_energia, caracteristicas_gamma )

    return caracteristicas_gamma



# NOTA : Los coeficientes estan en orden inverso. El primer coeficiente corresponde al de mayor frecuencia y el ultimo al de menor
def Aplicar_banco_gammaton(señal, fs, n_coef, fs_inicial):
    forward, feedback = make_erb_filters(fs, n_coef, fs_inicial)
    y = erb_filter_bank(forward, feedback, señal)
    return y


def make_erb_filters(fs, num_channels, low_freq):
    T       = 1 / fs
    EarQ    = 9.26449 
    minBW   = 24.7
    order   = 1

    E       = np.exp
    ln      = np.log
    PI      = np.pi
    COS     = np.cos
    SEN     = np.sin
    sqrt    = np.sqrt

    cf = -(EarQ * minBW) + E(np.arange(1, num_channels + 1).reshape(-1, 1) * (-ln(fs / 2 + EarQ * minBW) + ln(low_freq + EarQ * minBW)) / num_channels) * (fs / 2 + EarQ * minBW)
    cf = cf.flatten() 

    ERB = ((cf / EarQ) ** order + minBW ** order) ** (1 / order)
    B = 1.019 * 2 * PI * ERB

    gain = np.abs(
            (-2 * E(4 * 1j * cf * PI * T) * T +\
              2 * E(-B * T + 2 * 1j * cf * PI * T) * T *\
                (COS(2 * cf * PI * T) - sqrt(3 - 2 ** (3 / 2)) *\
                  SEN(2 * cf * PI * T))) *\
            (-2 * E(4 * 1j * cf * PI * T) * T +\
              2 * E(-B * T + 2 * 1j * cf * PI * T) * T *\
                (COS(2 * cf * PI * T) + sqrt(3 - 2 ** (3 / 2)) *\
                  SEN(2 * cf * PI * T))) *\
            (-2 * E(4 * 1j * cf * PI * T) * T +\
              2 * E(-B * T + 2 * 1j * cf * PI * T) * T *\
                (COS(2 * cf * PI * T) - sqrt(3 + 2 ** (3 / 2)) *\
                  SEN(2 * cf * PI * T))) *\
            (-2 * E(4 * 1j * cf * PI * T) * T +\
              2 * E(-B * T + 2 * 1j * cf * PI * T) * T *\
                (COS(2 * cf * PI * T) + sqrt(3 + 2 ** (3 / 2)) *\
                  SEN(2 * cf * PI * T))) /\
            (-2 / E(2 * B * T) - 2 * E(4 * 1j * cf * PI * T) +\
                2 * (1 + E(4 * 1j * cf * PI * T)) / E(B * T)) ** 4)

    feedback = np.zeros((len(cf), 9))
    forward = np.zeros((len(cf), 5))

    forward[:, 0] = (T ** 4) / gain
    forward[:, 1] = -4 * T ** 4 * COS(2 * cf * PI * T) / E(B * T) / gain
    forward[:, 2] = 6 * T ** 4 * COS(4 * cf * PI * T) / E(2 * B * T) / gain
    forward[:, 3] = -4 * T ** 4 * COS(6 * cf * PI * T) / E(3 * B * T) / gain
    forward[:, 4] = T ** 4 * COS(8 * cf * PI * T) / E(4 * B * T) / gain

    feedback[:, 0] = np.ones(len(cf))
    feedback[:, 1] = -8 * COS(2 * cf * PI * T) / E(B * T)
    feedback[:, 2] = 4 * (4 + 3 * COS(4 * cf * PI * T)) / E(2 * B * T)
    feedback[:, 3] = -8 * (6 * COS(2 * cf * PI * T) + COS(6 * cf * PI * T)) / E(3 * B * T)
    feedback[:, 4] = 2 * (18 + 16 * COS(4 * cf * PI * T) + COS(8 * cf * PI * T)) / E(4 * B * T)
    feedback[:, 5] = -8 * (6 * COS(2 * cf * PI * T) + COS(6 * cf * PI * T)) / E(5 * B * T)
    feedback[:, 6] = 4 * (4 + 3 * COS(4 * cf * PI * T)) / E(6 * B * T)
    feedback[:, 7] = -8 * COS(2 * cf * PI * T) / E(7 * B * T)
    feedback[:, 8] = E(-8 * B * T)

    return forward, feedback


def Aplicar_filtro(forward, feedback, señal):
    y = np.zeros_like(señal)
    for n in range(len(señal)):
        y[n] = forward[0] * señal[n]

        for i in range(1, len(forward)):
            if n >= i:
                y[n] += forward[i] * señal[n - i]

        for i in range(1, len(feedback)):
            if n >= i:
                y[n] -= feedback[i] * y[n - i]
    return y


def erb_filter_bank(forward, feedback, señal):
    rows, cols = feedback.shape
    y = np.zeros((rows, len(señal)))

    for i in range(rows):
        y[i] = Aplicar_filtro(forward[i], feedback[i], señal)

    return y



def _plot_resumen(señal, banco, espectro_banco, S, log_energia, caracteristicas_gamma):
    plt.figure()

    plt.subplot(711)
    plt.plot(señal)

    plt.subplot(712)
    plt.plot(np.fft.fft(señal)[:len(señal)//2])

    plt.subplot(713)
    for H_m in banco:
        plt.plot(H_m)

    plt.subplot(714)
    for H_m in espectro_banco:
        plt.plot(H_m)

    plt.subplot(715)
    plt.plot(S)

    plt.subplot(716)
    plt.plot(log_energia)

    plt.subplot(717)
    plt.plot(caracteristicas_gamma)

    plt.show()
