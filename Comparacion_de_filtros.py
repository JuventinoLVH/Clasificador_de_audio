import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import os  
import librosa

from libreria.preprocesamiento import Preprocesamiento
from libreria.utileria import Banco_filtros_no_eq, Energia_salida, Leer_archivo, FFT
from libreria.Caracteristicas_MFCC import Centros_MFCC
from libreria.Caracteristicas_Modelo_coclear import Centros_Coclea
from libreria.Caracteristicas_Mel_Coclea import Centros_Composicion
from libreria.Caracteristicas_Gammatone import Aplicar_banco_gammaton

## 
def main():
    ## Constantes
    DIR_BASE        = Path.home() / "Maestria" / "Primer_semestre" / "PDS" / "Proyecto_Filtros" 
    PATH_train      = DIR_BASE / "dataset_separado" / "train"

    I_PRUEBA        = 0
    I_SEGMENTO      = 2


    FS_INICIAL      = 100
    FS_FINAL        = 11025 
    SEGMENTOS       = 25600 
    N_COEF          = 24
    
    ## Banco de filtros de mel usando librosa
    descripcion_centros = {
        "fmin"    : FS_INICIAL,
        "fmax"    : FS_FINAL, 
        "htk"     : True
    }
    coef_librosa        = librosa.mel_frequencies(n_mels = N_COEF + 2, **descripcion_centros) 

    # El banco de librosa calcula los coeficientes en el dominio de la frecuencia
    #   por lo que divide la frecuencia y la cantidad de semgentos entre dos
    sr                  = FS_FINAL * 2
    n_fft               = SEGMENTOS * 2

    banco_MFCC_librosa  = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels = N_COEF, **descripcion_centros)
    banco_MFCC_librosa  = banco_MFCC_librosa / banco_MFCC_librosa.max(axis=1).reshape(-1, 1)
    banco_MFCC_librosa  = [ banco[:-1] for banco in banco_MFCC_librosa]

    ## Bancos de filtros propuestos en el laboratorio
    impulso             = np.zeros(SEGMENTOS*2)
    impulso[0]          = 1

    # Centros
    MFCC                = Centros_MFCC(FS_INICIAL, FS_FINAL, N_COEF)
    Coclea              = Centros_Coclea(FS_INICIAL, FS_FINAL, N_COEF)
    Comp                = Centros_Composicion(FS_INICIAL, FS_FINAL, N_COEF)

    # Bancos
    b_MFCC              = Banco_filtros_no_eq(MFCC, SEGMENTOS, FS_FINAL)
    b_Coclea            = Banco_filtros_no_eq(Coclea, SEGMENTOS, FS_FINAL)
    b_Comp              = Banco_filtros_no_eq(Comp, SEGMENTOS, FS_FINAL)
    b_Gammatone         = Aplicar_banco_gammaton(impulso, FS_FINAL *2, N_COEF, FS_INICIAL)
    b_Gammatone         = b_Gammatone[::-1]
    b_Gammatone_fft     = []
    for b in b_Gammatone:
        b_Gammatone_fft.append(np.abs(np.fft.fft(b)[:SEGMENTOS]) )

##  Efecto de los bancos de filtros propuestos sobre una señal
    # Se abre una señal para probar los bancos de filtro
    REGISTROS               = list(PATH_train.iterdir())
    archivo_prueba          = REGISTROS[I_PRUEBA]
    tiempo, señal, fs       = Leer_archivo(archivo_prueba)

    tiempo_seg, segmentos   = Preprocesamiento(tiempo, señal)
    señal                   = segmentos[I_SEGMENTO]
    bins, espectro, fs_mid  = FFT(señal, fs)
    len_esp                 = len(espectro)

    # Centros
    S_MFCC                  = Centros_MFCC(FS_INICIAL, fs_mid, N_COEF)
    S_Coclea                = Centros_Coclea(FS_INICIAL, fs_mid, N_COEF)
    S_Comp                  = Centros_Composicion(FS_INICIAL, fs_mid, N_COEF)

    # Bancos
    S_b_MFCC                = Banco_filtros_no_eq(S_MFCC, len_esp, fs_mid)
    S_b_Coclea              = Banco_filtros_no_eq(S_Coclea, len_esp, fs_mid)
    S_b_Comp                = Banco_filtros_no_eq(S_Comp, len_esp, fs_mid)

    # Efecto del banco
    S_f_MFCC                = [ b * espectro for b in S_b_MFCC]
    S_f_Coclea              = [ b * espectro for b in S_b_Coclea]
    S_f_Comp                = [ b * espectro for b in S_b_Comp]
    S_f_Gammatone           = Aplicar_banco_gammaton(señal, fs, N_COEF, FS_INICIAL)
    S_f_Gammatone           = S_f_Gammatone[::-1]
    S_f_Gammatone_fft       = []
    for b in S_f_Gammatone:
        S_f_Gammatone_fft.append(np.abs(np.fft.fft(b)[:len_esp]))

    ## Sección de gráficas
    Plot_Comparacion_MFFC_librosa_manual(coef_librosa, MFCC, banco_MFCC_librosa, b_MFCC)
    Plot_Bancos_de_filtros(b_MFCC, b_Coclea, b_Comp, b_Gammatone_fft)
    Plot_Efecto_de_filtros(señal, S_f_MFCC, S_f_Coclea, S_f_Comp, S_f_Gammatone_fft, fs)
    Plot_Frecuencias_centrales(b_Gammatone_fft, MFCC, Coclea, Comp, FS_FINAL , SEGMENTOS)

## 


def Plot_Comparacion_MFFC_librosa_manual(c_librosa,c_manual, b_librosa,b_manual):
    c_manual_fixed  = c_manual

    # Diferencias entre los coeficientes
    dif_coef_mel    = [ np.abs(b - c_librosa[i]) for i,b in enumerate(c_manual_fixed)]
    dif_bancos_mel  = [ np.abs(b - b_librosa[i]) for i,b in enumerate(b_manual)]

    # Indices de las barras de la primeras gráfica
    barWidth        = 0.3
    offset          = 0.027
    index_librosa   = np.arange( len(c_librosa) )
    index_manual    = [ i + barWidth + offset for i in index_librosa]
    index_dif       = [ i + barWidth + offset for i in index_manual]

    # Graficas

    # Primera sección
    plt.figure()

    #   Diferencia entre coeficientes
    plt.subplot(3,1,1)
    plt.title("Diferencia entre los coeficientes de mel de librosa y los calculados manualmente")

    plt.bar(index_librosa, c_librosa, label = "Librosa", width = barWidth )
    plt.bar(index_manual, c_manual_fixed, label = "Manual", width = barWidth )
    plt.bar(index_dif, dif_coef_mel, label = "Diferencia", width = barWidth )
    plt.legend()

    #   Centros manuales
    plt.subplot(3,1,2)
    plt.title("Banco de filtros calculado manualmente")
    for banco in b_manual:
        plt.plot(banco)

    #   Centros con la libreria de Mel
    plt.subplot(3,1,3)
    plt.title("Banco de filtros de mel de librosa")
    for banco in b_librosa:
        plt.plot(banco)

    # Segunda sección 
    plt.figure()
    plt.subplot(2,1,1)
    plt.title("Ambos bancos traslapados")

    plt.plot(b_manual[0], label = 'Banco manual', color = 'red' )
    plt.plot(b_librosa[0], label = 'Banco de librosa', color = 'blue' )
    for i in range(1, len(b_manual) ):
        plt.plot(b_manual[i], color = 'red' )
        plt.plot(b_librosa[i], color = 'blue' )

    plt.legend()

    plt.subplot(2,1,2)
    plt.title("Diferencia punto por punto entre los bancos de filtros")
    for banco in dif_bancos_mel:
        plt.plot(banco)

    plt.show()

def Plot_Bancos_de_filtros(b_MFCC, b_Coclea, b_Comp, b_Gammatone):
    plt.figure()

    # Banco MFCC
    plt.subplot(4,1,1)
    plt.title("Banco de filtros MFCC")
    for banco in b_MFCC:
        plt.plot(banco)

    # Banco de composicion
    plt.subplot(4,1,2)
    plt.title("Banco de filtros de composición")
    for i,banco in enumerate(b_Comp):
        plt.plot(banco)

    # Banco del modelo coclear
    plt.subplot(4,1,3)
    plt.title(" Banco de filtros del modelo coclear")
    for banco in b_Coclea:
        plt.plot(banco)

    # Banco del Gammatone
    plt.subplot(4,1,4)
    plt.title("Respuesta al impulso del modelo Gammatone en el dominio de la frecuencia")
    for banco in b_Gammatone:
        plt.plot(banco)

    plt.show()

def Plot_Efecto_de_filtros( señal, f_MFCC, f_Coclea, f_Comp, f_Gammatone, fs):
    bins, espectro, fs_mit  = FFT(señal, fs)

    # Utileria para graficar el banco y la potencia
    def Plot_banco_y_potencia(banco, indice_fig, titulo):
        potencias = []

        plt.subplot(6, 1, indice_fig)
        plt.title(titulo)
        for b in banco:
            plt.plot(b)
            potencias.append( np.sum( np.abs(b) ) )

        plt.subplot(6, 1, indice_fig + 1)
        plt.plot(potencias)

    # Primera sección
    plt.figure()

    #   Señal
    plt.subplot(6,1,1)
    plt.title("Señál y su espectro")
    plt.plot(señal)

    #   Espectro
    plt.subplot(6,1,2)
    plt.plot(bins, espectro)

    #   MFCC
    Plot_banco_y_potencia( f_MFCC, 3, "Banco de MFCC y la potencia de cada banco")

    #   Composición 
    Plot_banco_y_potencia( f_Comp, 5, "Banco de la composición y potencia de cada banco")

    # Segunda sección
    plt.figure()

    #   Señal
    plt.subplot(6,1,1)
    plt.title("Señál y su espectro")
    plt.plot(señal)

    #   Espectro
    plt.subplot(6,1,2)
    plt.plot(bins, espectro)

    #   Modelo coclear
    Plot_banco_y_potencia( f_Coclea, 3, "Banco del modelo coclear y potencia de cada banco")

    #   Modelo Gammatone
    Plot_banco_y_potencia( f_Gammatone, 5, "Banco del gammaton y potencia de cada banco")


    plt.show()

def Plot_Frecuencias_centrales(frec_Gammatone, MFCC, Coclea, Composicion, FS_FINAL, SEGMENTOS):
    MFCC        = MFCC[1:-1] # El primer centro y el ultimo son la frecuencia inicial y la final respectivamente
    Coclea      = Coclea[1:-1]
    Composicion = Composicion[1:-1]

    # Obtenemos el indice de la frecuencia cuyo valor es el máximo
    B_Gammatone = []
    for filtro in frec_Gammatone :
        max_val = -1
        indice  = -1

        for i,val in enumerate(filtro):
            if np.abs(val) > max_val:
                max_val = val
                indice = i    
        B_Gammatone.append(indice) 

    bin_to_freq = lambda x : (x * FS_FINAL) / SEGMENTOS
    Gammatone   = [ bin_to_freq(g) for g in B_Gammatone]

    # Se imprimen los centros para que se vea más claro
    print("Frecuencias centrales del filtro Gammatone")
    print(Gammatone, "\n")

    print("Frecuencias centrales de los filtros MFCC")
    print(MFCC, "\n")

    print("Frecuencias centrales de los filtros Coclea")
    print(Coclea, "\n")

    # Indices de las graficas 
    barWidth        = 0.20
    offset          = 0.027
    index_COMP      = np.arange( len(Composicion) )
    index_MFCC      = [ i + barWidth + offset for i in index_COMP]
    index_Gammatone = [ i + barWidth + offset for i in index_MFCC]
    index_Coclea    = [ i + barWidth + offset for i in index_Gammatone]
    index_line      = [1] * len(Composicion)

    # Graficas
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4], 'hspace' : 0.005})
    fig.tight_layout()
    
    ax[0].scatter(Composicion, index_line, color = 'blue')
    ax[0].scatter(MFCC, index_line, color = "orange")
    ax[0].scatter(Gammatone, index_line, color = 'green')
    ax[0].scatter(Coclea, index_line, color = 'red')
    ax[0].grid()

    ax[1].barh(index_COMP, Composicion[::-1], label = "Comp", height = barWidth )
    ax[1].barh(index_MFCC, MFCC[::-1], label = "MFCC", height = barWidth )
    ax[1].barh(index_Gammatone[::-1], Gammatone, label = "Gammatone" , height = barWidth )
    ax[1].barh(index_Coclea[::-1], Coclea, label = "Coclea", height = barWidth )
    ax[1].grid()
    plt.legend()

    plt.figure()

    plt.bar(index_COMP, Composicion, label = "Comp", width = barWidth )
    plt.bar(index_MFCC, MFCC, label = "MFCC", width = barWidth )
    plt.bar(index_Gammatone, Gammatone, label = "Gammatone" , width = barWidth )
    plt.bar(index_Coclea, Coclea, label = "Coclea", width = barWidth )
    plt.legend()

    plt.show()



if __name__ == "__main__":
    main()
