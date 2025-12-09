import numpy as np

def Preprocesamiento(tiempo, señal, SEG_TAMAÑO = 256, SEG_TRASLAPE = 0.5, PE_ALPHA = 0.97 ):
    señal_pre_enfasis       = Filtro_pre_enfasis(señal, PE_ALPHA)
    segmentos               = Segmentar_señal(señal_pre_enfasis, SEG_TAMAÑO, SEG_TRASLAPE)
    segmentos_ventaneados   = Ventaneo(segmentos, SEG_TAMAÑO)

    return  Segmentar_señal(tiempo), segmentos_ventaneados


def Segmentar_señal(señal, SEG_TAMAÑO = 256, SEG_TRASLAPE = 0.5):
    salto = int(SEG_TAMAÑO *  SEG_TRASLAPE)

    segmentos = []
    for i in range(0, len(señal) - SEG_TAMAÑO, salto):
        segmentos.append( señal[ i: i+SEG_TAMAÑO] )
    
    return segmentos


def Filtro_pre_enfasis(señal, PE_ALPHA = 0.97):
    PE_señál = np.zeros_like(señal)
    
    PE_señál[0] = señal[0]
    for n in range(1, len(señal)):
        PE_señál[n] = señal[n] - PE_ALPHA * señal[n - 1]
    
    return PE_señál


def Ventaneo(segmentos, SEG_TAMAÑO = 256):
    HAMING_ALPHA_1          = 0.54
    HAMING_ALPHA_2          = 1 - HAMING_ALPHA_1
    HAMING_WINDOW           = HAMING_ALPHA_1 - HAMING_ALPHA_2 *\
                                np.cos(2 * np.pi * np.arange(SEG_TAMAÑO) / (SEG_TAMAÑO - 1))

    segmentos_ventaneados   = [ segmento * HAMING_WINDOW for segmento in segmentos ]
    return segmentos_ventaneados
