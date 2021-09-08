from ClaseDiezInter import *
import sys

"""
python  main.py path_to_image
"""

path = sys.argv[1]

imagen = Cambiotamano(path)


ILL = imagen.descomposicion(2)
ILL = imagen.interpolacion(4,ILL)
