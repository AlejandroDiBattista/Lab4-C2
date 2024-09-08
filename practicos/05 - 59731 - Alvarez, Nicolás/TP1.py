"""
# EJERCICIO 1
informacion=[
        "HTML es un lengueje basado en etiquetas",
        "CSS es un lenguaje de estilos",
        "GIT es un sistema de control de versiones distribuido"
]
def realizar_operaciones () :
    cantidad_palabras_por_frase=0;
    cantidad_caracteres_por_frase=0;
    #Recorro la lista de informacion por medio del ciclo repetitivo for
    for palabra in informacion:
        #El metodo split lo utilizo para dividir
        #la lista de informacion en listas independienteses y
        #almaceno ese resultado en una variable
        respuesta = palabra.split()
        #print(respuesta)
        #La funcion len() se usa para obtener la longitud ya sea en objetos,listas
        cantidad_palabras_por_frase = len(respuesta);
        cantidad_caracteres_por_frase = len(palabra)
        print(f"{respuesta} tiene",cantidad_palabras_por_frase,"palabra/as y",cantidad_caracteres_por_frase,"caracteres"); 
realizar_operaciones()
"""
# EJERCICIO 2

def buscar_valores():
    # Establezco los puntos de paso en una lista de tuplas
    punto = [(0, 0), (1, 8), (2, 12), (3, 12), (5, 0)]
    # Por medio del metodo range genera una secuencia de numeros
    valor_rango = range(-10,11);
    #print(valor_rango)
    #recorro cada uno de los valores del rango
    for valor_a in valor_rango:
        #print(valor_a)
        for valor_b in valor_rango:
            for valor_c in valor_rango:
                # Establezco la bandera en un valor inicial y puede ser modificada
                bandera = 1  
                for coordenada_x, coordenada_y in punto:
                    # Recorro cada una de las coordenadas de la lista de tuplas y realizo la validacion
                    # de la funcion cuadratica
                    if (valor_a * coordenada_x**2 + valor_b * coordenada_x + valor_c) != coordenada_y:
                        bandera = 0
                        break
                # Si la bandera se encuentra en 1 imprimo los valores
                if bandera == 1:
                    return print(f"a= {valor_a}\nb= {valor_b}\nc= {valor_c}");
                    break;
buscar_valores()
