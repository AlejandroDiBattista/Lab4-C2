# Correciones

Los test no funcionan, el trabajo no cumple los requisitos para promocionar la materia.
Sin embargo, la mayoria de las funciones trabajan por lo que si permite regularizar la materia.

# Error 1: Incorrecta implementación de Oferta
1. Elimina el __init__ de la clase Oferta. Nunca se debe cambiar la forma de crear la clase porque rompe el resto del codigo. 

2. No implementa el método esAplicable correctamente. 
Debe devolver True si el producto cumple con la oferta y False en caso contrario. Esta funcionalidad es heredada por las demas clases. 

3. OfertaDescuento no se iniciliaza correctamente.
Cambiar la definicion del __init__, no llama a la inicializacion de la clase padre y no guarda el porcentaje.

4. Oferta2x1 no se inicializa correctamente.
Cambia el orden de los parametros y hace obligatorio el tipo.
No llama a la inicializacion de la clase padre.

```python
class Oferta:
    # Falta la inicialización de la clase
    def __init__(self, descripcion, codigos=[], tipos=[]):
        self.descripcion = descripcion
        self.codigos = codigos
        self.tipos = tipos

    # Esta debe ser la implementacion que deben heredar las demas clases.
    def esAplicable(self, producto, cantidad):
        return producto.codigo in self.codigos or producto.tipo in self.tipos
    #...

class OfertaDescuento:
    # def __init__(self, porcentaje, codigos, tipos=None):  
    def __init__(self, porcentaje, codigos=[], tipos=[]):
        # Se reemplaza por la inicializacion de la clase padre
        # self.codigos = codigos
        # self.tipos = tipos if tipos is not None else [] 
        super().__init__(f"Descuento {porcentaje}%", codigos, tipos)
        self.porcentaje = porcentaje

class Oferta2x1:
    # def __init__(self, tipos, codigos): 
    def __init__(self, codigos=[], tipos=[]): 
        super*().__init__("Oferta 2x1", codigos, tipos)
        
    def esAplicable(self, producto, cantidad):
        # No toma en cuenta la cantidad y no utiliza el metodo de la clase padre
        # return producto.codigo in self.codigos and producto.tipo in self.tipos
        return super().esAplicable(producto, cantidad) and cantidad >= 2
    #...
```

# Error 2: Incorrecta implementacion de Catalogo.
1. No implemento el metodo 'informe'
2. Al calcular los descuento llama a `es_aplicable` cuando la funcion se llama `esAplicable`

# Error 3: Factura mal implementada.
1. Guarda los productos en un diccionario cuya clave es el codigo de producto y el valor es la cantidad pero despues considera que el producto esta almacenado en el diccionario.

2. Usa `items` en el calculo de la factura pero no esta definido (usa productos en su lugar que es un diccionario) 

## Resumen

Nota: 6