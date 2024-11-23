# TP5: *1er Parcial*

Este trabajo cuenta como el `primer parcial`.
>  
> Es un `trabajo individual` y puede usar todos los recursos a su disposición, incluyendo el material del curso y búsquedas en internet para resolver dudas. 

> **Debe implementar su solución de manera individual** si comparte código con algún compañero invalida el trabajo de ambos.

> Debe ser presentado hasta las `23:59 del sábado 5 de octubre`.
> 

## Enunciado

El trabajo consiste en implementar, usando programación orientada a objetos, un sistema de facturación para una empresa de venta de productos.

Asociado a cada clase a implementar tiene los test correspondientes que verifican que la implementación es correcta.

Ademas estos test indican la forma exacta en que debe ser implementada la clase, incluyendo los nombres de los métodos y los parámetros que deben recibir y el comportamiento esperado.


### Requerimientos

#### Productos

- Los productos tienen un código único de 4 dígitos, un nombre (1 a 100 caracteres), un precio (entre 10 y 10,000), un tipo (0 a 20 caracteres) y una cantidad en existencia (entre 0 y 1000).
- Deben mantener la cantidad de productos en existencia y calcular su valor total.

#### Catálogo

- El catálogo se debe leer desde un archivo de texto `catalogo.csv` que tiene el siguiente formato (incluye encabezado):

```text 
codigo,nombre,precio,tipo,cantidad
```

- Debe descontar la existencia disponible.
- Debe agregar un producto.
- Debe buscar un producto por código.
- Debe poder analizar que oferta aplica a un producto
- Debe poder grabar los cambios en el catálogo en un archivo con el mismo formato.
- Debe poder generar un informe para ser impreso con el siguiente formato:

```text
INFORME CATALOGO 
Cantidad de productos:   <cantidad productos>
Cantidad de unidades:    <cantidad unidades>
Precio promedio:       $ <precio promedio>
Valor total:           $ <valor total>
Tipos de productos: 
  - <tipo>              :  <unidades>u x $ <precio promedio>
  - ...
Ofertas:
  - <descripción oferta>
  - ...
```

#### Ofertas

- La empresa tiene ofertas en algunos productos. Las ofertas pueden aplicarse a productos específicos (por código) o a todos los productos de un tipo determinado.
- Tipos de ofertas:
  - **Descuento porcentual**: se aplica un descuento porcentual al precio del producto.
  - **2x1**: si se compran 2 productos iguales, se cobra solo uno.
- Las ofertas no son acumulables; si un producto es elegible para múltiples ofertas, se aplica primera registrada. 
- Las ofertas deben determinar si son aplicables para un producto y cantidad dada.
- Debe poder calcular el descuento aplicado a un producto.

#### Clientes

- Los clientes tienen un nombre, un apellido y un CUIT de 11 dígitos.

#### Factura

- La factura tiene un número secuencial, una fecha, un cliente y una lista de productos con la cantidad vendida de cada uno.
- Debe calcular el total de la venta, teniendo en cuenta las ofertas aplicadas.
- Debe generar texto para imprimir la factura con el siguiente formato:
```text
Factura: <numero>
Fecha  : <fecha>
Cliente: <nombre cliente> (<CUIT>)

- <cantidad>u <nombre producto>            x $<precio> = $<subtotal>
      <descripción oferta>                             - $<descuento>
- ...

                                             Subtotal:   $<subtotal general>
                                             Descuentos: $<total descuentos>
                                             -----------------------
                                             Total:      $<total>
```
