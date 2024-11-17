# TP8: 2do Parcial

El trabajo práctico 8 actúa como segundo parcial.

El trabajo es personal y debe ser desarrollado de forma individual.
De no cumplirse este requisito, se calificará con 0 (cero) el trabajo práctico.

> Fecha de entrega:
>
> 21 de noviembre de 2024 hasta las 21:00 hs

## Enunciado

El trabajo consiste en realizar una aplicación con Streamlit que permita cargar datos de ventas y mostrarlos.

Los datos se encuentran en un archivo CSV con el siguiente formato:

```
Sucursal,Producto,Año,Mes,Unidades_vendidas,Ingreso_total,Costo_total
```

Una vez cargados los mismos, debe mostrarse para cada producto la siguiente información:
- Precio promedio (Ingreso total / Unidades vendidas)
- Margen promedio ((Ingreso total - Costo total) / Ingreso total)
- Unidades vendidas (Suma de unidades vendidas)

A su vez, se debe poder visualizar en un gráfico la evolución de la venta a lo largo de los meses.
Este gráfico debe incluir una línea de tendencia.

También debe permitir elegir si se muestran todos los datos o los de una sucursal en particular.

Por último, el programa debe ser publicado para que pueda ser accedido desde cualquier navegador.

![Pantalla Inicial](./practicos/enunciados/pantalla0.png)
![Pantalla Principal](./practicos/enunciados/pantalla1.png)

<video width="600" controls>
  <source src="./practicos/enunciados/video.mp4" type="video/mp4">
  Tu navegador no soporta la etiqueta de video.
</video>

> **Nota**: 
> Recomiendo que suban el trabajo a partir de las 20:00 hs.
