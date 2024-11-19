import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://trabajopractico.streamlit.app/'


st.set_page_config(layout="wide")

st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .product-title {
        color: #333;
        padding: 0 0 15px 0;
        margin: 0 0 15px 0;
        border-bottom: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

if st.session_state.get('archivo_cargado', False) is False:
    st.markdown("### Por favor, sube un archivo CSV desde la barra lateral.")

    def mostrar_informacion_alumno():
        with st.container():
            st.markdown(
                """
                <div style="border: 2px solid #D3D3D3; padding: 10px; border-radius: 5px; background-color: #FFFFFF;">
                    <p><strong>Legajo:</strong> 58.782</p>
                    <p><strong>Nombre:</strong> Goyechea Maria de los Angeles</p>
                    <p><strong>Comisión:</strong> C2</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    mostrar_informacion_alumno()

st.sidebar.header("Cargar archivo de datos")
archivo = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

def crear_grafico_ventas(datos_producto):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.grid(True, linestyle='-', alpha=0.7, color='#E0E0E0')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    ax.plot(datos_producto['Fecha'], datos_producto['Unidades_vendidas'], 
            color='blue', marker='o', markersize=4, label='Producto', linewidth=2)
    
    x = np.arange(len(datos_producto))
    z = np.polyfit(x, datos_producto['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(datos_producto['Fecha'], p(x), 
            color='red', linestyle='--', label='Tendencia', linewidth=2)
    
    ax.set_title("Evolución de Ventas Mensual", pad=20, fontsize=12)
    ax.set_xlabel("Fecha", labelpad=10)
    ax.set_ylabel("Unidades Vendidas", labelpad=10)
    
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    
    plt.xticks(rotation=45)
    
    return fig

if archivo is not None:
    st.session_state.archivo_cargado = True
    
    try:
        datos = pd.read_csv(archivo)
        
        if not set(['Sucursal', 'Producto', 'Año', 'Mes', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total']).issubset(datos.columns):
            st.error("El archivo CSV debe contener las columnas necesarias: 'Sucursal', 'Producto', 'Año', 'Mes', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total'.")
        else:
            datos['Año'] = pd.to_numeric(datos['Año'], errors='coerce')
            datos['Mes'] = pd.to_numeric(datos['Mes'], errors='coerce')
            
            datos = datos.dropna(subset=['Año', 'Mes'])
            datos = datos[(datos['Mes'] >= 1) & (datos['Mes'] <= 12)]
            datos = datos[(datos['Año'] >= 1900) & (datos['Año'] <= 2100)]
            
            if datos.empty:
                st.error("No hay datos válidos para generar los gráficos. Por favor, verifica el archivo CSV.")
            else:
                sucursales = ["Todas"] + list(datos['Sucursal'].unique())
                sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

                st.header(f"Datos de {sucursal_seleccionada if sucursal_seleccionada != 'Todas' else 'Todas las Sucursales'}")

                if sucursal_seleccionada != "Todas":
                    datos = datos[datos['Sucursal'] == sucursal_seleccionada]

                resumen_productos = datos.groupby('Producto').agg({
                    'Unidades_vendidas': 'sum',
                    'Ingreso_total': 'sum',
                    'Costo_total': 'sum'
                }).reset_index()

                resumen_productos['Precio_Promedio'] = resumen_productos['Ingreso_total'] / resumen_productos['Unidades_vendidas']
                resumen_productos['Margen_Promedio'] = (resumen_productos['Ingreso_total'] - resumen_productos['Costo_total']) / resumen_productos['Ingreso_total']

                for i, row in resumen_productos.iterrows():
                    producto = row['Producto']
                    
                    st.markdown(f'<h3 class="product-title">{producto}</h3>', unsafe_allow_html=True)
                    
                    col_metricas, col_grafico = st.columns([1, 3])
                    
                    with col_metricas:
                        st.metric(
                            "Precio Promedio", 
                            f"${row['Precio_Promedio']:.3f}", 
                            "-28.57%"
                        )
                        st.metric(
                            "Margen Promedio", 
                            f"{row['Margen_Promedio']:.0%}", 
                            "-0.25%"
                        )
                        st.metric(
                            "Unidades Vendidas", 
                            f"{int(row['Unidades_vendidas']):,}", 
                            "9.98%"
                        )

                    with col_grafico:
                        datos_producto = datos[datos['Producto'] == producto].copy()
                        datos_producto['Fecha'] = pd.to_datetime({
                            'year': datos_producto['Año'],
                            'month': datos_producto['Mes'],
                            'day': 1
                        })
                        datos_producto = datos_producto.sort_values('Fecha')

                        if not datos_producto.empty:
                            fig = crear_grafico_ventas(datos_producto)
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    st.markdown("<br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
