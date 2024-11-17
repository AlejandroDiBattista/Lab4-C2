import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# URL de la aplicación
# url = 'https://tp8-47121.streamlit.app/'


st.set_page_config(layout="wide")

st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid black;
        margin: 1rem 0;
    }
    .panel-datos {
        border: 2px solid black;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f0f0f0;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown('**Cargar archivos de datos**')
st.sidebar.markdown('<br>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="margin-bottom: -3rem;">Subir archivo CSV</p>', unsafe_allow_html=True)
archivoCsv = st.sidebar.file_uploader("", type=["csv"])

def mostrarInformacionAlumno():
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
    st.markdown("""
        <div class="panel-datos">
            <p><strong>Legajo:</strong> 47121</p>
            <p><strong>Apellido y Nombre:</strong> Caram Jesús Nicolás</p>
            <p><strong>Comisión:</strong> 2</p>
        </div>
    """, unsafe_allow_html=True)

if archivoCsv is None:
    mostrarInformacionAlumno()

@st.cache_data
def cargarDatos(archivoCsv):
    return pd.read_csv(archivoCsv)

def calcularMetricasPorPeriodo(df):
    df['Precio promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']
    df['Fecha'] = pd.to_datetime(df[['Año', 'Mes']].astype(str).agg('-'.join, axis=1), format='%Y-%m')
    
    metricasPeriodo = df.groupby(['Producto', 'Fecha']).agg(
        Ingreso_total=('Ingreso_total', 'sum'),
        Costo_total=('Costo_total', 'sum'),
        Unidades_vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()
    
    metricasPeriodo['Precio_promedio'] = metricasPeriodo['Ingreso_total'] / metricasPeriodo['Unidades_vendidas']
    metricasPeriodo['Margen_promedio'] = (metricasPeriodo['Ingreso_total'] - metricasPeriodo['Costo_total']) / metricasPeriodo['Ingreso_total']
    
    return metricasPeriodo

def calcularMetricasActuales(df):
    return df.groupby('Producto').agg(
        Precio_promedio=('Precio promedio', 'mean'),
        Margen_promedio=('Margen promedio', 'mean'),
        Unidades_vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()

def calcularVariacionPorcentual(valorActual, valorAnterior):
    if valorAnterior == 0:
        return 0
    return ((valorActual - valorAnterior) / valorAnterior) * 100

def calcularVariacionPromedio(metricasProducto, columna):
    cambiosPorcentuales = []
    for i in range(1, len(metricasProducto)):
        valorActual = metricasProducto.iloc[i][columna]
        valorAnterior = metricasProducto.iloc[i - 1][columna]
        variacion = calcularVariacionPorcentual(valorActual, valorAnterior)
        cambiosPorcentuales.append(variacion)
    
    return np.mean(cambiosPorcentuales) if cambiosPorcentuales else None

def filtrarPorSucursal(df, sucursal):
    if sucursal != 'Todas':
        return df[df['Sucursal'] == sucursal]
    return df

if archivoCsv is not None:
    df = cargarDatos(archivoCsv)
    sucursales = ['Todas'] + df['Sucursal'].unique().tolist()
    sucursalSeleccionada = st.sidebar.selectbox("Seleccionar una sucursal", sucursales)
    dfFiltrado = filtrarPorSucursal(df, sucursalSeleccionada)
    metricasPeriodo = calcularMetricasPorPeriodo(dfFiltrado)
    resumen = calcularMetricasActuales(dfFiltrado)

    def ordenarProducto(producto):
        letras = ''.join([c for c in producto if c.isalpha()])
        numeros = ''.join([c for c in producto if c.isdigit()])
        return (letras, int(numeros) if numeros else 0)

    productosOrdenados = sorted(resumen['Producto'], key=ordenarProducto)
    st.title("Datos de Todas las Sucursales")

    for producto in productosOrdenados:
        dfProducto = dfFiltrado[dfFiltrado['Producto'] == producto]
        metricasProducto = metricasPeriodo[metricasPeriodo['Producto'] == producto].sort_values('Fecha')

        if len(metricasProducto) >= 2:
            variacionPrecio = calcularVariacionPromedio(metricasProducto, 'Precio_promedio')
            variacionMargen = calcularVariacionPromedio(metricasProducto, 'Margen_promedio')
            variacionUnidades = calcularVariacionPromedio(metricasProducto, 'Unidades_vendidas')
        else:
            variacionPrecio = variacionMargen = variacionUnidades = None

        valoresActuales = resumen[resumen['Producto'] == producto].iloc[0]

        def colorVariacion(variacion):
            if variacion is None:
                return "gray"
            return "#28a745" if variacion > 0 else "red"

        def formatearNumeroPrecio(numero):
            return f"${int(numero):,}".replace(",", "X").replace(".", ",").replace("X", ".")

        def formatearNumeroUnidades(numero):
            return f"{numero:,.0f}".replace(",", ".")

        def formatearVariacion(variacion):
            if variacion is None:
                return "Sin datos suficientes"
            return f"{'↑' if variacion > 0 else '↓'} {abs(variacion):.2f}%"
        
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader(f"{producto}")
            st.markdown(f"""
                <p>Precio Promedio:
                    <h3>{formatearNumeroPrecio(valoresActuales['Precio_promedio'])}</h3>
                    <p style="color: {colorVariacion(variacionPrecio)}">
                        {formatearVariacion(variacionPrecio)}
                    </p>
                </p>

                <p>Margen Promedio:
                    <h3>{valoresActuales['Margen_promedio'] * 100:.0f}%</h3>
                    <p style="color: {colorVariacion(variacionMargen)}">
                        {formatearVariacion(variacionMargen)}
                    </p>
                </p>

                <p>Unidades Vendidas:
                    <h3>{formatearNumeroUnidades(valoresActuales['Unidades_vendidas'])}</h3>
                    <p style="color: {colorVariacion(variacionUnidades)}">
                        {formatearVariacion(variacionUnidades)}
                    </p>
                </p>
                """, unsafe_allow_html=True)

        with col2:
            dfProducto['Fecha'] = pd.to_datetime(
                dfProducto[['Año', 'Mes']].astype(str).agg('-'.join, axis=1), 
                format='%Y-%m'
            )
            ventasMensualesProducto = dfProducto.groupby('Fecha')['Unidades_vendidas'].sum()

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ventasMensualesProducto.index, ventasMensualesProducto.values, label='Unidades vendidas', color='blue')

            X = np.array([i for i in range(len(ventasMensualesProducto))]).reshape(-1, 1)
            y = ventasMensualesProducto.values
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            ax.plot(ventasMensualesProducto.index, y_pred, label='Tendencia', color='red', linestyle='--')
            
            ax.set_xlabel("Año")
            ax.set_ylabel("Unidades vendidas")
            ax.set_title(f"Evolución de ventas de {producto}")
            ax.grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.5)
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=ventasMensualesProducto.index.min(), right=ventasMensualesProducto.index.max())
            
            for fecha in ventasMensualesProducto.index:
                if fecha.month == 1:
                    ax.axvline(x=fecha, color='gray', linestyle='--', linewidth=1)
                ax.axvline(x=fecha, color='lightgray', linestyle='--', linewidth=0.5)

            ax.legend()
            ax.set_xticks(pd.to_datetime([str(year) + '-01-01' for year in ventasMensualesProducto.index.year.unique()]))
            ax.set_xticklabels([str(year) for year in ventasMensualesProducto.index.year.unique()])
            st.pyplot(fig)
            plt.close(fig)