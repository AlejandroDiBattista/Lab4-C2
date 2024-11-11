import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Configuración de la página
st.set_page_config(page_title="Estimación de Ventas Diarias", layout="wide")

# Título de la aplicación
st.sidebar.header("Parámetros de Entrenamiento")
st.title("Estimación de Ventas Diarias")

# Parámetros de la red neuronal
learning_rate = st.sidebar.number_input("Aprendizaje", min_value=0.001, max_value=1.0, step=0.001, value=0.01)
epochs = st.sidebar.number_input("Repeticiones", min_value=100, max_value=2000, step=100, value=500)
hidden_neurons = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=50, step=1, value=10)

# Datos de ventas proporcionados
X = np.array([[i] for i in range(1, 31)])
y = np.array([195, 169, 172, 178, 132, 123, 151, 127, 96, 110, 
            86, 82, 94, 60, 63, 76, 69, 98, 77, 71, 
            134, 107, 120, 99, 126, 150, 136, 179, 173, 194])

# Función para entrenar el modelo, cacheada para evitar recalcular si no cambian los parámetros
@st.cache_data
def train_model(hidden_neurons, epochs, learning_rate):
    model = MLPRegressor(hidden_layer_sizes=(hidden_neurons,), 
                        max_iter=epochs, 
                        learning_rate_init=learning_rate, 
                        random_state=0)
    model.fit(X, y)
    y_pred = model.predict(X)
    error = mean_squared_error(y, y_pred)
    return model, y_pred, error

# Entrenar el modelo al presionar el botón
if st.sidebar.button("Entrenar"):
    model, y_pred, error = train_model(hidden_neurons, epochs, learning_rate)

    # Mostrar estado del entrenamiento
    st.sidebar.success("Entrenamiento exitoso")
    st.sidebar.write(f"Error: {error:.5f}")

    # Gráfico de pérdida (error) con Plotly
    if hasattr(model, "loss_curve_"):
        st.sidebar.subheader("Pérdida durante el entrenamiento")
        loss_curve = go.Figure()
        loss_curve.add_trace(go.Scatter(y=model.loss_curve_, mode='lines', name="Pérdidas", line=dict(color='green')))
        loss_curve.update_layout(xaxis_title="Época", yaxis_title="Pérdida")
        st.sidebar.plotly_chart(loss_curve)

    # Gráfico de resultados con Plotly
    st.subheader("Estimación de Ventas Diarias")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name="Datos Reales", marker=dict(color="blue")))
    fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode='lines', name="Curva de Ajuste", line=dict(color="red")))
    fig.update_layout(xaxis_title="Día del Mes", yaxis_title="Ventas")
    st.plotly_chart(fig)



