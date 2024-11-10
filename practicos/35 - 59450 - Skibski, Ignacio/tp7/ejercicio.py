import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Configuración de la página
st.set_page_config(page_title="Estimación de Ventas Diarias", layout="wide")

# Título de la aplicación
st.sidebar.header("Parámetros de Entrenamiento")
st.title("Estimación de Ventas Diarias")

# Parámetros de la red neuronal
learning_rate = st.sidebar.number_input("Aprendizaje", min_value=0.001, max_value=1.0, step=0.001, value=0.01)
epochs = st.sidebar.number_input("Repeticiones", min_value=100, max_value=5000, step=100, value=1000)
hidden_neurons = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, step=1, value=10)

# Datos de ventas proporcionados
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
            [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
            [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]])
y = np.array([195, 169, 172, 178, 132, 123, 151, 127, 96, 110, 
            86, 82, 94, 60, 63, 76, 69, 98, 77, 71, 
            134, 107, 120, 99, 126, 150, 136, 179, 173, 194])

# Entrenar el modelo al presionar el botón
if st.sidebar.button("Entrenar"):
    # Inicialización del modelo
    model = MLPRegressor(hidden_layer_sizes=(hidden_neurons,), max_iter=epochs, learning_rate_init=learning_rate, random_state=0)
    
    # Entrenamiento del modelo
    model.fit(X, y)
    
    # Predicciones y error
    y_pred = model.predict(X)
    error = mean_squared_error(y, y_pred)
    
    # Mostrar estado del entrenamiento
    st.sidebar.success("Entrenamiento exitoso")
    st.sidebar.write(f"Epoch {epochs}/{epochs} - Error: {error:.5f}")
    
    # Gráfico de pérdida (error)
    st.sidebar.subheader("Pérdida durante el entrenamiento")
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(model.loss_curve_, color='green', label="Pérdidas")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Pérdida")
    ax_loss.legend()
    st.sidebar.pyplot(fig_loss)
    
    # Gráfico de resultados
    st.subheader("Estimación de Ventas Diarias")
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue", label="Datos Reales")
    ax.plot(X, y_pred, color="red", label="Curva de Ajuste")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)