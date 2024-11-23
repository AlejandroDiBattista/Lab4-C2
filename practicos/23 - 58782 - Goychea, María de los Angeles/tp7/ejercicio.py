import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Estimación de Ventas ", layout="wide")

st.sidebar.header("Parámetros de Entrenamiento")
st.title("Estimación de Ventas Diarias")

learning_rate = st.sidebar.number_input("Aprendizaje", min_value=0.001, max_value=1.0, step=0.001, value=0.01)
epochs = st.sidebar.number_input("Repeticiones", min_value=100, max_value=5000, step=100, value=1000)
hidden_neurons = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, step=1, value=10)

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
            [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
            [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]])
y = np.array([195, 169, 172, 178, 132, 123, 151, 127, 96, 110, 
            86, 82, 94, 60, 63, 76, 69, 98, 77, 71, 
            134, 107, 120, 99, 126, 150, 136, 179, 173, 194])

if st.sidebar.button("Entrenar"):
    model = MLPRegressor(hidden_layer_sizes=(hidden_neurons,), max_iter=epochs, learning_rate_init=learning_rate, random_state=0)
    
    model.fit(X, y)
    
    y_pred = model.predict(X)
    error = mean_squared_error(y, y_pred)
    
    st.sidebar.success("Entrenamiento exitoso")
    st.sidebar.write(f"Epoch {epochs}/{epochs} - Error: {error:.5f}")
    
    st.sidebar.subheader("Pérdida durante el entrenamiento")
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(model.loss_curve_, color='green', label="Pérdidas")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Pérdida")
    ax_loss.legend()
    st.sidebar.pyplot(fig_loss)
    
    st.subheader("Estimación de Ventas Diarias")
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue", label="Datos Reales")
    ax.plot(X, y_pred, color="red", label="Curva de Ajuste")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)