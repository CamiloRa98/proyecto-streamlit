# -*- coding: utf-8 -*-
"""
App Streamlit para predicción de ventas de pantalones STOP JEANS
"""

# ========================
# Importar librerías
# ========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# ========================
# Cargar modelo entrenado
# ========================
with open("modelo-TimeSeries.pkl", "rb") as file:
    forecaster_cargado, data_cargada = pickle.load(file)

# ========================
# Interfaz de la aplicación
# ========================
st.title("📈 Estimación de ventas mensuales de pantalón - STOP JEANS")
st.markdown("Esta aplicación permite estimar la **Cantidad** de ventas futuras.")

# ========================
# Inputs del usuario
# ========================
n_periodos = st.slider(
    "Selecciona cuántos meses deseas pronosticar:",
    min_value=1, max_value=24, value=6
)

# ========================
# Botón para generar predicción
# ========================
if st.button("Generar pronóstico"):
    # Generar predicciones
    y_pred = forecaster_cargado.predict(steps=n_periodos)

    # Crear DataFrame con resultados
    resultado = pd.DataFrame({
        "Fecha": y_pred.index,
        "Pronostico_Cantidad": y_pred.values.round(0).astype(int)  # sin decimales
    })

    # Mostrar tabla
    st.subheader("📊 Resultados del pronóstico")
    st.dataframe(resultado)

    # Graficar resultados
    st.line_chart(resultado.set_index("Fecha"))
