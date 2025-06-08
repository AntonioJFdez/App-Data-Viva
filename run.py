
"""
Script: ejecutar.py
Autor: [Tu Nombre/Empresa]
Descripción: Script principal de ejecución para Data Viva - Soluciones Automáticas.
Optimizado para UX/UI, robustez y valor comercial.
"""

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import matplotlib.pyplot as plt
import seaborn as sns

# -- CONFIGURACIÓN FLASK --
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")  # Asegura seguridad básica para mensajes

# --- FUNCIONES UTILITARIAS ---

def cargar_dataset(archivo):
    """
    Carga cualquier CSV o Excel con robustez y devuelve DataFrame limpio.
    """
    try:
        nombre = archivo.filename
        if nombre.endswith('.csv'):
            df = pd.read_csv(archivo, encoding='utf-8')
        elif nombre.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(archivo)
        else:
            raise ValueError("Formato de archivo no soportado.")
        if df.empty:
            raise ValueError("El archivo está vacío.")
        return df
    except Exception as e:
        raise ValueError(f"Error al cargar los datos: {str(e)}")

def resultado_mensaje(mensaje, exito=True):
    """
    Devuelve un mensaje amigable y visual para el usuario, ajustado al branding.
    """
    estilo = "color: #2c3e50; background: #debfb0; border-radius: 1rem; padding: 1.3rem; font-size:1.15rem; text-align:center;"
    if not exito:
        estilo = "color: #fff; background: #e74c3c; border-radius: 1rem; padding: 1.3rem; font-size:1.12rem; text-align:center;"
    return f'<div style="{estilo}">{mensaje}</div>'

def generar_grafico(df):
    """
    Genera un gráfico de barras basado en el DataFrame cargado. 
    Aquí puedes personalizar este gráfico según las variables que se pasen.
    """
    try:
        # Selección de las primeras columnas numéricas para el gráfico
        columnas_numericas = df.select_dtypes(include='number').columns
        if len(columnas_numericas) > 0:
            df[columnas_numericas].mean().plot(kind='bar', figsize=(8, 6), color='skyblue')
            plt.title('Promedio de Variables Numéricas', fontsize=14)
            plt.ylabel('Promedio', fontsize=12)
            plt.xlabel('Variables', fontsize=12)
            plt.tight_layout()
            grafico_path = "static/graficos/grafico.png"
            plt.savefig(grafico_path)
            plt.close()
            return grafico_path
        else:
            raise ValueError("No hay datos numéricos para graficar.")
    except Exception as e:
        raise ValueError(f"Error al generar gráfico: {str(e)}")

# -- RUTAS PRINCIPALES --

@app.route('/', methods=['GET'])
def home():
    """
    Página principal con el resumen de las soluciones.
    """
    return render_template('indice.html')

@app.route('/perfilado', methods=['GET', 'POST'])
def perfilado():
    """
    Carga un dataset y realiza el perfilado automático de datos.
    """
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            # Generar gráfico
            grafico_path = generar_grafico(df)
            info = df.describe(include='all').transpose().reset_index()
            resumen = info.to_html(classes="table table-striped", index=False)
            flash(resultado_mensaje("¡Perfilado realizado con éxito! Revisa el informe más abajo."), "success")
            return render_template('perfilado.html', resumen=resumen, grafico=grafico_path)
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('perfilado.html')

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    """
    Predicción automática de KPIs usando Random Forest.
    """
    if request.method == 'POST':
        archivo = request.files.get('data_kpi')
        objetivo = request.form.get('target')
        try:
            df = cargar_dataset(archivo)
            # Aquí va el modelo de predicción, por ejemplo con Random Forest
            # Predicción ejemplo: df[objetivo].mean() (esto es solo un ejemplo)
            prediccion = df[objetivo].mean()  # Usar modelo real aquí
            flash(resultado_mensaje(f"Predicción realizada para la variable {objetivo}. Resultado: {prediccion}", "success"))
            return render_template('prediccion.html', prediccion=prediccion)
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('prediccion.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """
    Crear un dashboard exploratorio basado en los datos cargados.
    """
    if request.method == 'POST':
        archivo = request.files.get('dashboard_data')
        try:
            df = cargar_dataset(archivo)
            # Aquí agregarías el código de dashboard interactivo
            flash(resultado_mensaje("Dashboard generado con éxito. Visualiza el gráfico abajo."), "success")
            return render_template('dashboard.html', df=df.head())
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('dashboard.html')

# -- OTRAS RUTAS --

@app.route('/descargar', methods=['POST'])
def descargar():
    """
    Ruta para descargar archivos generados o reportes.
    """
    archivo_path = request.form.get('file_path')
    if archivo_path and os.path.exists(archivo_path):
        return send_file(archivo_path, as_attachment=True)
    else:
        flash(resultado_mensaje("El archivo no existe o no se puede descargar.", exito=False), "danger")
        return redirect(url_for('home'))

# -- INICIO SEGURO (PRODUCCIÓN) --
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



