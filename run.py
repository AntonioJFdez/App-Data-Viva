# ejecutar.py (App Flask modular con múltiples funcionalidades)
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)
app.secret_key = 'secret'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Módulo 1: Carga y perfilado de datasets
@app.route('/perfilado', methods=['GET', 'POST'])
def perfilado():
    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)
            df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
            profile = ProfileReport(df, title='Informe de Perfilado de Datos', explorative=True)
            profile_path = os.path.join(UPLOAD_FOLDER, 'informe_perfilado.html')
            profile.to_file(profile_path)
            return redirect(url_for('uploaded_file', filename='informe_perfilado.html'))
    return render_template('perfilado.html')

# Módulo 2: Dashboard básico
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    graphs = []
    if request.method == 'POST':
        file = request.files['dashboard_data']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)
            df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
            for col in df.select_dtypes(include='number').columns:
                fig = px.histogram(df, x=col, title=f'Distribución de {col}')
                graphs.append(fig.to_html(full_html=False))
    return render_template('dashboard.html', graphs=graphs)

# Módulo 3: Predicción automática de KPIs
@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    resultado = ''
    if request.method == 'POST':
        file = request.files['data_kpi']
        target = request.form['target']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)
            df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
            df.dropna(inplace=True)
            if df[target].dtype == 'O':
                le = LabelEncoder()
                df[target] = le.fit_transform(df[target])
            X = df.drop(columns=[target])
            y = df[target]
            X = pd.get_dummies(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            resultado = f'Predicción realizada. Error cuadrático medio: {mse:.2f}'
    return render_template('prediccion.html', resultado=resultado)

# Módulo 4: Health Checker
@app.route('/healthcheck', methods=['GET', 'POST'])
def healthcheck():
    resultados = []
    if request.method == 'POST':
        file = request.files['health_data']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)
            df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
            if df.isnull().mean().max() > 0.2:
                resultados.append("Más del 20% de valores nulos en alguna columna.")
            for col in df.columns:
                if df[col].nunique() == 1:
                    resultados.append(f"Columna {col} tiene un único valor.")
                if df[col].nunique() > len(df) * 0.9:
                    resultados.append(f"Columna {col} tiene cardinalidad muy alta.")
    return render_template('healthcheck.html', resultados=resultados)

# Módulo 5: Smart Filter Recommender
@app.route('/smartfilter', methods=['GET', 'POST'])
def smartfilter():
    recomendaciones = []
    if request.method == 'POST':
        file = request.files['filter_data']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)
            df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
            df = df.select_dtypes(include='number').dropna()
            kmeans = KMeans(n_clusters=3)
            df['cluster'] = kmeans.fit_predict(df)
            recomendaciones = df.groupby('cluster').mean().to_dict()
    return render_template('smartfilter.html', recomendaciones=recomendaciones)

# Módulo 6: Comparador visual
@app.route('/comparador', methods=['GET', 'POST'])
def comparador():
    result_image = None
    if request.method == 'POST':
        image1 = request.files['image1']
        image2 = request.files['image2']
        if image1 and image2:
            img1 = Image.open(image1.stream).convert("RGBA")
            img2 = Image.open(image2.stream).convert("RGBA")
            img1 = img1.resize((600, 400))
            img2 = img2.resize((600, 400))
            blended = Image.blend(img1, img2, alpha=0.5)
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
            blended.save(result_path)
            result_image = 'result.png'
    return render_template('visual_compare.html', result_image=result_image)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return render_template('index.html')

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)



