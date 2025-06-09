# modulos_modelos.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

def segmentacion_clientes(df, columnas_segmento, n_clusters=4, output_dir="static/graficos"):
    # Preprocesamiento
    df_segmento = df[columnas_segmento].copy()
    df_segmento.fillna(df_segmento.mean(), inplace=True)
    scaler = StandardScaler()
    datos_normalizados = scaler.fit_transform(df_segmento)

    # KMeans clustering
    modelo = KMeans(n_clusters=n_clusters, random_state=42)
    etiquetas = modelo.fit_predict(datos_normalizados)

    # Visualización con PCA
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(datos_normalizados)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    grafico_path = os.path.join(output_dir, "segmentacion_clientes.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(componentes[:, 0], componentes[:, 1], c=etiquetas, cmap='viridis', alpha=0.6)
    plt.title("Segmentación de Clientes - Clustering K-Means")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(grafico_path)
    plt.close()

    # Devuelve el DataFrame original con segmento y la ruta del gráfico generado
    df['segmento'] = etiquetas
    return df, grafico_path

# modulos_modelos.py (scoring_leads)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def scoring_leads(df, target_col='convertido', output_dir="static/graficos"):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)

    # Aplicar scoring a todo el DataFrame
    df['score'] = modelo.predict_proba(X)[:, 1]
    df_ordenado = df.sort_values(by='score', ascending=False)

    # Exportar resultados
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, "leads_score.csv")
    output_xlsx = os.path.join(output_dir, "leads_score.xlsx")
    df_ordenado.to_csv(output_csv, index=False)
    try:
        df_ordenado.to_excel(output_xlsx, index=False)
    except Exception as e:
        pass

    # Guardar modelo
    modelo_path = os.path.join(output_dir, "modelo_scoring_leads.pkl")
    joblib.dump(modelo, modelo_path)
    return output_csv, output_xlsx, reporte

    # modulos_modelos.py (añade después del scoring_leads)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def prediccion_churn(df, columnas=None, target_col='churned', output_dir="static/graficos"):
    """
    Entrena un modelo para predecir churn de clientes, exporta reporte y resultados.
    """
    if columnas is None:
        # Por defecto columnas típicas para churn
        columnas = ['edad', 'prima_anual', 'num_pólizas', 'score_interacción']
    if target_col not in df.columns:
        raise ValueError(f"Falta la columna objetivo '{target_col}'.")

    X = df[columnas]
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)
    probas = modelo.predict_proba(X_test)[:, 1]
    reporte_dict = classification_report(y_test, preds, output_dict=True)
    roc_auc = roc_auc_score(y_test, probas)
    reporte_dict['roc_auc'] = {'f1-score': roc_auc}

    # Guardar reporte y scoring
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reporte_csv = os.path.join(output_dir, "reporte_churn.csv")
    reporte_xlsx = os.path.join(output_dir, "reporte_churn.xlsx")
    predicciones_csv = os.path.join(output_dir, "clientes_churn_score.csv")
    pd.DataFrame(reporte_dict).transpose().to_csv(reporte_csv)
    try:
        pd.DataFrame(reporte_dict).transpose().to_excel(reporte_xlsx)
    except Exception as e:
        pass

    # Añade score de churn a todo el DataFrame
    df['score_churn'] = modelo.predict_proba(df[columnas])[:, 1]
    df.to_csv(predicciones_csv, index=False)

    # Guardar modelo
    modelo_path = os.path.join(output_dir, "modelo_churn.pkl")
    joblib.dump(modelo, modelo_path)

    return reporte_csv, reporte_xlsx, predicciones_csv, reporte_dict

    # modulos_modelos.py (añadir después de prediccion_churn)

import pandas as pd
import os
import logging

def panel_rendimiento_agentes(df, output_dir="static/graficos"):
    """
    Calcula KPIs y genera el panel de rendimiento de agentes.
    Devuelve el path del panel exportado.
    """
    if 'agente' not in df.columns:
        raise ValueError("El dataset debe tener una columna 'agente'.")

    resumen = df.groupby('agente').agg({
        'clientes_contactados': 'sum',
        'ventas_cerradas': 'sum',
        'ventas_cruzadas': 'sum',
        'satisfaccion_cliente': 'mean',
        'tiempo_respuesta_horas': 'mean'
    }).reset_index()

    resumen['conversion_rate'] = (resumen['ventas_cerradas'] / resumen['clientes_contactados']).round(2)
    resumen['cross_sell_ratio'] = (
        resumen['ventas_cruzadas'] / resumen['ventas_cerradas']
    ).replace([float('inf'), float('nan')], 0).round(2)
    resumen['tiempo_respuesta_horas'] = resumen['tiempo_respuesta_horas'].round(1)
    resumen['nps_aproximado'] = (resumen['satisfaccion_cliente'] * 10).clip(upper=100).round(0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    panel_csv = os.path.join(output_dir, "panel_agentes.csv")
    resumen.to_csv(panel_csv, index=False)

    return panel_csv, resumen


