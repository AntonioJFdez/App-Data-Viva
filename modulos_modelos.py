# modulos_modelos.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from datetime import datetime
import logging
import os
from textblob import TextBlob
from lifelines import KaplanMeierFitter

# --------- Segmentación de Clientes ----------
def segmentacion_clientes(df, columnas_segmento, n_clusters=4, output_dir="static/graficos"):
    df_segmento = df[columnas_segmento].copy()
    df_segmento.fillna(df_segmento.mean(), inplace=True)
    scaler = StandardScaler()
    datos_normalizados = scaler.fit_transform(df_segmento)

    modelo = KMeans(n_clusters=n_clusters, random_state=42)
    etiquetas = modelo.fit_predict(datos_normalizados)

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

    df['segmento'] = etiquetas
    return df, grafico_path

# --------- Scoring Leads ----------
def scoring_leads(df, target_col='convertido', output_dir="static/graficos"):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    df['score'] = modelo.predict_proba(X)[:, 1]
    df_ordenado = df.sort_values(by='score', ascending=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, "leads_score.csv")
    output_xlsx = os.path.join(output_dir, "leads_score.xlsx")
    df_ordenado.to_csv(output_csv, index=False)
    try:
        df_ordenado.to_excel(output_xlsx, index=False)
    except Exception as e:
        pass

    modelo_path = os.path.join(output_dir, "modelo_scoring_leads.pkl")
    joblib.dump(modelo, modelo_path)
    return output_csv, output_xlsx, reporte

# --------- Predicción Churn ----------
def prediccion_churn(df, columnas=None, target_col='churned', output_dir="static/graficos"):
    if columnas is None:
        columnas = ['edad', 'prima_anual', 'num_pólizas', 'score_interacción']
    if target_col not in df.columns:
        raise ValueError(f"Falta la columna objetivo '{target_col}'.")
    X = df[columnas]
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    probas = modelo.predict_proba(X_test)[:, 1]
    reporte_dict = classification_report(y_test, preds, output_dict=True)
    roc_auc = roc_auc_score(y_test, probas)
    reporte_dict['roc_auc'] = {'f1-score': roc_auc}

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

    df['score_churn'] = modelo.predict_proba(df[columnas])[:, 1]
    df.to_csv(predicciones_csv, index=False)

    modelo_path = os.path.join(output_dir, "modelo_churn.pkl")
    joblib.dump(modelo, modelo_path)

    return reporte_csv, reporte_xlsx, predicciones_csv, reporte_dict

# --------- Panel de Rendimiento de Agentes ----------
def panel_rendimiento_agentes(df, output_dir="static/graficos"):
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
    resumen['cross_sell_ratio'] = (resumen['ventas_cruzadas'] / resumen['ventas_cerradas']).replace([float('inf'), float('nan')], 0).round(2)
    resumen['tiempo_respuesta_horas'] = resumen['tiempo_respuesta_horas'].round(1)
    resumen['nps_aproximado'] = (resumen['satisfaccion_cliente'] * 10).clip(upper=100).round(0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    panel_csv = os.path.join(output_dir, "panel_agentes.csv")
    resumen.to_csv(panel_csv, index=False)
    return panel_csv, resumen

# --------- Pricing Dinámico ----------
def pricing_dinamico(df, margen_deseado=0.25, output_dir="static/graficos"):
    if not all(col in df.columns for col in ['coste', 'elasticidad', 'volumen_estimado']):
        raise ValueError("El dataset debe incluir las columnas: 'coste', 'elasticidad', 'volumen_estimado'.")
    df['precio_objetivo'] = df['coste'] * (1 + margen_deseado)
    df['precio_ajustado'] = df['precio_objetivo'] * (1 - df['elasticidad'] * 0.1)
    df['precio_ajustado'] = df['precio_ajustado'].round(2)
    df['ganancia_estimada'] = (df['precio_ajustado'] - df['coste']) * df['volumen_estimado']
    simulacion = []
    for margen in [0.10, 0.20, 0.30, 0.40]:
        temp = df.copy()
        temp['precio_simulado'] = temp['coste'] * (1 + margen)
        temp['ganancia_simulada'] = (temp['precio_simulado'] - temp['coste']) * temp['volumen_estimado']
        total = temp['ganancia_simulada'].sum()
        simulacion.append((margen, round(total, 2)))
    sim_df = pd.DataFrame(simulacion, columns=['margen', 'ganancia_total'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, "precios_optimos.csv")
    graf_path = os.path.join(output_dir, "simulacion_ganancias.png")
    df.to_csv(csv_path, index=False)
    sim_df.plot(x='margen', y='ganancia_total', kind='line', marker='o', title='Simulación de Ganancia por Margen')
    plt.xlabel('Margen (%)')
    plt.ylabel('Ganancia Total (€)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graf_path)
    plt.close()

    return csv_path, graf_path, df, sim_df

# --------- Marketing Personalizado ----------
def marketing_personalizado(df, output_dir="static/graficos"):
    hoy = datetime.today()
    df['dias_para_renovar'] = (df['fecha_renovacion'] - hoy).dt.days
    df['es_cumple'] = df['fecha_nacimiento'].apply(lambda x: x.day == hoy.day and x.month == hoy.month)
    df['siniestro_reciente'] = (hoy - df['fecha_siniestro']).dt.days <= 30
    campañas = []
    for _, row in df.iterrows():
        if row['es_cumple']:
            campañas.append({
                "nombre": row['nombre'],
                "email": row['email'],
                "asunto": "¡Feliz cumpleaños!",
                "mensaje": f"{row['nombre']}, te felicitamos y te ofrecemos una revisión gratuita de tu póliza."
            })
        elif 0 <= row['dias_para_renovar'] <= 15:
            campañas.append({
                "nombre": row['nombre'],
                "email": row['email'],
                "asunto": "Renueva tu póliza a tiempo",
                "mensaje": f"{row['nombre']}, tu póliza vence pronto. Contáctanos para renovarla con beneficios exclusivos."
            })
        elif row['siniestro_reciente']:
            campañas.append({
                "nombre": row['nombre'],
                "email": row['email'],
                "asunto": "¿Cómo estás tras tu siniestro?",
                "mensaje": f"{row['nombre']}, queremos saber cómo estás y ayudarte en lo que necesites tras tu siniestro reciente."
            })
    df_mensajes = pd.DataFrame(campañas)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "campaña_email.csv")
    df_mensajes.to_csv(output_path, index=False)
    logging.info(f"Mensajes exportados en: {output_path}")
    return df_mensajes, output_path

# --------- Fidelización Clientes ----------
def fidelizacion_clientes(df, output_dir="static/graficos"):
    hoy = datetime.today()
    df['antiguedad_años'] = (hoy - df['fecha_alta']).dt.days // 365
    df['puntos'] = (
        df['antiguedad_años'] * 10 +
        df['num_pólizas'] * 15 -
        df['siniestros_12m'] * 5
    )
    df['puntos'] = df['puntos'].apply(lambda x: max(x, 0))  # No negativos

    def asignar_segmento(puntos):
        if puntos >= 100:
            return 'Oro'
        elif puntos >= 60:
            return 'Plata'
        else:
            return 'Bronce'

    df['segmento'] = df['puntos'].apply(asignar_segmento)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "clientes_segmentados.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"Clientes exportados a: {output_path}")
    return df, output_path

# --------- Análisis Sentimiento y NPS ----------
def analisis_sentimiento_nps(df, output_dir="static/graficos"):
    if 'comentario' not in df.columns or 'nps' not in df.columns:
        raise ValueError("El dataset debe contener las columnas 'comentario' y 'nps'.")
    df['sentimiento'] = df['comentario'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['estado'] = df['sentimiento'].apply(lambda x: 'Negativo' if x < -0.1 else ('Positivo' if x > 0.1 else 'Neutral'))

    promotores = df[df['nps'] >= 9].shape[0]
    detractores = df[df['nps'] <= 6].shape[0]
    total = df.shape[0]
    nps_score = ((promotores - detractores) / total) * 100 if total > 0 else 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_out = os.path.join(output_dir, "feedback_sentimiento.csv")
    txt_out = os.path.join(output_dir, "nps_resultado.txt")
    df.to_csv(csv_out, index=False)
    with open(txt_out, 'w') as f:
        f.write(f"NPS actual: {nps_score}")
    logging.info(f"Feedback exportado: {csv_out} | NPS: {nps_score}")
    return df, nps_score, csv_out, txt_out

# --------- Análisis de Siniestros ----------
def analizar_siniestros(df, output_dir="static/graficos"):
    if not {'recepcion_dias', 'revision_dias', 'resolucion_dias'} <= set(df.columns):
        raise ValueError("El dataset debe contener las columnas: 'recepcion_dias', 'revision_dias', 'resolucion_dias'.")
    df['total_dias'] = df[['recepcion_dias', 'revision_dias', 'resolucion_dias']].sum(axis=1)
    cuellos = df[['recepcion_dias', 'revision_dias', 'resolucion_dias']].mean().sort_values(ascending=False)
    logging.info(f"Cuellos de botella: {cuellos}")

    recomendaciones = {}
    for etapa, tiempo_medio in cuellos.items():
        if tiempo_medio > 5:
            recomendaciones[etapa] = "Automatizar con RPA (ej. correos, formularios, alertas)"
        else:
            recomendaciones[etapa] = "Mantener gestión manual actual"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cuellos_out = os.path.join(output_dir, "cuellos_de_botella.csv")
    cuellos.to_csv(cuellos_out)
    recomendaciones_out = os.path.join(output_dir, "recomendaciones_rpa.csv")
    pd.DataFrame(list(recomendaciones.items()), columns=["Etapa", "Recomendacion"]).to_csv(recomendaciones_out, index=False)

    logging.info(f"Resultados exportados a: {cuellos_out} y {recomendaciones_out}")
    return df, cuellos, recomendaciones

# --------- Priorizar Clientes Dormidos ----------
def priorizar_clientes_dormidos(df, output_dir="static/graficos"):
    columnas = ['cliente_id', 'nombre', 'meses_ultimo_contacto', 'prima_media', 'num_productos']
    if not all(col in df.columns for col in columnas):
        raise ValueError(f"El dataset debe tener las columnas: {columnas}")
    df['meses_inactivos'] = df['meses_ultimo_contacto']
    df['valor'] = df['prima_media'] * df['num_productos']
    df['score_reactivacion'] = df['meses_inactivos'] * df['valor']
    df_ordenado = df.sort_values(by='score_reactivacion', ascending=False)
    fecha = datetime.now().strftime('%Y%m%d')
    filename = f"{output_dir}/clientes_priorizados_{fecha}.xlsx"
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        columnas_exportar = ['cliente_id', 'nombre', 'score_reactivacion', 'meses_inactivos', 'valor']
        df_ordenado[columnas_exportar].to_excel(filename, index=False)
        logging.info(f"Archivo de clientes priorizados exportado: {filename}")
    except Exception as e:
        logging.error(f"Error al exportar archivo: {e}")
        raise
    return df_ordenado, filename

# --------- Digitalizar Renovaciones ----------
def digitalizar_renovaciones(df, dias_ventana=30, output_dir="static/graficos"):
    if 'fecha_vencimiento' not in df.columns or 'nombre' not in df.columns or 'numero_poliza' not in df.columns or 'email' not in df.columns:
        raise ValueError("El dataset debe tener las columnas: fecha_vencimiento, nombre, numero_poliza, email")
    df['fecha_vencimiento'] = pd.to_datetime(df['fecha_vencimiento'])
    hoy = datetime.today()
    df['dias_para_vencer'] = (df['fecha_vencimiento'] - hoy).dt.days
    renovaciones = df[(df['dias_para_vencer'] >= 0) & (df['dias_para_vencer'] <= dias_ventana)].copy()
    renovaciones['mensaje'] = (
        "Hola " + renovaciones['nombre'] +
        ", tu póliza con número " + renovaciones['numero_poliza'].astype(str) +
        " vence el " + renovaciones['fecha_vencimiento'].dt.strftime('%d/%m/%Y') +
        ". Por favor contáctanos para renovarla a tiempo."
    )
    filename = f"{output_dir}/recordatorios_renovacion_{datetime.now().strftime('%Y%m%d')}.csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    renovaciones[['email', 'nombre', 'mensaje']].to_csv(filename, index=False)
    logging.info(f"Recordatorios de renovación exportados a {filename}")
    return renovaciones[['email', 'nombre', 'mensaje']], filename

# --------- Panel Integral KPIs ----------
def calcular_panel_kpis(df, output_dir="static/graficos"):
    columnas = ['prima_emitida', 'poliza_id', 'cliente_id', 'estado', 'renovada', 'es_renovable',
                'nps', 'siniestros_pagados', 'producto', 'comision']
    if not all(col in df.columns for col in columnas):
        raise ValueError(f"El dataset debe contener las columnas: {columnas}")

    resumen = {}
    resumen['ventas_totales'] = df['prima_emitida'].sum()
    resumen['numero_polizas'] = df['poliza_id'].nunique()
    resumen['clientes_activos'] = df[df['estado'] == 'activa']['cliente_id'].nunique()
    resumen['tasa_renovacion'] = round(
        df[df['renovada'] == 1].shape[0] / max(1, df[df['es_renovable'] == 1].shape[0]) * 100, 2
    )
    resumen['nps_promedio'] = df['nps'].mean().round(2)
    resumen['siniestralidad'] = round(
        df['siniestros_pagados'].sum() / max(1, df['prima_emitida'].sum()) * 100, 2
    )
    resumen['productos_por_cliente'] = round(
        df.groupby('cliente_id')['producto'].nunique().mean(), 2
    )
    resumen['comision_media'] = round(df['comision'].mean(), 2)

    panel_kpi = pd.DataFrame([resumen])
    filename = f"{output_dir}/panel_kpis_integral.csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    panel_kpi.to_csv(filename, index=False)
    logging.info(f"Panel de KPIs exportado a {filename}")
    return panel_kpi, filename

# --------- Matriz Perfil-Producto y Correlaciones ----------
def analizar_perfiles_productos(df):
    matriz = pd.crosstab(index=df['perfil_cliente'], columns=df['producto'])
    correlaciones = matriz.corr()
    fecha = datetime.now().strftime('%Y%m%d_%H%M%S')
    archivo_matriz = f"static/graficos/matriz_perfil_producto_{fecha}.xlsx"
    archivo_corr = f"static/graficos/correlaciones_producto_{fecha}.xlsx"
    matriz.to_excel(archivo_matriz)
    correlaciones.to_excel(archivo_corr)
    return archivo_matriz, archivo_corr

# --------- Detectar Cross-Sell ----------
def detectar_cross_sell(df):
    matriz = df.pivot_table(index='cliente_id', columns='producto', aggfunc='size', fill_value=0)
    matriz['num_productos'] = matriz.sum(axis=1)
    media_productos = matriz['num_productos'].mean()
    candidatos = matriz[matriz['num_productos'] < media_productos].reset_index()
    fecha = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre_archivo = f"static/graficos/cross_sell_{fecha}.xlsx"
    candidatos.to_excel(nombre_archivo, index=False)
    return nombre_archivo, len(candidatos)

# --------- Comparar Precios Benchmarking (GENÉRICO) ----------
def comparar_precios(df_propios, df_competencia):
    df = df_propios.merge(df_competencia, on='producto', suffixes=('_propio', '_competencia'))
    df['diferencia_absoluta'] = df['precio_propio'] - df['precio_competencia']
    df['diferencia_relativa'] = ((df['precio_propio'] - df['precio_competencia']) / df['precio_competencia']).round(3)
    fecha = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre_archivo = f"static/graficos/benchmarking_precios_{fecha}.xlsx"
    df.to_excel(nombre_archivo, index=False)
    return nombre_archivo, len(df)

# --------- Calcular LTV Clientes ----------
def calcular_ltv(df):
    df['fecha_compra'] = pd.to_datetime(df['fecha_compra'])
    ultimo_dia = df['fecha_compra'].max()
    agrupado = df.groupby('cliente_id').agg({
        'fecha_compra': [np.min, np.max, 'count'],
        'importe': 'sum'
    })
    agrupado.columns = ['primera_compra', 'ultima_compra', 'num_compras', 'total_gastado']
    agrupado['duracion'] = (ultimo_dia - agrupado['primera_compra']).dt.days
    agrupado['observado'] = (agrupado['ultima_compra'] == ultimo_dia).astype(int)

    kmf = KaplanMeierFitter()
    kmf.fit(agrupado['duracion'], event_observed=1 - agrupado['observado'])

    agrupado['valor_medio'] = agrupado['total_gastado'] / agrupado['num_compras']
    agrupado['ltv_estimado'] = agrupado['valor_medio'] * agrupado['num_compras'] * 1.5
    agrupado['segmento'] = pd.qcut(agrupado['ltv_estimado'], 3, labels=['Bajo', 'Medio', 'Alto'])

    fecha = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre_archivo = f"static/graficos/ltv_clientes_{fecha}.xlsx"
    agrupado.reset_index().to_excel(nombre_archivo, index=False)
    return nombre_archivo, agrupado.shape[0]

