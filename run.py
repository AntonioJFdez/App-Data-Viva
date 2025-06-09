import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import matplotlib.pyplot as plt
import seaborn as sns
from modulos_modelos import segmentacion_clientes
import os
from modulos_modelos import scoring_leads
from modulos_modelos import prediccion_churn
from modulos_modelos import panel_rendimiento_agentes
from modulos_modelos import pricing_dinamico
from modulos_modelos import marketing_personalizado
from modulos_modelos import fidelizacion_clientes
from modulos_modelos import analisis_sentimiento_nps
from modulos_modelos import analizar_siniestros
from modulos_modelos import priorizar_clientes_dormidos
from modulos_modelos import digitalizar_renovaciones

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
        elif nombre.endswith(('.xlsx', '.xls')):  # Maneja Excel también
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
    return render_template('index.html')

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

@app.route('/smartfilter', methods=['GET', 'POST'])
def smartfilter():
    """
    Realiza el filtrado inteligente de los datos (ajusta esta funcionalidad).
    """
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            # Filtrado inteligente aquí, ejemplo: filtrar columnas con NaN o valores bajos
            df_clean = df.dropna()  # Simple ejemplo de eliminación de NaN
            flash(resultado_mensaje("¡Filtrado inteligente realizado con éxito!"), "success")
            return render_template('smartfilter.html', df=df_clean.head())
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('smartfilter.html')

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

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """
    Endpoint de comprobación de salud para garantizar que la aplicación está funcionando.
    """
    return "OK", 200

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

    @app.route('/segmentacion', methods=['GET', 'POST'])
def segmentacion():
    """
    Segmentación avanzada de clientes (K-Means)
    """
    columnas_por_defecto = ['edad', 'num_pólizas', 'prima_anual', 'siniestros_12m']
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        columnas = request.form.get('columnas')  # Esto será un string tipo 'edad,num_pólizas'
        n_clusters = int(request.form.get('n_clusters', 4))
        try:
            df = cargar_dataset(archivo)
            columnas = [c.strip() for c in (columnas.split(',') if columnas else columnas_por_defecto)]
            df_segmentado, grafico_path = segmentacion_clientes(df, columnas, n_clusters)
            # Exporta resultados
            output_csv = os.path.join('static', 'graficos', 'clientes_segmentados.csv')
            df_segmentado.to_csv(output_csv, index=False)
            flash(resultado_mensaje("¡Segmentación realizada con éxito! Descarga el resultado y visualiza el gráfico."), "success")
            return render_template(
                'segmentacion.html',
                grafico=grafico_path,
                descargar_csv=output_csv
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('segmentacion.html')

    @app.route('/scoring-leads', methods=['GET', 'POST'])
def scoring_leads_view():
    """
    Scoring y priorización de leads con Random Forest.
    """
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        target_col = request.form.get('target_col', 'convertido')
        try:
            df = cargar_dataset(archivo)
            output_csv, output_xlsx, reporte = scoring_leads(df, target_col=target_col)
            flash(resultado_mensaje("¡Scoring de leads realizado con éxito! Descarga el ranking y consulta el reporte."), "success")
            return render_template(
                'scoring_leads.html',
                descargar_csv=output_csv,
                descargar_xlsx=output_xlsx,
                reporte=reporte
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('scoring_leads.html')

    @app.route('/churn', methods=['GET', 'POST'])
def churn_view():
    """
    Predicción de churn de clientes (cancelación) usando Random Forest.
    """
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            columnas = request.form.get('columnas')
            # Puedes permitir que el usuario indique las columnas, por defecto las que indica la función
            if columnas:
                columnas = [c.strip() for c in columnas.split(',')]
            reporte_csv, reporte_xlsx, predicciones_csv, reporte_dict = prediccion_churn(df, columnas=columnas)
            flash(resultado_mensaje("Predicción de churn realizada con éxito. Descarga los resultados y revisa el reporte."), "success")
            return render_template(
                'churn.html',
                reporte_csv=reporte_csv,
                reporte_xlsx=reporte_xlsx,
                predicciones_csv=predicciones_csv,
                reporte=reporte_dict
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('churn.html')

    @app.route('/panel-agentes', methods=['GET', 'POST'])
def panel_agentes_view():
    """
    Panel 360° de rendimiento de agentes.
    """
    resumen = None
    panel_csv = None
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            panel_csv, resumen = panel_rendimiento_agentes(df)
            flash(resultado_mensaje("Panel generado con éxito. Descarga los resultados o revisa el resumen abajo."), "success")
            # Mostrar tabla HTML resumida (las primeras filas)
            resumen_html = resumen.head().to_html(classes="table table-striped", index=False)
            return render_template(
                'panel_agentes.html',
                panel_csv=panel_csv,
                resumen_html=resumen_html
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('panel_agentes.html')

    @app.route('/pricing', methods=['GET', 'POST'])
def pricing_view():
    """
    Pricing dinámico y simulación de escenarios.
    """
    resultados = None
    sim_df = None
    csv_path = None
    graf_path = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        margen = float(request.form.get('margen', 0.25))
        try:
            df = cargar_dataset(archivo)
            csv_path, graf_path, resultados, sim_df = pricing_dinamico(df, margen_deseado=margen)
            flash(resultado_mensaje("Cálculo de precios y simulación realizados con éxito. Descarga los resultados o revisa la gráfica."), "success")
            resultados_html = resultados.head().to_html(classes="table table-striped", index=False)
            sim_html = sim_df.to_html(classes="table table-bordered", index=False)
            return render_template(
                'pricing.html',
                resultados_html=resultados_html,
                sim_html=sim_html,
                csv_path=csv_path,
                graf_path=graf_path
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('pricing.html')

    @app.route('/marketing', methods=['GET', 'POST'])
def marketing_view():
    """
    Marketing personalizado 1:1 para campañas automáticas de clientes.
    """
    mensajes_html = None
    output_path = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            # Es importante convertir fechas si no vienen en formato datetime64
            for col in ['fecha_nacimiento', 'fecha_renovacion', 'fecha_siniestro']:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            df_mensajes, output_path = marketing_personalizado(df)
            flash(resultado_mensaje("Mensajes generados correctamente. Descarga el archivo o revisa los mensajes abajo."), "success")
            mensajes_html = df_mensajes.head(10).to_html(classes="table table-striped", index=False)
            return render_template(
                'marketing.html',
                mensajes_html=mensajes_html,
                output_path=output_path
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('marketing.html')

    @app.route('/fidelizacion', methods=['GET', 'POST'])
def fidelizacion_view():
    """
    Calcula el programa de fidelidad y segmenta los clientes.
    """
    tabla_html = None
    output_path = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            # Convertir fecha_alta si no está en formato datetime
            df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], errors='coerce')
            df_resultado, output_path = fidelizacion_clientes(df)
            flash(resultado_mensaje("Programa de fidelidad calculado y clientes segmentados correctamente."), "success")
            tabla_html = df_resultado.head(10).to_html(classes="table table-striped", index=False)
            return render_template(
                'fidelizacion.html',
                tabla_html=tabla_html,
                output_path=output_path
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('fidelizacion.html')

    @app.route('/sentimiento', methods=['GET', 'POST'])
def sentimiento_view():
    """
    Analiza sentimiento de comentarios y calcula NPS.
    """
    tabla_html = None
    nps_score = None
    csv_out = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            df_resultado, nps_score, csv_out, txt_out = analisis_sentimiento_nps(df)
            flash(resultado_mensaje("Análisis de sentimiento y NPS completado."), "success")
            tabla_html = df_resultado[['comentario', 'sentimiento', 'estado', 'nps']].head(10).to_html(classes="table table-striped", index=False)
            return render_template(
                'sentimiento.html',
                tabla_html=tabla_html,
                nps_score=nps_score,
                csv_out=csv_out
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('sentimiento.html')

    @app.route('/siniestros', methods=['GET', 'POST'])
def siniestros_view():
    """
    Analiza los tiempos de gestión de siniestros, detecta cuellos de botella y sugiere optimización.
    """
    df_resultado = None
    cuellos = None
    recomendaciones = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            df_resultado, cuellos, recomendaciones = analizar_siniestros(df)
            flash(resultado_mensaje("Análisis de siniestros completado."), "success")
            return render_template(
                'siniestros.html',
                df_resultado=df_resultado.head(10).to_html(classes="table table-striped", index=False),
                cuellos=cuellos.to_html(classes="table table-striped", index=True),
                recomendaciones=recomendaciones
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('siniestros.html')

    @app.route('/clientes_dormidos', methods=['GET', 'POST'])
def clientes_dormidos_view():
    """
    Identifica y prioriza clientes dormidos para reactivación comercial.
    """
    resultado = None
    file_export = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            resultado, file_export = priorizar_clientes_dormidos(df)
            flash(resultado_mensaje("Clientes priorizados por score de reactivación. Puedes descargar el archivo Excel abajo."), "success")
            return render_template(
                'clientes_dormidos.html',
                resultado=resultado.head(10).to_html(classes="table table-striped", index=False),
                file_export=file_export
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('clientes_dormidos.html')

# -- INICIO SEGURO (PRODUCCIÓN) --

if __name__ == "__main__":
    # Usa el puerto proporcionado por Render o 5000 como valor predeterminado
    port = int(os.environ.get('PORT', 5000))  # Render automáticamente proporciona la variable PORT
    app.run(host="0.0.0.0", port=port)  # Escucha en todas las interfaces y en el puerto proporcionado







