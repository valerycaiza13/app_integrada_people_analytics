# =============================================================
# PEOPLE ANALYTICS DASHBOARD — 3 TAREAS INTEGRADAS
# Tarea 1: Extracción de skills en CVs
# Tarea 2: Análisis de sentimiento con pysentimiento + comparación VADER/TextBlob
# Tarea 3: Rotación mensual
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt

from collections import Counter
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="People Analytics Dashboard", layout="wide")

st.title("People Analytics Dashboard")
st.caption("Dashboard ejecutivo para extracción de skills, análisis de sentimiento y rotación mensual.")

st.markdown("""
<style>
.insight-box {background-color: #F8FAFC; padding: 18px; border-left: 5px solid #2050F6; border-radius: 12px; margin-top: 10px; margin-bottom: 15px;}
.positive-box {background-color: #ECFDF3; padding: 15px; border-left: 5px solid #22C55E; border-radius: 12px;}
.warning-box {background-color: #FFF7ED; padding: 15px; border-left: 5px solid #F97316; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Tarea 1 · Skills en CVs", "Tarea 2 · Sentimiento", "Tarea 3 · Rotación"])

# =============================================================
# FUNCIONES AUXILIARES
# =============================================================

def validar_columnas(df, columnas_requeridas, nombre_dataset):
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        st.error(f"El archivo de {nombre_dataset} no tiene estas columnas requeridas: {faltantes}")
        st.stop()


def mostrar_bar_labels(ax, formato="{:.0f}"):
    for p in ax.patches:
        valor = p.get_height()
        ax.annotate(formato.format(valor), (p.get_x() + p.get_width() / 2., valor), ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

# =============================================================
# TAREA 1 — EXTRACCIÓN DE SKILLS EN CVs
# =============================================================

skills_dict = {
    "python": ["python"], "sql": ["sql"], "excel": ["excel", "excel avanzado"],
    "power bi": ["power bi", "powerbi"], "tableau": ["tableau"], "r": ["lenguaje r", " r "],
    "pandas": ["pandas"], "estadística": ["estadística", "estadistica"],
    "machine learning": ["machine learning"], "tensorflow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "sklearn"], "docker": ["docker"], "git": ["git"],
    "people analytics": ["people analytics"], "hris": ["hris"], "workday": ["workday"],
    "ats": ["ats"], "sourcing": ["sourcing"], "linkedin recruiter": ["linkedin recruiter"],
    "boolean search": ["boolean search"], "entrevista por competencias": ["entrevista por competencias"],
    "reclutamiento": ["reclutamiento"], "gestión del talento": ["gestión del talento", "gestion del talento"],
    "análisis predictivo": ["análisis predictivo", "analisis predictivo"],
    "comunicación": ["comunicación", "comunicacion"], "trabajo en equipo": ["trabajo en equipo"],
    "liderazgo": ["liderazgo"], "atención al detalle": ["atención al detalle", "atencion al detalle"]
}

requisitos = {
    "Data Analyst": ["python", "sql", "excel", "power bi", "tableau", "estadística"],
    "People Analytics Specialist": ["people analytics", "python", "r", "power bi", "excel", "análisis predictivo"],
    "Machine Learning Engineer": ["python", "machine learning", "tensorflow", "scikit-learn", "docker", "git"],
    "Recruiter Técnico": ["sourcing", "linkedin recruiter", "boolean search", "ats", "entrevista por competencias", "reclutamiento"],
    "HR Business Partner": ["gestión del talento", "people analytics", "hris", "workday", "excel", "comunicación"]
}


def extraer_skills(texto):
    texto = str(texto).lower()
    skills_encontradas = []
    for skill, variantes in skills_dict.items():
        for variante in variantes:
            patron = r"\b" + re.escape(variante.strip().lower()) + r"\b"
            if re.search(patron, texto):
                skills_encontradas.append(skill)
                break
    return skills_encontradas


def calcular_match(row):
    puesto = row["puesto_target"]
    skills_cv = set(row["skills_detectadas"])
    skills_req = set(requisitos.get(puesto, []))
    encontradas = skills_cv.intersection(skills_req)
    faltantes = skills_req.difference(skills_cv)
    match = round((len(encontradas) / len(skills_req)) * 100, 1) if skills_req else 0
    return pd.Series({"skills_requeridas": ", ".join(sorted(skills_req)), "skills_encontradas_requeridas": ", ".join(sorted(encontradas)), "skills_faltantes": ", ".join(sorted(faltantes)), "porcentaje_match": match})


def decision(match):
    if match >= 70:
        return "Preseleccionar"
    elif match >= 50:
        return "Revisar manualmente"
    else:
        return "No priorizar"


with tab1:
    st.header("Tarea 1 — Extracción de Skills en Currículums")
    st.write("Automatiza la preselección detectando habilidades clave en CVs y comparándolas con los requisitos del cargo.")
    uploaded_file1 = st.file_uploader("Sube el archivo tarea1_cvs.csv", type=["csv"], key="csv_tarea1")
    if uploaded_file1 is not None:
        df1 = pd.read_csv(uploaded_file1)
        validar_columnas(df1, ["nombre", "puesto_target", "cv_texto"], "Tarea 1")
        df1["skills_detectadas"] = df1["cv_texto"].apply(extraer_skills)
        df1_resultado = pd.concat([df1, df1.apply(calcular_match, axis=1)], axis=1)
        df1_resultado["decision_sugerida"] = df1_resultado["porcentaje_match"].apply(decision)
        df_skills = df1_resultado.explode("skills_detectadas")
        ranking = df1_resultado[["nombre", "puesto_target", "porcentaje_match", "skills_encontradas_requeridas", "skills_faltantes", "decision_sugerida"]].sort_values("porcentaje_match", ascending=False)
        skill_mas_frecuente = df_skills["skills_detectadas"].dropna().value_counts().idxmax()
        mejor_candidato = ranking.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidatos analizados", len(df1_resultado))
        c2.metric("Preseleccionados", (df1_resultado["decision_sugerida"] == "Preseleccionar").sum())
        c3.metric("Match promedio", f"{df1_resultado['porcentaje_match'].mean():.1f}%")
        c4.metric("Skill más frecuente", skill_mas_frecuente)
        st.markdown(f"""
        <div class="insight-box"><b>Insight ejecutivo:</b><br>
        El mejor match es <b>{mejor_candidato['nombre']}</b> para <b>{mejor_candidato['puesto_target']}</b>,
        con <b>{mejor_candidato['porcentaje_match']}%</b> de coincidencia. Esta automatización permite priorizar candidatos,
        detectar brechas de skills y reducir la revisión manual inicial.</div>
        """, unsafe_allow_html=True)
        st.subheader("Ranking de candidatos")
        st.dataframe(ranking, use_container_width=True)
        tabla_frecuencia = df_skills.groupby(["puesto_target", "skills_detectadas"]).size().reset_index(name="frecuencia").sort_values(["puesto_target", "frecuencia"], ascending=[True, False])
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Top skills generales")
            top_skills = df_skills["skills_detectadas"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top_skills.index[::-1], top_skills.values[::-1])
            ax.set_xlabel("Frecuencia")
            ax.set_title("Skills más frecuentes detectadas")
            st.pyplot(fig)
        with col_b:
            st.subheader("Decisiones sugeridas")
            decisiones = df1_resultado["decision_sugerida"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.bar(decisiones.index, decisiones.values)
            ax2.set_ylabel("Número de candidatos")
            ax2.set_title("Distribución de decisiones de preselección")
            ax2.tick_params(axis="x", rotation=20)
            mostrar_bar_labels(ax2)
            st.pyplot(fig2)
        st.subheader("Skills más frecuentes por perfil")
        st.dataframe(tabla_frecuencia, use_container_width=True)
        with st.expander("Ver dataset procesado"):
            st.dataframe(df1_resultado, use_container_width=True)
        st.download_button("Descargar ranking de candidatos", data=ranking.to_csv(index=False).encode("utf-8"), file_name="ranking_candidatos_por_match.csv", mime="text/csv")
        st.download_button("Descargar skills frecuentes por perfil", data=tabla_frecuencia.to_csv(index=False).encode("utf-8"), file_name="tabla_skills_frecuentes_por_perfil.csv", mime="text/csv")
    else:
        st.info("Sube el archivo CSV de la Tarea 1 para iniciar el análisis.")

# =============================================================
# TAREA 2 — ANÁLISIS DE SENTIMIENTO CON PYSENTIMIENTO
# =============================================================

vader = SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=False)
def cargar_modelo_pysentimiento():
    from pysentimiento import create_analyzer
    return create_analyzer(task="sentiment", lang="es")

stopwords_es = {"de", "la", "el", "y", "en", "que", "los", "las", "un", "una", "muy", "con", "por", "para", "del", "al", "me", "mi", "es", "hay", "se", "más", "mas", "todo", "todos", "porque", "pero", "esta", "este", "esto", "son", "sin", "sobre", "como", "bien", "tener", "tiene", "trabajo", "empresa", "equipo"}


def limpiar_palabras(texto):
    texto = str(texto).lower()
    for p in string.punctuation:
        texto = texto.replace(p, " ")
    return [p for p in texto.split() if p not in stopwords_es and len(p) > 3]


def top_palabras(df, sentimiento, top_n=10):
    textos = df[df["pred_pysentimiento"] == sentimiento]["comentario"]
    palabras = []
    for texto in textos:
        palabras.extend(limpiar_palabras(texto))
    return pd.DataFrame(Counter(palabras).most_common(top_n), columns=["palabra", "frecuencia"])


def clasificar_textblob(texto):
    score = TextBlob(str(texto)).sentiment.polarity
    if score > 0.05:
        return "positivo"
    elif score < -0.05:
        return "negativo"
    return "neutro"


def clasificar_vader(texto):
    score = vader.polarity_scores(str(texto))["compound"]
    if score >= 0.05:
        return "positivo"
    elif score <= -0.05:
        return "negativo"
    return "neutro"


def clasificar_pysentimiento(texto, analizador):
    resultado = analizador.predict(str(texto)).output
    return {"POS": "positivo", "NEG": "negativo", "NEU": "neutro"}.get(resultado, "neutro")


def resumen_sentimiento(df, columna_grupo):
    resumen = df.groupby([columna_grupo, "pred_pysentimiento"]).size().unstack(fill_value=0).reset_index()
    for col in ["negativo", "neutro", "positivo"]:
        if col not in resumen.columns:
            resumen[col] = 0
    resumen["total"] = resumen[["negativo", "neutro", "positivo"]].sum(axis=1)
    resumen["pct_negativo"] = (resumen["negativo"] / resumen["total"] * 100).round(1)
    resumen["pct_neutro"] = (resumen["neutro"] / resumen["total"] * 100).round(1)
    resumen["pct_positivo"] = (resumen["positivo"] / resumen["total"] * 100).round(1)
    return resumen

with tab2:
    st.header("Tarea 2 — Análisis de Sentimiento en Encuestas de Clima")
    st.write("Análisis estratégico del clima laboral utilizando NLP en español para detectar focos de atención, drivers emocionales y oportunidades de mejora.")
    uploaded_file2 = st.file_uploader("Sube el archivo tarea2_encuesta_clima.csv", type=["csv"], key="csv_tarea2")
    if uploaded_file2 is not None:
        df2 = pd.read_csv(uploaded_file2)
        validar_columnas(df2, ["comentario", "departamento", "nivel", "sentimiento_real"], "Tarea 2")
        analizador_es = cargar_modelo_pysentimiento()
        with st.spinner("Analizando comentarios con TextBlob, VADER y pysentimiento..."):
            df2["pred_textblob"] = df2["comentario"].apply(clasificar_textblob)
            df2["pred_vader"] = df2["comentario"].apply(clasificar_vader)
            df2["pred_pysentimiento"] = df2["comentario"].apply(lambda x: clasificar_pysentimiento(x, analizador_es))
        acc_textblob = accuracy_score(df2["sentimiento_real"], df2["pred_textblob"])
        acc_vader = accuracy_score(df2["sentimiento_real"], df2["pred_vader"])
        acc_pys = accuracy_score(df2["sentimiento_real"], df2["pred_pysentimiento"])
        total = len(df2)
        positivos = df2["pred_pysentimiento"].eq("positivo").sum()
        negativos = df2["pred_pysentimiento"].eq("negativo").sum()
        neutros = df2["pred_pysentimiento"].eq("neutro").sum()
        pct_pos = positivos / total * 100
        pct_neg = negativos / total * 100
        pct_neu = neutros / total * 100
        resumen_depto = resumen_sentimiento(df2, "departamento")
        resumen_nivel = resumen_sentimiento(df2, "nivel")
        depto_critico = resumen_depto.sort_values("pct_negativo", ascending=False).iloc[0]
        depto_fuerte = resumen_depto.sort_values("pct_positivo", ascending=False).iloc[0]
        nivel_critico = resumen_nivel.sort_values("pct_negativo", ascending=False).iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Comentarios analizados", total)
        c2.metric("% positivo", f"{pct_pos:.1f}%", f"{positivos} comentarios")
        c3.metric("% negativo", f"{pct_neg:.1f}%", f"{negativos} comentarios")
        c4.metric("% neutro", f"{pct_neu:.1f}%", f"{neutros} comentarios")
        c5.metric("Accuracy NLP español", f"{acc_pys:.1%}")
        st.markdown(f"""
        <div class="insight-box"><b>Resumen ejecutivo:</b><br><br>
        • El departamento con mejor percepción es <b>{depto_fuerte['departamento']}</b>, con <b>{depto_fuerte['pct_positivo']:.1f}%</b> de comentarios positivos.<br><br>
        • El principal foco de atención es <b>{depto_critico['departamento']}</b>, donde el <b>{depto_critico['pct_negativo']:.1f}%</b> de comentarios presentan sentimiento negativo.<br><br>
        • El nivel organizacional más sensible es <b>{nivel_critico['nivel']}</b>, con <b>{nivel_critico['pct_negativo']:.1f}%</b> de sentimiento negativo.<br><br>
        • El modelo principal fue <b>pysentimiento</b>, porque está entrenado para español y obtuvo una precisión de <b>{acc_pys:.1%}</b>.
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Comparación de modelos NLP")
        comparacion = pd.DataFrame({"Modelo": ["TextBlob", "VADER", "pysentimiento"], "Enfoque": ["Léxico general", "Léxico social", "Modelo NLP en español"], "Accuracy": [round(acc_textblob * 100, 1), round(acc_vader * 100, 1), round(acc_pys * 100, 1)]}).sort_values("Accuracy", ascending=False)
        col_modelo1, col_modelo2 = st.columns([1, 1])
        with col_modelo1:
            st.dataframe(comparacion, use_container_width=True)
        with col_modelo2:
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            datos_acc = comparacion.sort_values("Accuracy", ascending=True)
            ax1.barh(datos_acc["Modelo"], datos_acc["Accuracy"])
            ax1.set_xlabel("Accuracy (%)")
            ax1.set_title("Precisión por modelo")
            ax1.set_xlim(0, 100)
            for i, v in enumerate(datos_acc["Accuracy"]):
                ax1.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
            st.pyplot(fig1)
        st.subheader("Drivers emocionales detectados")
        palabras_pos = top_palabras(df2, "positivo")
        palabras_neg = top_palabras(df2, "negativo")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Factores asociados a comentarios positivos")
            if not palabras_pos.empty:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.barh(palabras_pos["palabra"][::-1], palabras_pos["frecuencia"][::-1])
                ax2.set_title("Drivers positivos")
                ax2.set_xlabel("Frecuencia")
                st.pyplot(fig2)
            else:
                st.info("No se detectaron suficientes comentarios positivos.")
        with col2:
            st.markdown("### Factores asociados a comentarios negativos")
            if not palabras_neg.empty:
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                ax3.barh(palabras_neg["palabra"][::-1], palabras_neg["frecuencia"][::-1])
                ax3.set_title("Drivers negativos")
                ax3.set_xlabel("Frecuencia")
                st.pyplot(fig3)
            else:
                st.info("No se detectaron suficientes comentarios negativos.")
        st.subheader("Focos de atención por departamento")
        datos_neg = resumen_depto.sort_values("pct_negativo", ascending=True)
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        ax4.barh(datos_neg["departamento"], datos_neg["pct_negativo"])
        ax4.set_xlabel("% negativo")
        ax4.set_title("Sentimiento negativo por departamento")
        for i, v in enumerate(datos_neg["pct_negativo"]):
            ax4.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        st.pyplot(fig4)
        st.subheader("Análisis por nivel organizacional")
        datos_nivel = resumen_nivel.sort_values("pct_negativo", ascending=True)
        fig5, ax5 = plt.subplots(figsize=(9, 5))
        ax5.barh(datos_nivel["nivel"], datos_nivel["pct_negativo"])
        ax5.set_xlabel("% negativo")
        ax5.set_title("Sentimiento negativo por nivel")
        for i, v in enumerate(datos_nivel["pct_negativo"]):
            ax5.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        st.pyplot(fig5)
        st.subheader("Tablas ejecutivas de seguimiento")
        col_tabla1, col_tabla2 = st.columns(2)
        with col_tabla1:
            st.write("Departamento")
            st.dataframe(resumen_depto[["departamento", "negativo", "neutro", "positivo", "pct_negativo", "pct_positivo"]].sort_values("pct_negativo", ascending=False), use_container_width=True)
        with col_tabla2:
            st.write("Nivel organizacional")
            st.dataframe(resumen_nivel[["nivel", "negativo", "neutro", "positivo", "pct_negativo", "pct_positivo"]].sort_values("pct_negativo", ascending=False), use_container_width=True)
        st.subheader("Recomendaciones estratégicas para HR")
        st.write(f"1. Priorizar sesiones de escucha activa en **{depto_critico['departamento']}**, donde se concentra el mayor sentimiento negativo.")
        st.write(f"2. Diseñar acciones específicas para el nivel **{nivel_critico['nivel']}**, considerando carga laboral, liderazgo, comunicación y oportunidades de desarrollo.")
        st.write(f"3. Replicar buenas prácticas del departamento **{depto_fuerte['departamento']}**, donde se observa la mayor proporción de comentarios positivos.")
        st.write("4. Monitorear periódicamente los drivers negativos para detectar riesgos de burnout, desmotivación o problemas de comunicación interna.")
        with st.expander("Ver comentarios clasificados por los 3 modelos"):
            st.dataframe(df2[["departamento", "nivel", "comentario", "sentimiento_real", "pred_textblob", "pred_vader", "pred_pysentimiento"]], use_container_width=True)
        st.download_button("Descargar resultados de sentimiento", data=df2.to_csv(index=False).encode("utf-8"), file_name="resultados_sentimiento_pysentimiento.csv", mime="text/csv")
    else:
        st.info("Sube el archivo CSV de la Tarea 2 para iniciar el análisis.")

# =============================================================
# TAREA 3 — ROTACIÓN MENSUAL
# =============================================================

with tab3:
    st.header("Tarea 3 — Cálculo Automatizado de Tasa de Rotación Mensual")
    st.write("Calcula la tasa de rotación mensual y detecta áreas o meses con mayor salida de empleados para orientar decisiones de retención.")
    uploaded_file3 = st.file_uploader("Sube el archivo tarea3_rotacion_empleados.csv", type=["csv"], key="csv_tarea3")
    if uploaded_file3 is not None:
        df3 = pd.read_csv(uploaded_file3)
        validar_columnas(df3, ["fecha_ingreso", "fecha_salida", "departamento"], "Tarea 3")
        df3["fecha_ingreso"] = pd.to_datetime(df3["fecha_ingreso"], errors="coerce")
        df3["fecha_salida"] = pd.to_datetime(df3["fecha_salida"], errors="coerce")
        bajas = df3[df3["fecha_salida"].notna()].copy()
        bajas["mes_salida"] = bajas["fecha_salida"].dt.to_period("M").astype(str)
        bajas_mensuales = bajas.groupby("mes_salida").size().reset_index(name="bajas")
        total_empleados = len(df3)
        total_bajas = len(bajas)
        tasa_global = total_bajas / total_empleados * 100 if total_empleados else 0
        bajas_mensuales["tasa_rotacion"] = (bajas_mensuales["bajas"] / total_empleados * 100).round(2)
        empleados_departamento = df3.groupby("departamento").size().reset_index(name="total_empleados")
        bajas_departamento = bajas.groupby("departamento").size().reset_index(name="bajas")
        rotacion_departamento = empleados_departamento.merge(bajas_departamento, on="departamento", how="left").fillna(0)
        rotacion_departamento["bajas"] = rotacion_departamento["bajas"].astype(int)
        rotacion_departamento["tasa_rotacion_departamento"] = (rotacion_departamento["bajas"] / rotacion_departamento["total_empleados"] * 100).round(2)
        rotacion_departamento = rotacion_departamento.sort_values("tasa_rotacion_departamento", ascending=False)
        mes_mayor = bajas_mensuales.sort_values("tasa_rotacion", ascending=False).iloc[0]
        depto_mayor = rotacion_departamento.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Empleados analizados", total_empleados)
        c2.metric("Bajas registradas", total_bajas)
        c3.metric("Rotación global", f"{tasa_global:.1f}%")
        c4.metric("Departamento crítico", depto_mayor["departamento"], f"{depto_mayor['tasa_rotacion_departamento']}%")
        st.markdown(f"""
        <div class="insight-box"><b>Insight de rotación:</b><br>
        El mes con mayor rotación fue <b>{mes_mayor['mes_salida']}</b>, con <b>{mes_mayor['tasa_rotacion']}%</b>.
        El departamento más crítico es <b>{depto_mayor['departamento']}</b>, con una tasa de <b>{depto_mayor['tasa_rotacion_departamento']}%</b>.</div>
        """, unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.subheader("Tasa de rotación mensual")
            fig5, ax5 = plt.subplots(figsize=(9, 5))
            ax5.plot(bajas_mensuales["mes_salida"], bajas_mensuales["tasa_rotacion"], marker="o")
            ax5.set_xlabel("Mes")
            ax5.set_ylabel("Tasa de rotación (%)")
            ax5.set_title("Evolución mensual de rotación")
            ax5.tick_params(axis="x", rotation=45)
            ax5.grid(True, alpha=0.3)
            st.pyplot(fig5)
        with col_r2:
            st.subheader("Rotación por departamento")
            fig6, ax6 = plt.subplots(figsize=(9, 5))
            datos_rot = rotacion_departamento.sort_values("tasa_rotacion_departamento", ascending=True)
            ax6.barh(datos_rot["departamento"], datos_rot["tasa_rotacion_departamento"])
            ax6.set_xlabel("Tasa de rotación (%)")
            ax6.set_title("Áreas con mayor rotación")
            for i, v in enumerate(datos_rot["tasa_rotacion_departamento"]):
                ax6.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
            st.pyplot(fig6)
        st.subheader("Tablas resumen")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write("Meses con mayor rotación")
            st.dataframe(bajas_mensuales.sort_values("tasa_rotacion", ascending=False), use_container_width=True)
        with col_t2:
            st.write("Rotación por departamento")
            st.dataframe(rotacion_departamento, use_container_width=True)
        st.subheader("Acciones de retención sugeridas")
        st.write(f"1. Realizar entrevistas de salida estructuradas en **{depto_mayor['departamento']}** para entender causas recurrentes.")
        st.write("2. Revisar carga laboral, oportunidades de desarrollo y estilo de liderazgo en las áreas con mayor rotación.")
        st.write("3. Monitorear la tasa mensualmente para detectar picos tempranos y activar planes de retención antes de que el problema escale.")
        with st.expander("Ver dataset procesado"):
            st.dataframe(df3, use_container_width=True)
        st.download_button("Descargar tasa de rotación mensual", data=bajas_mensuales.to_csv(index=False).encode("utf-8"), file_name="tasa_rotacion_mensual.csv", mime="text/csv")
        st.download_button("Descargar rotación por departamento", data=rotacion_departamento.to_csv(index=False).encode("utf-8"), file_name="rotacion_por_departamento.csv", mime="text/csv")
    else:
        st.info("Sube el archivo CSV de la Tarea 3 para iniciar el análisis.")

