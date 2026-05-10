import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, confusion_matrix

# =============================================================
# CONFIGURACIÓN GENERAL
# =============================================================

st.set_page_config(page_title="People Analytics Dashboard", layout="wide")

st.title("People Analytics Dashboard")
st.caption("Dashboard integrado con insights ejecutivos para selección, clima laboral y rotación.")

st.markdown(
    """
    Esta app resume tres análisis de People Analytics usando Python: extracción de skills en CVs,
    clasificación de sentimiento en encuestas de clima y cálculo de tasa de rotación mensual.
    El foco está en mostrar **insights accionables**, no en visualizar todo el dataset crudo.
    """
)

tab1, tab2, tab3 = st.tabs([
    "Tarea 1 · Skills en CVs",
    "Tarea 2 · Sentimiento",
    "Tarea 3 · Rotación"
])

# =============================================================
# FUNCIONES GENERALES
# =============================================================

def validar_columnas(df, columnas_requeridas, nombre_archivo):
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        st.error(f"El archivo {nombre_archivo} no contiene estas columnas requeridas: {', '.join(faltantes)}")
        st.stop()


def plot_barh(df, x_col, y_col, title, xlabel):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df[y_col], df[x_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig)


def plot_bar(df, x_col, y_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)


def plot_line(df, x_col, y_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[x_col], df[y_col], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


# =============================================================
# TAREA 1 — EXTRACCIÓN DE SKILLS EN CVS
# =============================================================

skills_dict = {
    "python": ["python"],
    "sql": ["sql"],
    "excel": ["excel", "excel avanzado"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "r": ["lenguaje r", " r "],
    "pandas": ["pandas"],
    "estadística": ["estadística", "estadistica"],
    "machine learning": ["machine learning"],
    "tensorflow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "scikit learn"],
    "docker": ["docker"],
    "git": ["git"],
    "people analytics": ["people analytics"],
    "hris": ["hris"],
    "workday": ["workday"],
    "ats": ["ats"],
    "sourcing": ["sourcing"],
    "linkedin recruiter": ["linkedin recruiter"],
    "boolean search": ["boolean search"],
    "entrevista por competencias": ["entrevista por competencias"],
    "reclutamiento": ["reclutamiento", "selección", "seleccion"],
    "gestión del talento": ["gestión del talento", "gestion del talento"],
    "análisis predictivo": ["análisis predictivo", "analisis predictivo"],
    "comunicación": ["comunicación", "comunicacion"],
    "trabajo en equipo": ["trabajo en equipo"],
    "liderazgo": ["liderazgo"],
    "atención al detalle": ["atención al detalle", "atencion al detalle"]
}

requisitos = {
    "Data Analyst": ["python", "sql", "excel", "power bi", "tableau", "estadística"],
    "People Analytics Specialist": ["people analytics", "python", "r", "power bi", "excel", "análisis predictivo"],
    "Machine Learning Engineer": ["python", "machine learning", "tensorflow", "scikit-learn", "docker", "git"],
    "Recruiter Técnico": ["sourcing", "linkedin recruiter", "boolean search", "ats", "entrevista por competencias", "reclutamiento"],
    "HR Business Partner": ["gestión del talento", "people analytics", "hris", "workday", "excel", "comunicación"]
}


def extraer_skills(texto):
    texto = f" {str(texto).lower()} "
    skills_encontradas = []
    for skill, variantes in skills_dict.items():
        for variante in variantes:
            patron = r"\b" + re.escape(variante.strip().lower()) + r"\b"
            if re.search(patron, texto):
                skills_encontradas.append(skill)
                break
    return sorted(list(set(skills_encontradas)))


def calcular_match(row):
    puesto = row["puesto_target"]
    skills_cv = set(row["skills_detectadas"])
    skills_requeridas = set(requisitos.get(puesto, []))
    encontradas = sorted(skills_cv.intersection(skills_requeridas))
    faltantes = sorted(skills_requeridas.difference(skills_cv))
    match = round((len(encontradas) / len(skills_requeridas)) * 100, 2) if skills_requeridas else 0
    return pd.Series({
        "skills_requeridas": ", ".join(sorted(skills_requeridas)),
        "skills_encontradas_requeridas": ", ".join(encontradas),
        "skills_faltantes": ", ".join(faltantes),
        "porcentaje_match": match
    })


def decision(match):
    if match >= 70:
        return "Preseleccionar"
    if match >= 50:
        return "Revisar manualmente"
    return "No priorizar"


with tab1:
    st.header("Tarea 1 — Extracción de Skills en Currículums")
    st.write("Automatización de preselección identificando habilidades clave en CVs de texto libre.")

    uploaded_file = st.file_uploader("Sube el archivo tarea1_cvs.csv", type=["csv"], key="tarea1")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        validar_columnas(df, ["nombre", "puesto_target", "cv_texto"], "tarea1_cvs.csv")

        df["skills_detectadas"] = df["cv_texto"].apply(extraer_skills)
        resultado_match = df.apply(calcular_match, axis=1)
        df_resultado = pd.concat([df, resultado_match], axis=1)
        df_resultado["decision_sugerida"] = df_resultado["porcentaje_match"].apply(decision)
        df_resultado["n_skills_detectadas"] = df_resultado["skills_detectadas"].apply(len)

        ranking = df_resultado[[
            "nombre", "puesto_target", "porcentaje_match", "n_skills_detectadas",
            "skills_encontradas_requeridas", "skills_faltantes", "decision_sugerida"
        ]].sort_values("porcentaje_match", ascending=False)

        df_exploded = df_resultado.explode("skills_detectadas")
        tabla_frecuencia = (
            df_exploded.dropna(subset=["skills_detectadas"])
            .groupby(["puesto_target", "skills_detectadas"])
            .size()
            .reset_index(name="frecuencia")
            .sort_values(["puesto_target", "frecuencia"], ascending=[True, False])
        )

        top_skills = (
            df_exploded.dropna(subset=["skills_detectadas"])
            .groupby("skills_detectadas")
            .size()
            .reset_index(name="frecuencia")
            .sort_values("frecuencia", ascending=False)
            .head(10)
        )

        decisiones = df_resultado["decision_sugerida"].value_counts().reset_index()
        decisiones.columns = ["decisión", "candidatos"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CVs analizados", len(df_resultado))
        col2.metric("Skills únicas detectadas", df_exploded["skills_detectadas"].nunique())
        col3.metric("Match promedio", f"{df_resultado['porcentaje_match'].mean():.1f}%")
        col4.metric("Preseleccionados", int((df_resultado["decision_sugerida"] == "Preseleccionar").sum()))

        st.subheader("Insight 1: skills más frecuentes en los CVs")
        plot_barh(top_skills.sort_values("frecuencia", ascending=True), "frecuencia", "skills_detectadas", "Top 10 skills detectadas", "Frecuencia")

        st.subheader("Insight 2: distribución de decisiones sugeridas")
        plot_bar(decisiones, "decisión", "candidatos", "Decisión sugerida según match", "Decisión", "Número de candidatos")

        st.subheader("Insight 3: ranking de candidatos por match")
        st.dataframe(ranking, use_container_width=True, hide_index=True)

        st.subheader("Skills más frecuentes por perfil")
        st.dataframe(tabla_frecuencia, use_container_width=True, hide_index=True)

        mejor_candidato = ranking.iloc[0]
        st.success(
            f"El candidato con mayor ajuste es **{mejor_candidato['nombre']}** para **{mejor_candidato['puesto_target']}**, "
            f"con un match de **{mejor_candidato['porcentaje_match']}%**."
        )

        st.info(
            "Reflexión: este análisis permite automatizar una primera preselección, priorizando candidatos con mayor coincidencia "
            "entre las skills detectadas y los requisitos del cargo. La decisión final debe complementarse con revisión humana, "
            "experiencia previa, entrevista y ajuste cultural."
        )

        with st.expander("Ver muestra del dataset procesado"):
            st.dataframe(df_resultado[["nombre", "puesto_target", "cv_texto", "skills_detectadas"]].head(10), use_container_width=True)

        st.download_button("Descargar ranking de candidatos", ranking.to_csv(index=False).encode("utf-8"), "ranking_candidatos_por_match.csv", "text/csv")
    else:
        st.info("Sube el archivo CSV para iniciar el análisis de skills.")


# =============================================================
# TAREA 2 — ANÁLISIS DE SENTIMIENTO
# =============================================================

analyzer = SentimentIntensityAnalyzer()


def clasificar_vader(texto):
    score = analyzer.polarity_scores(str(texto))["compound"]
    if score >= 0.05:
        return "positivo"
    if score <= -0.05:
        return "negativo"
    return "neutro"


with tab2:
    st.header("Tarea 2 — Análisis de Sentimiento en Encuestas de Clima")
    st.write("Clasificación de comentarios como positivos, negativos o neutros para detectar focos de atención.")

    uploaded_file2 = st.file_uploader("Sube el archivo tarea2_encuesta_clima.csv", type=["csv"], key="tarea2")

    if uploaded_file2 is not None:
        df2 = pd.read_csv(uploaded_file2)
        validar_columnas(df2, ["comentario", "departamento", "nivel", "sentimiento_real"], "tarea2_encuesta_clima.csv")

        df2["sentimiento_predicho"] = df2["comentario"].apply(clasificar_vader)
        precision = accuracy_score(df2["sentimiento_real"], df2["sentimiento_predicho"])

        distribucion = df2["sentimiento_predicho"].value_counts().reset_index()
        distribucion.columns = ["sentimiento", "respuestas"]

        resumen_depto = (
            df2.groupby(["departamento", "sentimiento_predicho"])
            .size().unstack(fill_value=0).reset_index()
        )
        for col in ["negativo", "neutro", "positivo"]:
            if col not in resumen_depto.columns:
                resumen_depto[col] = 0
        resumen_depto["total"] = resumen_depto[["negativo", "neutro", "positivo"]].sum(axis=1)
        resumen_depto["pct_negativo"] = (resumen_depto["negativo"] / resumen_depto["total"] * 100).round(2)
        resumen_depto = resumen_depto.sort_values("pct_negativo", ascending=False)

        resumen_nivel = (
            df2.groupby(["nivel", "sentimiento_predicho"])
            .size().unstack(fill_value=0).reset_index()
        )
        for col in ["negativo", "neutro", "positivo"]:
            if col not in resumen_nivel.columns:
                resumen_nivel[col] = 0
        resumen_nivel["total"] = resumen_nivel[["negativo", "neutro", "positivo"]].sum(axis=1)
        resumen_nivel["pct_negativo"] = (resumen_nivel["negativo"] / resumen_nivel["total"] * 100).round(2)
        resumen_nivel = resumen_nivel.sort_values("pct_negativo", ascending=False)

        depto_critico = resumen_depto.iloc[0]
        nivel_critico = resumen_nivel.iloc[0]
        negativos = int((df2["sentimiento_predicho"] == "negativo").sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Respuestas analizadas", len(df2))
        col2.metric("Precisión VADER", f"{precision:.1%}")
        col3.metric("Comentarios negativos", negativos)
        col4.metric("Área crítica", depto_critico["departamento"], f"{depto_critico['pct_negativo']}% negativo")

        st.subheader("Insight 1: distribución general del sentimiento")
        plot_bar(distribucion, "sentimiento", "respuestas", "Distribución de sentimiento predicho", "Sentimiento", "Número de respuestas")

        st.subheader("Insight 2: departamentos con mayor porcentaje negativo")
        plot_barh(resumen_depto.head(10).sort_values("pct_negativo", ascending=True), "pct_negativo", "departamento", "% negativo por departamento", "% negativo")

        st.subheader("Insight 3: focos de atención por nivel")
        st.dataframe(resumen_nivel[["nivel", "positivo", "neutro", "negativo", "total", "pct_negativo"]], use_container_width=True, hide_index=True)

        st.warning(
            f"El foco principal de atención es **{depto_critico['departamento']}**, con **{depto_critico['pct_negativo']}%** de comentarios negativos. "
            f"Por nivel, el grupo con mayor alerta es **{nivel_critico['nivel']}**."
        )

        st.info(
            "Acciones sugeridas: realizar sesiones de escucha con el área crítica, revisar carga laboral y comunicación de managers, "
            "y complementar el modelo con revisión humana porque VADER puede fallar con ironía, contexto cultural o comentarios mixtos."
        )

        with st.expander("Ver comparación real vs predicho"):
            columnas = [col for col in ["id_respuesta", "departamento", "nivel", "comentario", "sentimiento_real", "sentimiento_predicho"] if col in df2.columns]
            st.dataframe(df2[columnas], use_container_width=True, hide_index=True)

        st.download_button("Descargar resultados de sentimiento", df2.to_csv(index=False).encode("utf-8"), "resultados_sentimiento.csv", "text/csv")
    else:
        st.info("Sube el archivo CSV para iniciar el análisis de sentimiento.")


# =============================================================
# TAREA 3 — ROTACIÓN MENSUAL
# =============================================================

with tab3:
    st.header("Tarea 3 — Cálculo Automatizado de Tasa de Rotación Mensual")
    st.write("Cálculo y visualización de la rotación mensual para apoyar decisiones de retención del talento.")

    uploaded_file3 = st.file_uploader("Sube el archivo tarea3_rotacion_empleados.csv", type=["csv"], key="tarea3")

    if uploaded_file3 is not None:
        df3 = pd.read_csv(uploaded_file3)
        validar_columnas(df3, ["fecha_ingreso", "fecha_salida", "departamento"], "tarea3_rotacion_empleados.csv")

        df3["fecha_ingreso"] = pd.to_datetime(df3["fecha_ingreso"], errors="coerce")
        df3["fecha_salida"] = pd.to_datetime(df3["fecha_salida"], errors="coerce")

        bajas = df3[df3["fecha_salida"].notna()].copy()
        bajas["mes_salida"] = bajas["fecha_salida"].dt.to_period("M")

        if bajas.empty:
            st.warning("No se encontraron fechas de salida. No es posible calcular rotación.")
            st.stop()

        meses = pd.period_range(bajas["mes_salida"].min(), bajas["mes_salida"].max(), freq="M")
        bajas_mensuales = bajas.groupby("mes_salida").size().reindex(meses, fill_value=0).reset_index()
        bajas_mensuales.columns = ["mes", "bajas"]

        # Headcount promedio mensual: empleados activos durante cada mes.
        headcount_promedio = []
        for mes in meses:
            inicio_mes = mes.to_timestamp()
            fin_mes = mes.to_timestamp(how="end")
            activos_inicio = ((df3["fecha_ingreso"] <= inicio_mes) & ((df3["fecha_salida"].isna()) | (df3["fecha_salida"] >= inicio_mes))).sum()
            activos_fin = ((df3["fecha_ingreso"] <= fin_mes) & ((df3["fecha_salida"].isna()) | (df3["fecha_salida"] >= fin_mes))).sum()
            headcount_promedio.append((activos_inicio + activos_fin) / 2)

        bajas_mensuales["headcount_promedio"] = headcount_promedio
        bajas_mensuales["tasa_rotacion"] = (bajas_mensuales["bajas"] / bajas_mensuales["headcount_promedio"] * 100).round(2)
        bajas_mensuales["mes"] = bajas_mensuales["mes"].astype(str)

        bajas_departamento = bajas.groupby("departamento").size().reset_index(name="bajas")
        empleados_departamento = df3.groupby("departamento").size().reset_index(name="total_empleados")
        rotacion_departamento = bajas_departamento.merge(empleados_departamento, on="departamento", how="left")
        rotacion_departamento["tasa_rotacion_departamento"] = (rotacion_departamento["bajas"] / rotacion_departamento["total_empleados"] * 100).round(2)
        rotacion_departamento = rotacion_departamento.sort_values("tasa_rotacion_departamento", ascending=False)

        if "motivo_baja" in bajas.columns:
            motivos = bajas["motivo_baja"].value_counts().reset_index()
            motivos.columns = ["motivo_baja", "bajas"]
        else:
            motivos = pd.DataFrame(columns=["motivo_baja", "bajas"])

        mes_mayor = bajas_mensuales.sort_values("tasa_rotacion", ascending=False).iloc[0]
        depto_mayor = rotacion_departamento.iloc[0]
        tasa_total = round(len(bajas) / len(df3) * 100, 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Empleados analizados", len(df3))
        col2.metric("Bajas totales", len(bajas))
        col3.metric("Rotación total", f"{tasa_total}%")
        col4.metric("Mes más crítico", mes_mayor["mes"], f"{mes_mayor['tasa_rotacion']}%")

        st.subheader("Insight 1: evolución de la tasa de rotación mensual")
        plot_line(bajas_mensuales, "mes", "tasa_rotacion", "Tasa de rotación mensual", "Mes", "Tasa de rotación (%)")

        st.subheader("Insight 2: departamentos con mayor rotación")
        plot_barh(rotacion_departamento.sort_values("tasa_rotacion_departamento", ascending=True), "tasa_rotacion_departamento", "departamento", "Tasa de rotación por departamento", "Tasa de rotación (%)")

        if not motivos.empty:
            st.subheader("Insight 3: principales motivos de baja")
            plot_barh(motivos.sort_values("bajas", ascending=True), "bajas", "motivo_baja", "Motivos de baja más frecuentes", "Número de bajas")

        st.subheader("Resumen mensual")
        st.dataframe(bajas_mensuales, use_container_width=True, hide_index=True)

        st.warning(
            f"El mes con mayor rotación fue **{mes_mayor['mes']}** con una tasa de **{mes_mayor['tasa_rotacion']}%**. "
            f"El departamento más crítico fue **{depto_mayor['departamento']}** con **{depto_mayor['tasa_rotacion_departamento']}%**."
        )

        st.info(
            f"Acciones sugeridas: 1) realizar entrevistas de salida estructuradas en **{depto_mayor['departamento']}**; "
            "2) revisar carga laboral, liderazgo y oportunidades de crecimiento en áreas con mayor rotación; "
            "3) monitorear mensualmente picos de salida para actuar antes de que el patrón se repita."
        )

        with st.expander("Ver muestra del dataset procesado"):
            st.dataframe(df3.head(15), use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("Descargar rotación mensual", bajas_mensuales.to_csv(index=False).encode("utf-8"), "tasa_rotacion_mensual.csv", "text/csv")
        with col_b:
            st.download_button("Descargar rotación por departamento", rotacion_departamento.to_csv(index=False).encode("utf-8"), "rotacion_por_departamento.csv", "text/csv")
    else:
        st.info("Sube el archivo CSV para iniciar el análisis de rotación.")

