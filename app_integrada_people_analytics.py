
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="People Analytics Dashboard", layout="wide")

st.title("People Analytics Dashboard")
st.write(
    "App integrada con tres análisis: extracción de skills en CVs, análisis de sentimiento "
    "en clima laboral y cálculo de rotación mensual."
)

tab1, tab2, tab3 = st.tabs([
    "Tarea 1 - Skills en CVs",
    "Tarea 2 - Sentimiento",
    "Tarea 3 - Rotación"
])

# =============================================================
# TAREA 1 — EXTRACCIÓN DE SKILLS EN CVS
# =============================================================

skills_dict = {
    "python": ["python"],
    "sql": ["sql"],
    "excel": ["excel", "excel avanzado"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "r": ["lenguaje r"],
    "pandas": ["pandas"],
    "estadística": ["estadística", "estadistica"],
    "machine learning": ["machine learning"],
    "tensorflow": ["tensorflow"],
    "scikit-learn": ["scikit-learn"],
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
    "reclutamiento": ["reclutamiento"],
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
    skills_requeridas = set(requisitos.get(puesto, []))

    skills_encontradas = skills_cv.intersection(skills_requeridas)
    skills_faltantes = skills_requeridas.difference(skills_cv)

    if len(skills_requeridas) == 0:
        match = 0
    else:
        match = round((len(skills_encontradas) / len(skills_requeridas)) * 100, 2)

    return pd.Series({
        "skills_requeridas": ", ".join(skills_requeridas),
        "skills_encontradas_requeridas": ", ".join(skills_encontradas),
        "skills_faltantes": ", ".join(skills_faltantes),
        "porcentaje_match": match
    })

def decision(match):
    if match >= 70:
        return "Preseleccionar"
    elif match >= 50:
        return "Revisar manualmente"
    else:
        return "No priorizar"

with tab1:
    st.header("Tarea 1 — Extracción de Skills en Currículums")
    st.write(
        "Objetivo: automatizar la preselección de candidatos identificando skills clave en CVs en texto libre."
    )

    uploaded_file = st.file_uploader("Sube el archivo tarea1_cvs.csv", type=["csv"], key="tarea1")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Vista inicial del dataset")
        st.dataframe(df)

        df["skills_detectadas"] = df["cv_texto"].apply(extraer_skills)

        st.subheader("Skills detectadas por candidato")
        st.dataframe(df[["nombre", "puesto_target", "skills_detectadas"]])

        df_exploded = df.explode("skills_detectadas")

        tabla_frecuencia = (
            df_exploded
            .groupby(["puesto_target", "skills_detectadas"])
            .size()
            .reset_index(name="frecuencia")
            .sort_values(["puesto_target", "frecuencia"], ascending=[True, False])
        )

        st.subheader("Tabla resumen: skills más frecuentes por perfil")
        st.dataframe(tabla_frecuencia)

        resultado_match = df.apply(calcular_match, axis=1)
        df_resultado = pd.concat([df, resultado_match], axis=1)
        df_resultado["decision_sugerida"] = df_resultado["porcentaje_match"].apply(decision)

        ranking = df_resultado[[
            "nombre",
            "puesto_target",
            "porcentaje_match",
            "skills_encontradas_requeridas",
            "skills_faltantes",
            "decision_sugerida"
        ]].sort_values("porcentaje_match", ascending=False)

        st.subheader("Ranking de candidatos según match con requisitos del cargo")
        st.dataframe(ranking)

        st.download_button(
            label="Descargar tabla de skills frecuentes",
            data=tabla_frecuencia.to_csv(index=False).encode("utf-8"),
            file_name="tabla_skills_frecuentes_por_perfil.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Descargar ranking de candidatos",
            data=ranking.to_csv(index=False).encode("utf-8"),
            file_name="ranking_candidatos_por_match.csv",
            mime="text/csv"
        )
    else:
        st.info("Sube el archivo CSV para iniciar el análisis de skills.")


# =============================================================
# TAREA 2 — ANÁLISIS DE SENTIMIENTO
# =============================================================

def clasificar_vader(texto):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(str(texto))["compound"]

    if score >= 0.05:
        return "positivo"
    elif score <= -0.05:
        return "negativo"
    else:
        return "neutro"

with tab2:
    st.header("Tarea 2 — Análisis de Sentimiento en Encuestas de Clima")
    st.write(
        "Objetivo: clasificar comentarios de empleados como positivos, negativos o neutros "
        "para identificar focos de atención por departamento y nivel."
    )

    uploaded_file2 = st.file_uploader("Sube el archivo tarea2_encuesta_clima.csv", type=["csv"], key="tarea2")

    if uploaded_file2 is not None:
        df2 = pd.read_csv(uploaded_file2)

        st.subheader("Vista inicial del dataset")
        st.dataframe(df2.head())

        df2["sentimiento_predicho"] = df2["comentario"].apply(clasificar_vader)

        precision = accuracy_score(df2["sentimiento_real"], df2["sentimiento_predicho"])

        st.metric("Precisión del modelo VADER", f"{precision:.2%}")

        st.subheader("Comparación real vs predicho")
        st.dataframe(df2[[
            "id_respuesta",
            "departamento",
            "nivel",
            "comentario",
            "sentimiento_real",
            "sentimiento_predicho"
        ]])

        resumen_depto = (
            df2
            .groupby(["departamento", "sentimiento_predicho"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        for col in ["negativo", "neutro", "positivo"]:
            if col not in resumen_depto.columns:
                resumen_depto[col] = 0

        resumen_depto["total"] = resumen_depto[["negativo", "neutro", "positivo"]].sum(axis=1)
        resumen_depto["pct_negativo"] = (resumen_depto["negativo"] / resumen_depto["total"] * 100).round(2)

        st.subheader("Resultados por departamento")
        st.dataframe(resumen_depto.sort_values("pct_negativo", ascending=False))

        resumen_nivel = (
            df2
            .groupby(["nivel", "sentimiento_predicho"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        for col in ["negativo", "neutro", "positivo"]:
            if col not in resumen_nivel.columns:
                resumen_nivel[col] = 0

        resumen_nivel["total"] = resumen_nivel[["negativo", "neutro", "positivo"]].sum(axis=1)
        resumen_nivel["pct_negativo"] = (resumen_nivel["negativo"] / resumen_nivel["total"] * 100).round(2)

        st.subheader("Resultados por nivel")
        st.dataframe(resumen_nivel.sort_values("pct_negativo", ascending=False))

        st.subheader("Visualización: % negativo por departamento")

        fig, ax = plt.subplots(figsize=(10, 5))
        datos_plot = resumen_depto.sort_values("pct_negativo", ascending=False)

        ax.bar(datos_plot["departamento"], datos_plot["pct_negativo"])
        ax.set_title("% de sentimiento negativo por departamento")
        ax.set_xlabel("Departamento")
        ax.set_ylabel("% negativo")
        ax.tick_params(axis="x", rotation=45)

        st.pyplot(fig)

        depto_critico = resumen_depto.sort_values("pct_negativo", ascending=False).iloc[0]["departamento"]

        st.subheader("Insights ejecutivos")
        st.write(f"- El departamento con mayor porcentaje de sentimiento negativo es **{depto_critico}**.")
        st.write("- Se recomienda realizar sesiones de escucha activa en áreas con mayor sentimiento negativo.")
        st.write("- Se recomienda complementar el análisis automático con revisión humana de comentarios críticos.")

    else:
        st.info("Sube el archivo CSV para iniciar el análisis de sentimiento.")


# =============================================================
# TAREA 3 — ROTACIÓN MENSUAL
# =============================================================

with tab3:
    st.header("Tarea 3 — Cálculo Automatizado de Tasa de Rotación Mensual")
    st.write(
        "Objetivo: calcular y visualizar la tasa de rotación mensual de la compañía "
        "para apoyar decisiones de retención del talento."
    )

    uploaded_file3 = st.file_uploader("Sube el archivo tarea3_rotacion_empleados.csv", type=["csv"], key="tarea3")

    if uploaded_file3 is not None:
        df3 = pd.read_csv(uploaded_file3)

        st.subheader("Vista inicial del dataset")
        st.dataframe(df3.head())

        df3["fecha_ingreso"] = pd.to_datetime(df3["fecha_ingreso"])
        df3["fecha_salida"] = pd.to_datetime(df3["fecha_salida"])

        bajas = df3[df3["fecha_salida"].notna()].copy()
        bajas["mes_salida"] = bajas["fecha_salida"].dt.to_period("M").astype(str)

        bajas_mensuales = (
            bajas
            .groupby("mes_salida")
            .size()
            .reset_index(name="bajas")
        )

        total_empleados = len(df3)

        bajas_mensuales["tasa_rotacion"] = (
            bajas_mensuales["bajas"] / total_empleados * 100
        ).round(2)

        st.subheader("Tasa de rotación mensual")
        st.dataframe(bajas_mensuales)

        bajas_departamento = (
            bajas
            .groupby("departamento")
            .size()
            .reset_index(name="bajas")
        )

        empleados_departamento = (
            df3
            .groupby("departamento")
            .size()
            .reset_index(name="total_empleados")
        )

        rotacion_departamento = bajas_departamento.merge(
            empleados_departamento,
            on="departamento"
        )

        rotacion_departamento["tasa_rotacion_departamento"] = (
            rotacion_departamento["bajas"] / rotacion_departamento["total_empleados"] * 100
        ).round(2)

        rotacion_departamento = rotacion_departamento.sort_values(
            "tasa_rotacion_departamento",
            ascending=False
        )

        st.subheader("Rotación por departamento")
        st.dataframe(rotacion_departamento)

        mes_mayor = bajas_mensuales.sort_values("tasa_rotacion", ascending=False).iloc[0]
        depto_mayor = rotacion_departamento.iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Mes con mayor rotación", mes_mayor["mes_salida"], f"{mes_mayor['tasa_rotacion']}%")

        with col2:
            st.metric("Departamento crítico", depto_mayor["departamento"], f"{depto_mayor['tasa_rotacion_departamento']}%")

        st.subheader("Visualización: tasa de rotación mensual")

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(
            bajas_mensuales["mes_salida"],
            bajas_mensuales["tasa_rotacion"],
            marker="o"
        )
        ax1.set_title("Tasa de rotación mensual")
        ax1.set_xlabel("Mes")
        ax1.set_ylabel("Tasa de rotación (%)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        st.pyplot(fig1)

        st.subheader("Visualización: rotación por departamento")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(
            rotacion_departamento["departamento"],
            rotacion_departamento["tasa_rotacion_departamento"]
        )
        ax2.set_title("Tasa de rotación por departamento")
        ax2.set_xlabel("Departamento")
        ax2.set_ylabel("Tasa de rotación (%)")
        ax2.tick_params(axis="x", rotation=45)

        st.pyplot(fig2)

        st.subheader("Acciones de retención sugeridas")
        st.write(
            f"1. Realizar entrevistas de salida estructuradas en **{depto_mayor['departamento']}** "
            "para identificar causas recurrentes de baja."
        )
        st.write(
            "2. Diseñar planes de retención específicos para áreas con mayor rotación, "
            "incluyendo revisión de carga laboral, desarrollo profesional y acompañamiento de managers."
        )
        st.write(
            "3. Monitorear mensualmente la rotación para detectar picos tempranos y actuar antes "
            "de que se conviertan en un problema sostenido."
        )

        st.download_button(
            label="Descargar tasa de rotación mensual",
            data=bajas_mensuales.to_csv(index=False).encode("utf-8"),
            file_name="tasa_rotacion_mensual.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Descargar rotación por departamento",
            data=rotacion_departamento.to_csv(index=False).encode("utf-8"),
            file_name="rotacion_por_departamento.csv",
            mime="text/csv"
        )

    else:
        st.info("Sube el archivo CSV para iniciar el análisis de rotación.")
