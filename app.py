# =============================================================
# PEOPLE ANALYTICS DASHBOARD — 3 TAREAS INTEGRADAS
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
.insight-box {
    background-color: #F8FAFC;
    padding: 18px;
    border-left: 5px solid #2050F6;
    border-radius: 12px;
    margin-top: 10px;
    margin-bottom: 15px;
}

.positive-box {
    background-color: #ECFDF3;
    padding: 15px;
    border-left: 5px solid #22C55E;
    border-radius: 12px;
}

.warning-box {
    background-color: #FFF7ED;
    padding: 15px;
    border-left: 5px solid #F97316;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Tarea 1 · Skills en CVs",
    "Tarea 2 · Sentimiento",
    "Tarea 3 · Rotación"
])

# =============================================================
# FUNCIONES GENERALES
# =============================================================

def validar_columnas(df, columnas_requeridas, nombre_dataset):
    faltantes = [col for col in columnas_requeridas if col not in df.columns]

    if faltantes:
        st.error(
            f"El archivo de {nombre_dataset} no tiene estas columnas: {faltantes}"
        )
        st.stop()


def mostrar_bar_labels(ax, formato="{:.0f}"):
    for p in ax.patches:
        valor = p.get_height()

        ax.annotate(
            formato.format(valor),
            (p.get_x() + p.get_width() / 2., valor),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points"
        )

# =============================================================
# TAREA 1 — EXTRACCIÓN DE SKILLS
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
    "scikit-learn": ["scikit-learn", "sklearn"],
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
    "Data Analyst": [
        "python",
        "sql",
        "excel",
        "power bi",
        "tableau",
        "estadística"
    ],

    "People Analytics Specialist": [
        "people analytics",
        "python",
        "r",
        "power bi",
        "excel",
        "análisis predictivo"
    ],

    "Machine Learning Engineer": [
        "python",
        "machine learning",
        "tensorflow",
        "scikit-learn",
        "docker",
        "git"
    ],

    "Recruiter Técnico": [
        "sourcing",
        "linkedin recruiter",
        "boolean search",
        "ats",
        "entrevista por competencias",
        "reclutamiento"
    ],

    "HR Business Partner": [
        "gestión del talento",
        "people analytics",
        "hris",
        "workday",
        "excel",
        "comunicación"
    ]
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

    match = round(
        (len(encontradas) / len(skills_req)) * 100,
        1
    ) if skills_req else 0

    return pd.Series({
        "skills_requeridas": ", ".join(sorted(skills_req)),
        "skills_encontradas_requeridas": ", ".join(sorted(encontradas)),
        "skills_faltantes": ", ".join(sorted(faltantes)),
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

    st.header("Tarea 1 — Extracción de Skills en CVs")

    uploaded_file1 = st.file_uploader(
        "Sube el archivo tarea1_cvs.csv",
        type=["csv"],
        key="csv_tarea1"
    )

    if uploaded_file1 is not None:

        df1 = pd.read_csv(uploaded_file1)

        validar_columnas(
            df1,
            ["nombre", "puesto_target", "cv_texto"],
            "Tarea 1"
        )

        df1["skills_detectadas"] = df1["cv_texto"].apply(extraer_skills)

        df1_resultado = pd.concat([
            df1,
            df1.apply(calcular_match, axis=1)
        ], axis=1)

        df1_resultado["decision_sugerida"] = (
            df1_resultado["porcentaje_match"].apply(decision)
        )

        ranking = df1_resultado[
            [
                "nombre",
                "puesto_target",
                "porcentaje_match",
                "decision_sugerida"
            ]
        ].sort_values("porcentaje_match", ascending=False)

        c1, c2, c3 = st.columns(3)

        c1.metric("Candidatos", len(df1_resultado))

        c2.metric(
            "Preseleccionados",
            (df1_resultado["decision_sugerida"] == "Preseleccionar").sum()
        )

        c3.metric(
            "Match promedio",
            f"{df1_resultado['porcentaje_match'].mean():.1f}%"
        )

        st.subheader("Ranking de candidatos")

        st.dataframe(ranking, use_container_width=True)

        st.subheader("Skills más frecuentes")

        df_skills = df1_resultado.explode("skills_detectadas")

        top_skills = (
            df_skills["skills_detectadas"]
            .value_counts()
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            top_skills.index[::-1],
            top_skills.values[::-1]
        )

        ax.set_xlabel("Frecuencia")
        ax.set_title("Skills más frecuentes")

        st.pyplot(fig)

# =============================================================
# TAREA 2 — SENTIMIENTO
# =============================================================

vader = SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=False)
def cargar_modelo_pysentimiento():
    from pysentimiento import create_analyzer

    return create_analyzer(
        task="sentiment",
        lang="es"
    )


drivers_hr = {

    "Liderazgo": [
        "jefe",
        "manager",
        "lider",
        "líder",
        "liderazgo"
    ],

    "Comunicación": [
        "comunicación",
        "comunicacion",
        "feedback",
        "escucha"
    ],

    "Carga laboral": [
        "carga",
        "sobrecarga",
        "horas",
        "presión",
        "presion"
    ],

    "Ambiente laboral": [
        "ambiente",
        "clima",
        "equipo",
        "compañeros"
    ],

    "Desarrollo / formación": [
        "formación",
        "desarrollo",
        "crecimiento",
        "aprendizaje"
    ],

    "Reconocimiento": [
        "reconocimiento",
        "valora",
        "agradece"
    ],

    "Salario / beneficios": [
        "salario",
        "sueldo",
        "beneficios",
        "bonus"
    ],

    "Cultura": [
        "cultura",
        "valores",
        "compromiso"
    ],

    "Gestión": [
        "gestión",
        "gestion",
        "procesos"
    ],

    "Estrés / burnout": [
        "estrés",
        "estres",
        "burnout",
        "agotado"
    ]
}


def detectar_drivers(texto):

    texto = str(texto).lower()

    encontrados = []

    for driver, palabras_clave in drivers_hr.items():

        for palabra in palabras_clave:

            if palabra in texto:
                encontrados.append(driver)
                break

    return encontrados if encontrados else ["Sin driver claro"]


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

    return {
        "POS": "positivo",
        "NEG": "negativo",
        "NEU": "neutro"
    }.get(resultado, "neutro")


def resumen_sentimiento(df, columna):

    resumen = (
        df.groupby([columna, "pred_pysentimiento"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["negativo", "positivo", "neutro"]:

        if col not in resumen.columns:
            resumen[col] = 0

    resumen["total"] = (
        resumen["negativo"] +
        resumen["positivo"] +
        resumen["neutro"]
    )

    resumen["pct_negativo"] = (
        resumen["negativo"] /
        resumen["total"] * 100
    ).round(1)

    resumen["pct_positivo"] = (
        resumen["positivo"] /
        resumen["total"] * 100
    ).round(1)

    return resumen


def resumen_drivers(df, sentimiento):

    df_temp = (
        df[df["pred_pysentimiento"] == sentimiento]
        .copy()
    )

    df_temp["drivers"] = (
        df_temp["comentario"]
        .apply(detectar_drivers)
    )

    df_temp = df_temp.explode("drivers")

    resumen = (
        df_temp["drivers"]
        .value_counts()
        .reset_index()
    )

    resumen.columns = ["driver", "frecuencia"]

    return resumen


def top_drivers_de_departamento(
    df,
    departamento,
    sentimiento,
    top_n=5
):

    df_dep = df[
        (df["departamento"] == departamento) &
        (df["pred_pysentimiento"] == sentimiento)
    ].copy()

    df_dep["drivers"] = (
        df_dep["comentario"]
        .apply(detectar_drivers)
    )

    df_dep = df_dep.explode("drivers")

    resumen = (
        df_dep["drivers"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    resumen.columns = ["driver", "frecuencia"]

    return resumen


with tab2:

    st.header("Tarea 2 — Análisis de Sentimiento")

    uploaded_file2 = st.file_uploader(
        "Sube tarea2_encuesta_clima.csv",
        type=["csv"],
        key="csv_tarea2"
    )

    if uploaded_file2 is not None:

        df2 = pd.read_csv(uploaded_file2)

        validar_columnas(
            df2,
            [
                "comentario",
                "departamento",
                "nivel",
                "sentimiento_real"
            ],
            "Tarea 2"
        )

        analizador_es = cargar_modelo_pysentimiento()

        df2["pred_textblob"] = (
            df2["comentario"]
            .apply(clasificar_textblob)
        )

        df2["pred_vader"] = (
            df2["comentario"]
            .apply(clasificar_vader)
        )

        df2["pred_pysentimiento"] = (
            df2["comentario"]
            .apply(
                lambda x:
                clasificar_pysentimiento(
                    x,
                    analizador_es
                )
            )
        )

        acc_pys = accuracy_score(
            df2["sentimiento_real"],
            df2["pred_pysentimiento"]
        )

        total = len(df2)

        positivos = (
            df2["pred_pysentimiento"]
            .eq("positivo")
            .sum()
        )

        negativos = (
            df2["pred_pysentimiento"]
            .eq("negativo")
            .sum()
        )

        neutros = (
            df2["pred_pysentimiento"]
            .eq("neutro")
            .sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Comentarios", total)

        c2.metric(
            "% positivo",
            f"{positivos/total*100:.1f}%"
        )

        c3.metric(
            "% negativo",
            f"{negativos/total*100:.1f}%"
        )

        c4.metric(
            "Accuracy",
            f"{acc_pys:.1%}"
        )

        resumen_depto = resumen_sentimiento(
            df2,
            "departamento"
        )

        st.subheader("Sentimiento negativo por departamento")

        datos_neg = (
            resumen_depto
            .sort_values("pct_negativo", ascending=True)
        )

        fig2, ax2 = plt.subplots(figsize=(9, 5))

        ax2.barh(
            datos_neg["departamento"],
            datos_neg["pct_negativo"]
        )

        for i, v in enumerate(datos_neg["pct_negativo"]):

            ax2.text(
                v + 0.5,
                i,
                f"{v:.1f}%",
                va="center"
            )

        st.pyplot(fig2)

        st.subheader("Drivers generales")

        drivers_pos = resumen_drivers(df2, "positivo")
        drivers_neg = resumen_drivers(df2, "negativo")

        col1, col2 = st.columns(2)

        with col1:

            fig3, ax3 = plt.subplots(figsize=(8, 5))

            ax3.barh(
                drivers_pos["driver"][::-1],
                drivers_pos["frecuencia"][::-1]
            )

            ax3.set_title("Drivers positivos")

            st.pyplot(fig3)

        with col2:

            fig4, ax4 = plt.subplots(figsize=(8, 5))

            ax4.barh(
                drivers_neg["driver"][::-1],
                drivers_neg["frecuencia"][::-1]
            )

            ax4.set_title("Drivers negativos")

            st.pyplot(fig4)

        st.subheader("Drivers por departamento")

        departamento_seleccionado = st.selectbox(
            "Selecciona un departamento",
            sorted(df2["departamento"].unique())
        )

        col_neg, col_pos = st.columns(2)

        with col_neg:

            st.markdown(
                f"### Negativos — {departamento_seleccionado}"
            )

            drivers_neg_dep = top_drivers_de_departamento(
                df2,
                departamento_seleccionado,
                "negativo"
            )

            fig5, ax5 = plt.subplots(figsize=(7, 4))

            ax5.barh(
                drivers_neg_dep["driver"][::-1],
                drivers_neg_dep["frecuencia"][::-1]
            )

            st.pyplot(fig5)

            st.dataframe(
                drivers_neg_dep,
                use_container_width=True
            )

        with col_pos:

            st.markdown(
                f"### Positivos — {departamento_seleccionado}"
            )

            drivers_pos_dep = top_drivers_de_departamento(
                df2,
                departamento_seleccionado,
                "positivo"
            )

            fig6, ax6 = plt.subplots(figsize=(7, 4))

            ax6.barh(
                drivers_pos_dep["driver"][::-1],
                drivers_pos_dep["frecuencia"][::-1]
            )

            st.pyplot(fig6)

            st.dataframe(
                drivers_pos_dep,
                use_container_width=True
            )

# =============================================================
# TAREA 3 — ROTACIÓN
# =============================================================

with tab3:

    st.header("Tarea 3 — Rotación")

    uploaded_file3 = st.file_uploader(
        "Sube tarea3_rotacion_empleados.csv",
        type=["csv"],
        key="csv_tarea3"
    )

    if uploaded_file3 is not None:

        df3 = pd.read_csv(uploaded_file3)

        validar_columnas(
            df3,
            [
                "fecha_ingreso",
                "fecha_salida",
                "departamento"
            ],
            "Tarea 3"
        )

        df3["fecha_ingreso"] = pd.to_datetime(
            df3["fecha_ingreso"],
            errors="coerce"
        )

        df3["fecha_salida"] = pd.to_datetime(
            df3["fecha_salida"],
            errors="coerce"
        )

        bajas = df3[df3["fecha_salida"].notna()].copy()

        bajas["mes_salida"] = (
            bajas["fecha_salida"]
            .dt.to_period("M")
            .astype(str)
        )

        bajas_mensuales = (
            bajas.groupby("mes_salida")
            .size()
            .reset_index(name="bajas")
        )

        total_empleados = len(df3)

        bajas_mensuales["tasa_rotacion"] = (
            bajas_mensuales["bajas"] /
            total_empleados * 100
        ).round(2)

        fig7, ax7 = plt.subplots(figsize=(9, 5))

        ax7.plot(
            bajas_mensuales["mes_salida"],
            bajas_mensuales["tasa_rotacion"],
            marker="o"
        )

        ax7.set_xlabel("Mes")
        ax7.set_ylabel("Tasa de rotación (%)")
        ax7.set_title("Rotación mensual")

        ax7.tick_params(axis="x", rotation=45)

        st.pyplot(fig7)

        st.dataframe(
            bajas_mensuales,
            use_container_width=True
        )
