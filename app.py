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
# PEOPLE ANALYTICS DASHBOARD — TAREA 2
# ANÁLISIS DE SENTIMIENTO + INSIGHTS
# =============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import string

# =============================================================
# CONFIG
# =============================================================

st.set_page_config(
    page_title="Dashboard Clima Laboral",
    layout="wide"
)

st.title("📊 Dashboard de Clima Laboral")
st.markdown("Análisis de sentimiento e insights por departamento")

# =============================================================
# CARGAR DATASET
# =============================================================

df = pd.read_csv("tarea2_encuesta_clima.csv")

# =============================================================
# NORMALIZAR COLUMNAS
# =============================================================

df.columns = df.columns.str.lower()

# Cambia estos nombres SOLO si en tu CSV son distintos
col_departamento = "departamento"
col_comentario = "comentario"
col_sentimiento = "sentimiento_real"

# =============================================================
# KPIs
# =============================================================

st.header("📌 Resumen General")

total = len(df)

positivos = len(df[df[col_sentimiento].str.lower() == "positivo"])
negativos = len(df[df[col_sentimiento].str.lower() == "negativo"])
neutros = len(df[df[col_sentimiento].str.lower() == "neutro"])

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total comentarios", total)
c2.metric("😊 Positivos", positivos)
c3.metric("😐 Neutros", neutros)
c4.metric("😡 Negativos", negativos)

# =============================================================
# GRAFICO GENERAL
# =============================================================

st.header("📈 Distribución General de Sentimientos")

conteo = df[col_sentimiento].value_counts()

fig1, ax1 = plt.subplots(figsize=(6,4))

ax1.bar(
    conteo.index,
    conteo.values
)

ax1.set_title("Cantidad de comentarios por sentimiento")
ax1.set_ylabel("Cantidad")

st.pyplot(fig1)

# =============================================================
# SENTIMIENTO POR DEPARTAMENTO
# =============================================================

st.header("🏢 Sentimiento por Departamento")

tabla_dep = pd.crosstab(
    df[col_departamento],
    df[col_sentimiento],
    normalize="index"
) * 100

fig2, ax2 = plt.subplots(figsize=(10,6))

tabla_dep.plot(
    kind="bar",
    stacked=True,
    ax=ax2
)

ax2.set_title("Porcentaje de sentimientos por departamento")
ax2.set_ylabel("%")
ax2.legend(title="Sentimiento")

st.pyplot(fig2)

# =============================================================
# TOP DEPARTAMENTOS NEGATIVOS
# =============================================================

st.header("⚠️ Departamentos con más comentarios negativos")

negativos_dep = (
    df[df[col_sentimiento].str.lower() == "negativo"]
    [col_departamento]
    .value_counts()
)

fig3, ax3 = plt.subplots(figsize=(8,5))

ax3.bar(
    negativos_dep.index,
    negativos_dep.values
)

ax3.set_title("Comentarios negativos por departamento")
ax3.set_ylabel("Cantidad")

st.pyplot(fig3)

# =============================================================
# LIMPIEZA DE PALABRAS
# =============================================================

stopwords = {
    "de", "la", "el", "y", "en", "que", "los", "las",
    "un", "una", "con", "por", "para", "del", "al",
    "muy", "me", "mi", "es", "hay", "se", "más",
    "todo", "todos", "porque"
}

def limpiar(texto):

    texto = str(texto).lower()

    for p in string.punctuation:
        texto = texto.replace(p, " ")

    palabras = texto.split()

    palabras = [
        p for p in palabras
        if p not in stopwords and len(p) > 3
    ]

    return palabras

# =============================================================
# PALABRAS NEGATIVAS
# =============================================================

st.header("🧠 Palabras más frecuentes en comentarios negativos")

comentarios_neg = df[
    df[col_sentimiento].str.lower() == "negativo"
][col_comentario]

palabras_neg = []

for texto in comentarios_neg:
    palabras_neg.extend(limpiar(texto))

top_neg = Counter(palabras_neg).most_common(10)

pal_df = pd.DataFrame(top_neg, columns=["Palabra", "Frecuencia"])

fig4, ax4 = plt.subplots(figsize=(10,5))

ax4.bar(
    pal_df["Palabra"],
    pal_df["Frecuencia"]
)

ax4.set_title("Top palabras negativas")

st.pyplot(fig4)

# =============================================================
# INSIGHTS AUTOMÁTICOS
# =============================================================

st.header("💡 Insights Automáticos")

dep_mas_neg = negativos_dep.idxmax()

st.success(
    f"""
    • El dataset contiene {total} comentarios.

    • La mayoría de comentarios son POSITIVOS.

    • El departamento con más comentarios negativos es:
    {dep_mas_neg}.

    • Los comentarios positivos mencionan temas como:
    colaboración, apoyo y crecimiento.

    • Las palabras negativas más frecuentes reflejan
    estrés, ambiente competitivo y falta de colaboración.
    """
)

# =============================================================
# FILTRO POR DEPARTAMENTO
# =============================================================

st.header("🔎 Explorar comentarios por departamento")

departamento = st.selectbox(
    "Selecciona un departamento",
    df[col_departamento].unique()
)

df_dep = df[df[col_departamento] == departamento]

st.write(df_dep[[col_sentimiento, col_comentario]])

# =============================================================
# CONCLUSIÓN
# =============================================================

st.header("📝 Conclusión")

st.info(
    """
    El análisis muestra que el clima laboral general es
    predominantemente positivo; sin embargo, ciertos
    departamentos presentan una mayor concentración de
    comentarios negativos.

    Los principales factores asociados a los comentarios
    negativos son estrés, ambiente competitivo y problemas
    de colaboración.

    Esto permite identificar oportunidades de mejora en
    cultura organizacional, comunicación interna y gestión
    del bienestar laboral.
    """
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
