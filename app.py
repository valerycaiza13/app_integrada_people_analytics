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
# DASHBOARD — ANÁLISIS DE SENTIMIENTO CLIMA LABORAL
# =============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import string

# =============================================================
# CONFIGURACIÓN
# =============================================================

st.set_page_config(
    page_title="Dashboard Clima Laboral",
    layout="wide"
)

st.title("📊 Dashboard de Clima Laboral")
st.markdown("Análisis de sentimiento por departamento con insights automáticos")

# =============================================================
# CARGAR DATASET
# =============================================================

df = pd.read_csv("tarea2_encuesta_clima.csv")

df.columns = df.columns.str.lower()

col_departamento = "departamento"
col_comentario = "comentario"
col_sentimiento = "sentimiento_real"

df[col_sentimiento] = df[col_sentimiento].str.lower()
df[col_departamento] = df[col_departamento].astype(str)

# =============================================================
# STOPWORDS
# =============================================================

stopwords = {
    "de", "la", "el", "y", "en", "que", "los", "las", "un", "una",
    "con", "por", "para", "del", "al", "muy", "me", "mi", "es",
    "hay", "se", "más", "mas", "todo", "todos", "porque", "pero",
    "como", "son", "sin", "las", "los", "este", "esta", "eso",
    "ser", "fue", "han", "ha", "lo", "su", "sus", "nos", "también",
    "tambien", "cada", "aunque", "cuando", "sobre", "entre"
}

def limpiar_palabras(texto):
    texto = str(texto).lower()

    for p in string.punctuation:
        texto = texto.replace(p, " ")

    palabras = texto.split()

    palabras = [
        p for p in palabras
        if p not in stopwords and len(p) > 3
    ]

    return palabras

def obtener_top_palabras(dataframe, sentimiento, top_n=10):
    comentarios = dataframe[dataframe[col_sentimiento] == sentimiento][col_comentario]

    palabras = []

    for comentario in comentarios:
        palabras.extend(limpiar_palabras(comentario))

    return Counter(palabras).most_common(top_n)

def grafico_barras_horizontal(data, titulo, xlabel):
    palabras = [x[0] for x in data]
    frecuencias = [x[1] for x in data]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.barh(palabras, frecuencias)
    ax.set_title(titulo, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

    plt.tight_layout()

    return fig

# =============================================================
# KPIs GENERALES
# =============================================================

st.header("📌 Resumen general")

total = len(df)
positivos = len(df[df[col_sentimiento] == "positivo"])
negativos = len(df[df[col_sentimiento] == "negativo"])
neutros = len(df[df[col_sentimiento] == "neutro"])

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total comentarios", total)
c2.metric("😊 Positivos", positivos)
c3.metric("😐 Neutros", neutros)
c4.metric("😡 Negativos", negativos)

# =============================================================
# DISTRIBUCIÓN GENERAL
# =============================================================

st.header("📈 Distribución general de sentimientos")

conteo_general = df[col_sentimiento].value_counts()

fig1, ax1 = plt.subplots(figsize=(6, 4))

ax1.bar(conteo_general.index, conteo_general.values)
ax1.set_title("Cantidad de comentarios por sentimiento")
ax1.set_ylabel("Cantidad")

plt.tight_layout()
st.pyplot(fig1)

# =============================================================
# SENTIMIENTO POR DEPARTAMENTO
# =============================================================

st.header("🏢 Sentimiento por departamento")

tabla_dep = pd.crosstab(
    df[col_departamento],
    df[col_sentimiento],
    normalize="index"
) * 100

fig2, ax2 = plt.subplots(figsize=(12, 6))

tabla_dep.plot(
    kind="bar",
    stacked=True,
    ax=ax2
)

ax2.set_title("Porcentaje de sentimientos por departamento")
ax2.set_ylabel("Porcentaje")
ax2.set_xlabel("Departamento")
ax2.legend(title="Sentimiento", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

st.pyplot(fig2)

# =============================================================
# COMENTARIOS NEGATIVOS POR DEPARTAMENTO
# =============================================================

st.header("⚠️ Comentarios negativos por departamento")

negativos_dep = (
    df[df[col_sentimiento] == "negativo"]
    [col_departamento]
    .value_counts()
)

fig3, ax3 = plt.subplots(figsize=(10, 5))

ax3.barh(negativos_dep.index, negativos_dep.values)
ax3.set_title("Cantidad de comentarios negativos por departamento")
ax3.set_xlabel("Cantidad de comentarios negativos")
ax3.invert_yaxis()

plt.tight_layout()

st.pyplot(fig3)

# =============================================================
# SELECTOR DE DEPARTAMENTO
# =============================================================

st.header("🔎 Análisis detallado por departamento")

departamento_seleccionado = st.selectbox(
    "Selecciona un departamento",
    sorted(df[col_departamento].unique())
)

df_dep = df[df[col_departamento] == departamento_seleccionado]

total_dep = len(df_dep)
pos_dep = len(df_dep[df_dep[col_sentimiento] == "positivo"])
neg_dep = len(df_dep[df_dep[col_sentimiento] == "negativo"])
neu_dep = len(df_dep[df_dep[col_sentimiento] == "neutro"])

porc_pos = round((pos_dep / total_dep) * 100, 1)
porc_neg = round((neg_dep / total_dep) * 100, 1)
porc_neu = round((neu_dep / total_dep) * 100, 1)

d1, d2, d3, d4 = st.columns(4)

d1.metric("Total comentarios", total_dep)
d2.metric("😊 Positivos", f"{porc_pos}%")
d3.metric("😐 Neutros", f"{porc_neu}%")
d4.metric("😡 Negativos", f"{porc_neg}%")

# =============================================================
# GRÁFICAS DE PALABRAS POR DEPARTAMENTO
# =============================================================

st.subheader(f"🧠 Palabras más frecuentes en {departamento_seleccionado}")

top_positivas = obtener_top_palabras(df_dep, "positivo", 10)
top_negativas = obtener_top_palabras(df_dep, "negativo", 10)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 😊 Palabras en comentarios positivos")

    if top_positivas:
        fig_pos = grafico_barras_horizontal(
            top_positivas,
            "Top palabras positivas",
            "Frecuencia"
        )
        st.pyplot(fig_pos)
    else:
        st.info("No hay comentarios positivos para este departamento.")

with col2:
    st.markdown("### 😡 Palabras en comentarios negativos")

    if top_negativas:
        fig_neg = grafico_barras_horizontal(
            top_negativas,
            "Top palabras negativas",
            "Frecuencia"
        )
        st.pyplot(fig_neg)
    else:
        st.info("No hay comentarios negativos para este departamento.")

# =============================================================
# INSIGHTS AUTOMÁTICOS POR DEPARTAMENTO
# =============================================================

st.subheader("💡 Insights del departamento seleccionado")

if porc_neg >= 40:
    nivel_riesgo = "alto"
    mensaje_riesgo = "Este departamento presenta una proporción alta de comentarios negativos, por lo que podría requerir una revisión prioritaria del clima laboral."
elif porc_neg >= 25:
    nivel_riesgo = "medio"
    mensaje_riesgo = "Este departamento presenta algunos focos de alerta que podrían analizarse con mayor detalle."
else:
    nivel_riesgo = "bajo"
    mensaje_riesgo = "Este departamento presenta un nivel bajo de comentarios negativos en comparación con el total de respuestas."

palabras_pos_txt = ", ".join([x[0] for x in top_positivas[:5]]) if top_positivas else "no se identificaron palabras positivas frecuentes"
palabras_neg_txt = ", ".join([x[0] for x in top_negativas[:5]]) if top_negativas else "no se identificaron palabras negativas frecuentes"

st.info(
    f"""
    En el departamento de **{departamento_seleccionado}** se analizaron **{total_dep} comentarios**.

    El **{porc_pos}%** de los comentarios son positivos, el **{porc_neu}%** son neutros y el **{porc_neg}%** son negativos.

    Las palabras positivas más frecuentes son: **{palabras_pos_txt}**.

    Las palabras negativas más frecuentes son: **{palabras_neg_txt}**.

    Nivel de alerta del departamento: **{nivel_riesgo.upper()}**.

    {mensaje_riesgo}
    """
)

# =============================================================
# COMENTARIOS REALES
# =============================================================

st.subheader("📝 Comentarios reales del departamento")

tab1, tab2, tab3 = st.tabs(["😊 Positivos", "😐 Neutros", "😡 Negativos"])

with tab1:
    comentarios_pos = df_dep[df_dep[col_sentimiento] == "positivo"][[col_comentario]]

    if len(comentarios_pos) > 0:
        st.dataframe(comentarios_pos, use_container_width=True)
    else:
        st.info("No hay comentarios positivos.")

with tab2:
    comentarios_neu = df_dep[df_dep[col_sentimiento] == "neutro"][[col_comentario]]

    if len(comentarios_neu) > 0:
        st.dataframe(comentarios_neu, use_container_width=True)
    else:
        st.info("No hay comentarios neutros.")

with tab3:
    comentarios_neg = df_dep[df_dep[col_sentimiento] == "negativo"][[col_comentario]]

    if len(comentarios_neg) > 0:
        st.dataframe(comentarios_neg, use_container_width=True)
    else:
        st.info("No hay comentarios negativos.")

# =============================================================
# CONCLUSIÓN GENERAL
# =============================================================

st.header("📝 Conclusión general")

st.success(
    """
    El análisis muestra que el clima laboral general tiene una tendencia positiva,
    pero algunos departamentos concentran una mayor proporción de comentarios negativos.

    El análisis por departamento permite identificar no solo cuántos comentarios son
    positivos o negativos, sino también qué palabras se repiten con mayor frecuencia.
    Esto ayuda a interpretar mejor los posibles factores detrás del clima laboral,
    como colaboración, ambiente, crecimiento, estrés o comunicación.
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
