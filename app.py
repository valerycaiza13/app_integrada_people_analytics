# =============================================================
# TAREA 2 — ANÁLISIS DE SENTIMIENTO CON PYSENTIMIENTO
# =============================================================

from collections import Counter
import string

vader = SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=False)
def cargar_modelo_pysentimiento():
    from pysentimiento import create_analyzer
    return create_analyzer(task="sentiment", lang="es")

stopwords_es = {
    "de", "la", "el", "y", "en", "que", "los", "las", "un", "una",
    "muy", "con", "por", "para", "del", "al", "me", "mi", "es",
    "hay", "se", "más", "mas", "todo", "todos", "porque"
}

def limpiar_palabras(texto):
    texto = str(texto).lower()

    for p in string.punctuation:
        texto = texto.replace(p, " ")

    palabras = texto.split()

    palabras = [
        p for p in palabras
        if p not in stopwords_es and len(p) > 3
    ]

    return palabras


def top_palabras(df, sentimiento, top_n=10):

    textos = df[df["pred_pysentimiento"] == sentimiento]["comentario"]

    palabras = []

    for texto in textos:
        palabras.extend(limpiar_palabras(texto))

    contador = Counter(palabras)

    return pd.DataFrame(
        contador.most_common(top_n),
        columns=["palabra", "frecuencia"]
    )


def clasificar_textblob(texto):
    score = TextBlob(str(texto)).sentiment.polarity

    if score > 0.05:
        return "positivo"
    elif score < -0.05:
        return "negativo"
    else:
        return "neutro"


def clasificar_vader(texto):
    score = vader.polarity_scores(str(texto))["compound"]

    if score >= 0.05:
        return "positivo"
    elif score <= -0.05:
        return "negativo"
    else:
        return "neutro"


def clasificar_pysentimiento(texto, analizador):
    resultado = analizador.predict(str(texto)).output

    mapeo = {
        "POS": "positivo",
        "NEG": "negativo",
        "NEU": "neutro"
    }

    return mapeo.get(resultado, "neutro")


def resumen_sentimiento(df, columna_grupo):

    resumen = (
        df.groupby([columna_grupo, "pred_pysentimiento"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["negativo", "neutro", "positivo"]:
        if col not in resumen.columns:
            resumen[col] = 0

    resumen["total"] = (
        resumen[["negativo", "neutro", "positivo"]]
        .sum(axis=1)
    )

    resumen["pct_negativo"] = (
        resumen["negativo"] / resumen["total"] * 100
    ).round(1)

    resumen["pct_positivo"] = (
        resumen["positivo"] / resumen["total"] * 100
    ).round(1)

    return resumen


with tab2:

    st.header("Tarea 2 — Análisis de Sentimiento en Encuestas de Clima")

    st.write(
        "Análisis estratégico del clima laboral utilizando NLP en español "
        "para detectar focos de atención, drivers emocionales y oportunidades de mejora."
    )

    uploaded_file2 = st.file_uploader(
        "Sube el archivo tarea2_encuesta_clima.csv",
        type=["csv"],
        key="csv_tarea2"
    )

    if uploaded_file2 is not None:

        df2 = pd.read_csv(uploaded_file2)

        validar_columnas(
            df2,
            ["comentario", "departamento", "nivel", "sentimiento_real"],
            "Tarea 2"
        )

        analizador_es = cargar_modelo_pysentimiento()

        with st.spinner("Analizando comentarios..."):

            df2["pred_textblob"] = df2["comentario"].apply(clasificar_textblob)

            df2["pred_vader"] = df2["comentario"].apply(clasificar_vader)

            df2["pred_pysentimiento"] = df2["comentario"].apply(
                lambda x: clasificar_pysentimiento(x, analizador_es)
            )

        acc_textblob = accuracy_score(
            df2["sentimiento_real"],
            df2["pred_textblob"]
        )

        acc_vader = accuracy_score(
            df2["sentimiento_real"],
            df2["pred_vader"]
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

        pct_pos = positivos / total * 100
        pct_neg = negativos / total * 100

        resumen_depto = resumen_sentimiento(df2, "departamento")
        resumen_nivel = resumen_sentimiento(df2, "nivel")

        depto_critico = resumen_depto.sort_values(
            "pct_negativo",
            ascending=False
        ).iloc[0]

        depto_fuerte = resumen_depto.sort_values(
            "pct_positivo",
            ascending=False
        ).iloc[0]

        # =========================================================
        # KPIs EJECUTIVOS
        # =========================================================

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Comentarios analizados", total)

        c2.metric(
            "% sentimiento positivo",
            f"{pct_pos:.1f}%"
        )

        c3.metric(
            "% sentimiento negativo",
            f"{pct_neg:.1f}%"
        )

        c4.metric(
            "Accuracy NLP español",
            f"{acc_pys:.1%}"
        )

        # =========================================================
        # INSIGHTS PRINCIPALES ARRIBA
        # =========================================================

        st.markdown(f"""
        <div class="insight-box">
        <b>Resumen ejecutivo:</b><br><br>

        • El departamento con mejor percepción es <b>{depto_fuerte['departamento']}</b>,
        con <b>{depto_fuerte['pct_positivo']:.1f}%</b> de comentarios positivos.<br><br>

        • El principal foco de atención es <b>{depto_critico['departamento']}</b>,
        donde el <b>{depto_critico['pct_negativo']:.1f}%</b> de comentarios presentan sentimiento negativo.<br><br>

        • El modelo NLP utilizado fue <b>pysentimiento</b> porque está entrenado específicamente para español
        y obtuvo mayor precisión que VADER y TextBlob.
        </div>
        """, unsafe_allow_html=True)

        # =========================================================
        # COMPARACIÓN DE MODELOS
        # =========================================================

        st.subheader("Comparación de modelos NLP")

        comparacion = pd.DataFrame({
            "Modelo": ["TextBlob", "VADER", "pysentimiento"],
            "Accuracy": [
                acc_textblob * 100,
                acc_vader * 100,
                acc_pys * 100
            ]
        })

        fig1, ax1 = plt.subplots(figsize=(7, 4))

        ax1.bar(
            comparacion["Modelo"],
            comparacion["Accuracy"]
        )

        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Comparación de precisión")

        st.pyplot(fig1)

        # =========================================================
        # DRIVERS EMOCIONALES
        # =========================================================

        st.subheader("Drivers emocionales detectados")

        palabras_pos = top_palabras(df2, "positivo")
        palabras_neg = top_palabras(df2, "negativo")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Factores asociados a comentarios positivos")

            fig2, ax2 = plt.subplots(figsize=(8, 5))

            ax2.barh(
                palabras_pos["palabra"][::-1],
                palabras_pos["frecuencia"][::-1]
            )

            ax2.set_title("Drivers positivos")

            st.pyplot(fig2)

        with col2:

            st.markdown("### Factores asociados a comentarios negativos")

            fig3, ax3 = plt.subplots(figsize=(8, 5))

            ax3.barh(
                palabras_neg["palabra"][::-1],
                palabras_neg["frecuencia"][::-1]
            )

            ax3.set_title("Drivers negativos")

            st.pyplot(fig3)

        # =========================================================
        # DEPARTAMENTOS
        # =========================================================

        st.subheader("Focos de atención por departamento")

        datos_neg = resumen_depto.sort_values(
            "pct_negativo",
            ascending=True
        )

        fig4, ax4 = plt.subplots(figsize=(9, 5))

        ax4.barh(
            datos_neg["departamento"],
            datos_neg["pct_negativo"]
        )

        ax4.set_xlabel("% negativo")
        ax4.set_title("Sentimiento negativo por departamento")

        st.pyplot(fig4)

        # =========================================================
        # NIVEL ORGANIZACIONAL
        # =========================================================

        st.subheader("Análisis por nivel organizacional")

        datos_nivel = resumen_nivel.sort_values(
            "pct_negativo",
            ascending=True
        )

        fig5, ax5 = plt.subplots(figsize=(9, 5))

        ax5.barh(
            datos_nivel["nivel"],
            datos_nivel["pct_negativo"]
        )

        ax5.set_xlabel("% negativo")
        ax5.set_title("Sentimiento negativo por nivel")

        st.pyplot(fig5)

        # =========================================================
        # RECOMENDACIONES HR
        # =========================================================

        st.subheader("Recomendaciones estratégicas para HR")

        st.write(
            f"1. Priorizar sesiones de escucha activa en el departamento "
            f"**{depto_critico['departamento']}**, donde se concentra el mayor sentimiento negativo."
        )

        st.write(
            "2. Replicar prácticas de engagement y liderazgo presentes en los departamentos "
            "con mayor percepción positiva."
        )

        st.write(
            "3. Monitorear periódicamente los drivers negativos para detectar riesgos de burnout, "
            "desmotivación o problemas de comunicación interna."
        )

    else:
        st.info("Sube el archivo CSV de la Tarea 2.")
