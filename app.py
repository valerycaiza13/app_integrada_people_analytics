# =============================================================
# PEOPLE ANALYTICS DASHBOARD — 3 TAREAS INTEGRADAS
# Tarea 1: Extracción de skills en CVs
# Tarea 2: Análisis de sentimiento con VADER
# Tarea 3: Rotación mensual
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------------
st.set_page_config(page_title="People Analytics Dashboard", layout="wide")

st.title("People Analytics Dashboard")
st.caption("Dashboard integrado para extracción de skills, análisis de sentimiento y rotación mensual.")

st.markdown("""
<style>
.metric-card {
    background-color: #F3F5FE;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E3E8FF;
}
.insight-box {
    background-color: #F8FAFC;
    padding: 18px;
    border-left: 5px solid #2050F6;
    border-radius: 12px;
    margin-top: 10px;
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
# TAREA 1 — EXTRACCIÓN DE SKILLS EN CVs
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
    st.header("Tarea 1 — Extracción de Skills en Currículums")
    st.write("Automatiza la preselección detectando habilidades clave en CVs de texto libre.")

    uploaded_file1 = st.file_uploader("Sube el archivo tarea1_cvs.csv", type=["csv"], key="csv_tarea1")

    if uploaded_file1 is not None:
        df1 = pd.read_csv(uploaded_file1)
        validar_columnas(df1, ["nombre", "puesto_target", "cv_texto"], "Tarea 1")

        df1["skills_detectadas"] = df1["cv_texto"].apply(extraer_skills)
        resultado_match = df1.apply(calcular_match, axis=1)
        df1_resultado = pd.concat([df1, resultado_match], axis=1)
        df1_resultado["decision_sugerida"] = df1_resultado["porcentaje_match"].apply(decision)

        total_candidatos = len(df1_resultado)
        preseleccionados = (df1_resultado["decision_sugerida"] == "Preseleccionar").sum()
        match_promedio = df1_resultado["porcentaje_match"].mean()
        skill_mas_frecuente = (
            df1_resultado.explode("skills_detectadas")["skills_detectadas"]
            .dropna().value_counts().idxmax()
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidatos analizados", total_candidatos)
        c2.metric("Preseleccionados", preseleccionados)
        c3.metric("Match promedio", f"{match_promedio:.1f}%")
        c4.metric("Skill más frecuente", skill_mas_frecuente)

        ranking = df1_resultado[[
            "nombre", "puesto_target", "porcentaje_match", "skills_encontradas_requeridas",
            "skills_faltantes", "decision_sugerida"
        ]].sort_values("porcentaje_match", ascending=False)

        st.subheader("Ranking de candidatos")
        st.dataframe(ranking, use_container_width=True)

        df_skills = df1_resultado.explode("skills_detectadas")
        tabla_frecuencia = (
            df_skills.groupby(["puesto_target", "skills_detectadas"])
            .size().reset_index(name="frecuencia")
            .sort_values(["puesto_target", "frecuencia"], ascending=[True, False])
        )

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

        mejor_candidato = ranking.iloc[0]
        st.markdown(f"""
        <div class="insight-box">
        <b>Insight para selección:</b><br>
        El mejor match es <b>{mejor_candidato['nombre']}</b> para el perfil <b>{mejor_candidato['puesto_target']}</b>,
        con un <b>{mejor_candidato['porcentaje_match']}%</b> de coincidencia. Este análisis permite priorizar candidatos,
        detectar brechas de skills y reducir la revisión manual inicial.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Ver dataset procesado"):
            st.dataframe(df1_resultado, use_container_width=True)

    else:
        st.info("Sube el archivo CSV de la Tarea 1 para iniciar el análisis.")

# =============================================================
# TAREA 2 — ANÁLISIS DE SENTIMIENTO CON VADER
# =============================================================

vader = SentimentIntensityAnalyzer()

def clasificar_vader(texto):
    score = vader.polarity_scores(str(texto))["compound"]
    if score >= 0.05:
        return "positivo"
    elif score <= -0.05:
        return "negativo"
    else:
        return "neutro"


def score_vader(texto):
    return vader.polarity_scores(str(texto))["compound"]


def resumen_sentimiento(df, columna_grupo, columna_sentimiento):
    resumen = (
        df.groupby([columna_grupo, columna_sentimiento])
        .size().unstack(fill_value=0).reset_index()
    )
    for col in ["negativo", "neutro", "positivo"]:
        if col not in resumen.columns:
            resumen[col] = 0
    resumen["total"] = resumen[["negativo", "neutro", "positivo"]].sum(axis=1)
    resumen["pct_negativo"] = (resumen["negativo"] / resumen["total"] * 100).round(1)
    resumen["pct_neutro"] = (resumen["neutro"] / resumen["total"] * 100).round(1)
    resumen["pct_positivo"] = (resumen["positivo"] / resumen["total"] * 100).round(1)
    return resumen

with tab2:
    st.header("Tarea 2 — Análisis de Sentimiento en Encuestas de Clima Laboral")
    st.write("Clasificación de comentarios como positivos, negativos o neutros usando VADER, con comparación frente al sentimiento real.")

    uploaded_file2 = st.file_uploader("Sube el archivo tarea2_encuesta_clima.csv", type=["csv"], key="csv_tarea2")

    if uploaded_file2 is not None:
        df2 = pd.read_csv(uploaded_file2)
        validar_columnas(df2, ["id_respuesta", "departamento", "nivel", "comentario", "sentimiento_real"], "Tarea 2")

        df2["vader_score"] = df2["comentario"].apply(score_vader)
        df2["sentimiento_predicho_vader"] = df2["comentario"].apply(clasificar_vader)

        precision = accuracy_score(df2["sentimiento_real"], df2["sentimiento_predicho_vader"])

        dist_real = df2["sentimiento_real"].value_counts()
        total = len(df2)
        positivos = dist_real.get("positivo", 0)
        negativos = dist_real.get("negativo", 0)
        neutros = dist_real.get("neutro", 0)

        pct_pos = positivos / total * 100
        pct_neg = negativos / total * 100
        pct_neu = neutros / total * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Respuestas", total)
        c2.metric("Positivos reales", f"{pct_pos:.1f}%", f"{positivos} comentarios")
        c3.metric("Negativos reales", f"{pct_neg:.1f}%", f"{negativos} comentarios")
        c4.metric("Neutros reales", f"{pct_neu:.1f}%", f"{neutros} comentarios")
        c5.metric("Precisión VADER", f"{precision:.1%}")

        resumen_depto_real = resumen_sentimiento(df2, "departamento", "sentimiento_real")
        resumen_nivel_real = resumen_sentimiento(df2, "nivel", "sentimiento_real")
        resumen_depto_vader = resumen_sentimiento(df2, "departamento", "sentimiento_predicho_vader")

        depto_mas_positivo = resumen_depto_real.sort_values("pct_positivo", ascending=False).iloc[0]
        depto_mas_negativo = resumen_depto_real.sort_values("pct_negativo", ascending=False).iloc[0]
        nivel_mas_positivo = resumen_nivel_real.sort_values("pct_positivo", ascending=False).iloc[0]
        nivel_mas_negativo = resumen_nivel_real.sort_values("pct_negativo", ascending=False).iloc[0]

        st.subheader("Insights ejecutivos para el CHRO")
        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown(f"""
            <div class="positive-box">
            <b>Fortaleza positiva detectada</b><br>
            El departamento con mayor sentimiento positivo es <b>{depto_mas_positivo['departamento']}</b>
            con <b>{depto_mas_positivo['pct_positivo']:.1f}%</b> de comentarios positivos.
            El nivel con mejor percepción es <b>{nivel_mas_positivo['nivel']}</b>
            con <b>{nivel_mas_positivo['pct_positivo']:.1f}%</b> positivo.
            </div>
            """, unsafe_allow_html=True)
        with col_neg:
            st.markdown(f"""
            <div class="warning-box">
            <b>Foco de atención</b><br>
            El departamento con mayor sentimiento negativo es <b>{depto_mas_negativo['departamento']}</b>
            con <b>{depto_mas_negativo['pct_negativo']:.1f}%</b> de comentarios negativos.
            El nivel más afectado es <b>{nivel_mas_negativo['nivel']}</b>
            con <b>{nivel_mas_negativo['pct_negativo']:.1f}%</b> negativo.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box">
        <b>Lectura del modelo:</b><br>
        VADER alcanza una precisión de <b>{precision:.1%}</b>. Como VADER fue diseñado principalmente para inglés,
        puede fallar con matices en español. Por eso, para la decisión de clima se muestran los resultados reales del dataset
        y también la predicción automática para evaluar qué tan confiable es el modelo.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Distribución real vs predicción VADER")
        labels = ["negativo", "neutro", "positivo"]
        real_vals = [df2["sentimiento_real"].value_counts().get(l, 0) for l in labels]
        pred_vals = [df2["sentimiento_predicho_vader"].value_counts().get(l, 0) for l in labels]

        fig1, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(len(labels))
        width = 0.35
        ax1.bar(x - width/2, real_vals, width, label="Real")
        ax1.bar(x + width/2, pred_vals, width, label="Predicho VADER")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("Número de comentarios")
        ax1.set_title("Distribución de sentimiento: real vs VADER")
        ax1.legend()
        for container in ax1.containers:
            ax1.bar_label(container, fontsize=9)
        st.pyplot(fig1)

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("% positivo por departamento")
            datos_pos = resumen_depto_real.sort_values("pct_positivo", ascending=True)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.barh(datos_pos["departamento"], datos_pos["pct_positivo"])
            ax2.axvline(pct_pos, linestyle="--", alpha=0.6, label=f"Media global {pct_pos:.1f}%")
            ax2.set_xlabel("% positivo")
            ax2.set_title("Departamentos con mayor clima positivo")
            ax2.legend()
            for i, v in enumerate(datos_pos["pct_positivo"]):
                ax2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
            st.pyplot(fig2)

        with col_g2:
            st.subheader("% negativo por departamento")
            datos_neg = resumen_depto_real.sort_values("pct_negativo", ascending=True)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.barh(datos_neg["departamento"], datos_neg["pct_negativo"])
            ax3.axvline(pct_neg, linestyle="--", alpha=0.6, label=f"Media global {pct_neg:.1f}%")
            ax3.set_xlabel("% negativo")
            ax3.set_title("Departamentos con mayor foco de atención")
            ax3.legend()
            for i, v in enumerate(datos_neg["pct_negativo"]):
                ax3.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
            st.pyplot(fig3)

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            st.subheader("Sentimiento por nivel")
            st.dataframe(
                resumen_nivel_real[["nivel", "negativo", "neutro", "positivo", "pct_negativo", "pct_positivo"]]
                .sort_values("pct_negativo", ascending=False),
                use_container_width=True
            )
        with col_n2:
            st.subheader("Sentimiento por departamento")
            st.dataframe(
                resumen_depto_real[["departamento", "negativo", "neutro", "positivo", "pct_negativo", "pct_positivo"]]
                .sort_values("pct_negativo", ascending=False),
                use_container_width=True
            )

        st.subheader("Matriz de confusión — VADER")
        cm = confusion_matrix(df2["sentimiento_real"], df2["sentimiento_predicho_vader"], labels=labels)
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        im = ax4.imshow(cm)
        ax4.set_xticks(np.arange(len(labels)))
        ax4.set_yticks(np.arange(len(labels)))
        ax4.set_xticklabels(labels)
        ax4.set_yticklabels(labels)
        ax4.set_xlabel("Predicho VADER")
        ax4.set_ylabel("Real")
        ax4.set_title("Matriz de confusión")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax4.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
        fig4.colorbar(im, ax=ax4)
        st.pyplot(fig4)

        st.subheader("Acciones sugeridas")
        st.write(f"1. Replicar buenas prácticas del departamento **{depto_mas_positivo['departamento']}**, donde se observa mayor sentimiento positivo.")
        st.write(f"2. Priorizar sesiones de escucha en **{depto_mas_negativo['departamento']}**, por ser el departamento con mayor porcentaje negativo.")
        st.write(f"3. Analizar comentarios del nivel **{nivel_mas_negativo['nivel']}** para entender si el problema se relaciona con liderazgo, carga laboral, comunicación o desarrollo.")

        with st.expander("Ver comentarios clasificados"):
            st.dataframe(df2[[
                "id_respuesta", "departamento", "nivel", "comentario",
                "sentimiento_real", "vader_score", "sentimiento_predicho_vader"
            ]], use_container_width=True)

        st.download_button(
            label="Descargar resultados de sentimiento",
            data=df2.to_csv(index=False).encode("utf-8"),
            file_name="resultados_sentimiento_vader.csv",
            mime="text/csv"
        )

    else:
        st.info("Sube el archivo CSV de la Tarea 2 para iniciar el análisis.")

# =============================================================
# TAREA 3 — ROTACIÓN MENSUAL
# =============================================================

with tab3:
    st.header("Tarea 3 — Cálculo Automatizado de Tasa de Rotación Mensual")
    st.write("Calcula la tasa de rotación mensual y detecta áreas o meses con mayor salida de empleados.")

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
        rotacion_departamento["tasa_rotacion_departamento"] = (
            rotacion_departamento["bajas"] / rotacion_departamento["total_empleados"] * 100
        ).round(2)
        rotacion_departamento = rotacion_departamento.sort_values("tasa_rotacion_departamento", ascending=False)

        mes_mayor = bajas_mensuales.sort_values("tasa_rotacion", ascending=False).iloc[0]
        depto_mayor = rotacion_departamento.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Empleados analizados", total_empleados)
        c2.metric("Bajas registradas", total_bajas)
        c3.metric("Rotación global", f"{tasa_global:.1f}%")
        c4.metric("Departamento crítico", depto_mayor["departamento"], f"{depto_mayor['tasa_rotacion_departamento']}%")

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
            st.dataframe(bajas_mensuales.sort_values("tasa_rotacion", ascending=False), use_container_width=True)
        with col_t2:
            st.dataframe(rotacion_departamento, use_container_width=True)

        st.markdown(f"""
        <div class="insight-box">
        <b>Insight de rotación:</b><br>
        El mes con mayor rotación fue <b>{mes_mayor['mes_salida']}</b> con <b>{mes_mayor['tasa_rotacion']}%</b>.
        El departamento más crítico es <b>{depto_mayor['departamento']}</b>, con una tasa de <b>{depto_mayor['tasa_rotacion_departamento']}%</b>.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Acciones de retención sugeridas")
        st.write(f"1. Realizar entrevistas de salida estructuradas en **{depto_mayor['departamento']}** para entender causas recurrentes.")
        st.write("2. Revisar carga laboral, oportunidades de desarrollo y estilo de liderazgo en las áreas con mayor rotación.")
        st.write("3. Monitorear la tasa mensualmente para detectar picos tempranos y activar planes de retención antes de que el problema escale.")

        with st.expander("Ver dataset procesado"):
            st.dataframe(df3, use_container_width=True)

        st.download_button(
            label="Descargar tasa de rotación mensual",
            data=bajas_mensuales.to_csv(index=False).encode("utf-8"),
            file_name="tasa_rotacion_mensual.csv",
            mime="text/csv"
        )

    else:
        st.info("Sube el archivo CSV de la Tarea 3 para iniciar el análisis.")

