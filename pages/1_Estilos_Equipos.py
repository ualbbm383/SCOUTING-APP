from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.data_loader import load_team_metrics
from src.team_logos import show_team_logo
from src.auth import check_login

if not check_login():
    st.stop()


TEAM_METRIC_LABELS = {
    "matches_played": "Partidos jugados",

    "passes_match": "Pases por partido",
    "passes_final_third_match": "Pases en último tercio por partido",
    "successful_passes_match": "Pases completados por partido",
    "short_passes_match": "Pases cortos por partido",
    "medium_passes_match": "Pases medios por partido",
    "long_passes_match": "Pases largos por partido",
    "forward_passes_match": "Pases hacia delante por partido",
    "backward_passes_match": "Pases hacia atrás por partido",
    "lateral_passes_match": "Pases laterales por partido",
    "crosses_match": "Centros por partido",
    "key_passes_match": "Pases clave por partido",
    "progressive_passes_match": "Pases progresivos por partido",
    "progressive_passes_final_third_match": "Pases progresivos al último tercio por partido",

    "pass_accuracy": "Precisión de pase",
    "short_pass_accuracy": "Precisión pase corto",
    "medium_pass_accuracy": "Precisión pase medio",
    "long_pass_accuracy": "Precisión pase largo",
    "forward_pass_accuracy": "Precisión pase hacia delante",
    "backward_pass_accuracy": "Precisión pase hacia atrás",
    "lateral_pass_accuracy": "Precisión pase lateral",
    "cross_accuracy": "Precisión de centro",

    "long_pass_pct": "% de pases largos",
    "progressive_pass_pct": "% de pases progresivos",
    "passes_final_third_pct": "% de pases en último tercio",
    "avg_pass_length": "Distancia media de pase",
    "avg_progressive_distance_m": "Distancia progresiva media",

    "recoveries_match": "Recuperaciones por partido",
    "tackles_match": "Entradas por partido",
    "successful_tackles_match": "Entradas exitosas por partido",
    "interceptions_match": "Intercepciones por partido",
    "clearances_match": "Despejes por partido",
    "aerials_match": "Duelos aéreos por partido",
    "successful_aerials_match": "Duelos aéreos ganados por partido",
    "blocked_passes_match": "Pases bloqueados por partido",
    "fouls_match": "Faltas por partido",

    "recoveries_def_third_match": "Recuperaciones en tercio defensivo",
    "recoveries_mid_third_match": "Recuperaciones en tercio medio",
    "recoveries_final_third_match": "Recuperaciones en tercio final",
    "tackles_def_third_match": "Entradas en tercio defensivo",
    "tackles_mid_third_match": "Entradas en tercio medio",
    "tackles_final_third_match": "Entradas en tercio final",

    "tackle_success_pct": "% de éxito en entradas",
    "aerial_win_pct": "% de duelos aéreos ganados",

    "recoveries_def_third_pct": "% recuperaciones en tercio defensivo",
    "recoveries_mid_third_pct": "% recuperaciones en tercio medio",
    "recoveries_final_third_pct": "% recuperaciones en tercio final",
    "tackles_def_third_pct": "% entradas en tercio defensivo",
    "tackles_mid_third_pct": "% entradas en tercio medio",
    "tackles_final_third_pct": "% entradas en tercio final",

    "shots_match": "Tiros por partido",
    "goals_match": "Goles por partido",
    "shots_on_target_match": "Tiros a puerta por partido",
    "big_chances_match": "Grandes ocasiones por partido",
    "shots_inside_box_match": "Tiros dentro del área por partido",
    "shots_outside_box_match": "Tiros fuera del área por partido",
    "shot_accuracy": "Precisión de tiro",
    "goal_conversion": "Conversión de gol",
    "inside_box_shot_pct": "% tiros dentro del área",
    "avg_shot_distance_m": "Distancia media de tiro",

    "takeons_match": "Regates por partido",
    "successful_takeons_match": "Regates exitosos por partido",
    "takeon_success_pct": "% de éxito en regates",
    "dispossessed_match": "Pérdidas por partido",
    "big_chances_created_match": "Grandes ocasiones creadas por partido",

    "ppda": "PPDA",
}


def format_team_metric_name(metric: str) -> str:
    return TEAM_METRIC_LABELS.get(metric, metric)


def build_style_mix_from_kmeans_distances(distances: np.ndarray, alpha: float = 2.5) -> np.ndarray:
    similarity = 1 / (1 + distances)
    similarity = similarity ** alpha
    style_mix = similarity / similarity.sum(axis=1, keepdims=True)
    return style_mix


def get_report_metric_cols(df: pd.DataFrame, extra_excluded: set[str] | None = None) -> list[str]:
    excluded = {
        "Team ID",
        "team_name",
        "league",
        "season",
        "matches_played",
        "Cluster",
        "ClusterName",
        "PCA1",
        "PCA2",
        "style_possession",
        "style_vertical",
        "style_direct",
        "style_high_press",
        "style_mid_block",
        "style_low_block",
    }

    if extra_excluded:
        excluded |= extra_excluded

    metric_cols = [
        c for c in df.columns
        if (
            c not in excluded
            and pd.api.types.is_numeric_dtype(df[c])
            and "accuracy" not in c.lower()
        )
    ]
    return metric_cols


def style_description_mcb(style_name: str) -> str:
    descriptions = {
        "Directo": "Mayor peso del juego largo, la disputa aérea y una progresión menos apoyada en la circulación.",
        "Llegadores / Vertical": "Más presencia en último tercio, centros, desborde y ataque con intención de acelerar y finalizar.",
        "Posesión": "Más control del balón, circulación más paciente y progresión apoyada en el pase.",
    }
    return descriptions.get(style_name, "")


st.set_page_config(page_title="Identificación de Estilos de Juego", layout="wide")

# --------------------------------------------------
# ESTILO GENERAL
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .top-note {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .style-card {
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        height: 100%;
    }

    .mini-card {
        padding: 0.8rem 0.9rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        height: 100%;
    }

    .team-head-card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .small-muted {
        color: #475569;
        font-size: 0.93rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Identificación de estilos de juego")
st.caption("Análisis del comportamiento colectivo de los equipos en momento con balón (MCB) y momento sin balón (MSB).")

st.markdown(
    """
    <div class="top-note">
        <div class="small-muted">
            <b>MCB:</b> clasificación de estilos de juego a partir de métricas de posesión, verticalidad y juego directo mediante KMeans.<br>  
            <b>MSB:</b> análisis de la intensidad de presión a través del indicador PPDA.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# CARGA DE DATOS
# --------------------------------------------------
df_all = load_team_metrics().copy()
df_all = df_all[df_all["matches_played"] >= 10].copy().reset_index(drop=True)

st.divider()

# --------------------------------------------------
# SELECTOR DE MOMENTO DEL JUEGO
# --------------------------------------------------
mode = st.radio(
    "Selecciona el momento del juego",
    ["MCB - Momento con balón", "MSB - Momento sin balón"],
    horizontal=True
)

st.divider()

# ==================================================
# MCB - MOMENTO CON BALÓN
# ==================================================
def relabel_team_clusters(cluster_means: pd.DataFrame) -> dict:
    scores = {}

    for cluster_id, row in cluster_means.iterrows():

        possession_score = (
            row["takeons_match"]
            - row["long_pass_pct"]
            - row["avg_pass_length"]
        )

        direct_score = (
            row["long_pass_pct"]
            + row["avg_pass_length"]
            + row["aerials_match"]
        )

        vertical_score = (
            row["passes_final_third_pct"]
            + row["crosses_match"]
            + row["progressive_pass_pct"]
        )

        scores[cluster_id] = {
            "Posesión": possession_score,
            "Directo": direct_score,
            "Llegadores / Vertical": vertical_score,
        }

    assigned = {}
    used = set()

    # Directo
    c_direct = max(scores, key=lambda c: scores[c]["Directo"])
    assigned[c_direct] = "Directo"
    used.add(c_direct)

    # Posesión
    remaining = [c for c in scores if c not in used]
    c_pos = max(remaining, key=lambda c: scores[c]["Posesión"])
    assigned[c_pos] = "Posesión"
    used.add(c_pos)

    # Vertical
    remaining = [c for c in scores if c not in used]
    assigned[remaining[0]] = "Llegadores / Vertical"

    return assigned

if mode == "MCB - Momento con balón":

    style_features = [
        "progressive_pass_pct",
        "long_pass_pct",
        "passes_final_third_pct",
        "avg_pass_length",
        "avg_progressive_distance_m",
        "crosses_match",
        "takeons_match",
        "aerials_match"
    ]

  
    cluster_order = [
        "Directo",
        "Llegadores / Vertical",
        "Posesión"
    ]

    df_model = df_all.copy()

    X_model = df_model[style_features].copy()
    X_model = X_model.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_model)

    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_scaled)

    df_model["PCA1"] = X_2d[:, 0]
    df_model["PCA2"] = X_2d[:, 1]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    k_clusters = kmeans.fit_predict(X_scaled)

    df_model["Cluster"] = k_clusters

    cluster_profile = pd.DataFrame(X_scaled, columns=X_model.columns)
    cluster_profile["Cluster"] = df_model["Cluster"].values

    cluster_means = cluster_profile.groupby("Cluster").mean()
    global_mean = cluster_profile[X_model.columns].mean()
    diff = (cluster_means - global_mean).round(2)

    cluster_names = relabel_team_clusters(cluster_means)
    df_model["ClusterName"] = df_model["Cluster"].map(cluster_names)

    distances = kmeans.transform(X_scaled)
    style_mix = build_style_mix_from_kmeans_distances(distances, alpha=2.5)

    df_model["style_direct"] = 0.0
    df_model["style_vertical"] = 0.0
    df_model["style_possession"] = 0.0

    style_label_to_col = {
        "Directo": "style_direct",
        "Llegadores / Vertical": "style_vertical",
        "Posesión": "style_possession",
    }

    for cluster_id, style_label in cluster_names.items():
        col_name = style_label_to_col[style_label]
        df_model[col_name] = style_mix[:, cluster_id]

    st.sidebar.header("Filtros")

    leagues = sorted(df_model["league"].dropna().unique().tolist())
    league_options = ["Todas"] + leagues
    selected_league = st.sidebar.selectbox("Liga", league_options)

    seasons = sorted(df_model["season"].dropna().astype(str).unique().tolist())
    season_options = ["Todas"] + seasons
    selected_season = st.sidebar.selectbox("Temporada", season_options)

    df_view = df_model.copy()

    if selected_league != "Todas":
        df_view = df_view[df_view["league"] == selected_league].copy()

    if selected_season != "Todas":
        df_view = df_view[df_view["season"].astype(str) == selected_season].copy()

    df_view = df_view.reset_index(drop=True)

    st.caption(f"Equipos visualizados: {len(df_view)}")

    # --------------------------------------------------
    # VISTA GLOBAL
    # --------------------------------------------------
    st.header("Vista global")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Distribución por estilo")

        cluster_counts = (
            df_view["ClusterName"]
            .value_counts()
            .reindex(cluster_order)
            .dropna()
        )

        fig_bar = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={"x": "Estilo", "y": "Número de equipos"},
            title="Distribución"
        )
        fig_bar.update_layout(height=560, xaxis_tickangle=0)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Mapa PCA de equipos")

        fig_scatter = px.scatter(
            df_view,
            x="PCA1",
            y="PCA2",
            color="ClusterName",
            hover_name="team_name",
            hover_data=[
                "league",
                "season",
                "progressive_pass_pct",
                "long_pass_pct",
                "passes_final_third_pct",
                "avg_pass_length",
                "avg_progressive_distance_m",
                "crosses_match",
                "takeons_match",
                "aerials_match"
            ],
            category_orders={"ClusterName": cluster_order},
            title="Mapa PCA"
        )
        fig_scatter.update_traces(marker=dict(size=11, opacity=0.85))
        fig_scatter.update_layout(height=560)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --------------------------------------------------
    # RESUMEN DE CADA CLUSTER
    # --------------------------------------------------
    st.header("Resumen de estilos")

    summary_cols = st.columns(3)

    for i, cluster_label in enumerate(cluster_order):

        cluster_id = next(
            cid for cid, label in cluster_names.items() if label == cluster_label
        )

        cluster_diff = diff.loc[cluster_id].sort_values(ascending=False)

        with summary_cols[i]:
            st.subheader(cluster_label)
            st.caption(style_description_mcb(cluster_label))

            high_df = pd.DataFrame({
                "Más altas": [format_team_metric_name(m) for m in cluster_diff.head(5).index],
            })

            low_df = pd.DataFrame({
                "Más bajas": [format_team_metric_name(m) for m in cluster_diff.tail(5).sort_values().index],
            })

            st.markdown("**Variables más altas**")
            st.dataframe(high_df, use_container_width=True, hide_index=True)

            st.markdown("**Variables más bajas**")
            st.dataframe(low_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # INFORME INDIVIDUAL
    # --------------------------------------------------
    st.markdown("---")
    st.header("Informe individual de equipo")

    team_list = sorted(df_view["team_name"].dropna().unique().tolist())
    selected_team = st.selectbox("Selecciona un equipo", team_list)

    team_row = df_view[df_view["team_name"] == selected_team].iloc[0]
    team_cluster_name = team_row["ClusterName"]

    head_col1, head_col2, head_col3 = st.columns([1, 3.2, 2.2])

    with head_col1:
        show_team_logo(selected_team, width=150)
        st.markdown("</div>", unsafe_allow_html=True)

    with head_col2:
        st.subheader(selected_team)
        st.write(f"**Liga:** {team_row['league']}")
        st.write(f"**Temporada:** {team_row['season']}")
        st.write(f"**Estilo detectado:** {team_cluster_name}")
        st.caption(style_description_mcb(team_cluster_name))
        st.markdown("</div>", unsafe_allow_html=True)

    with head_col3:
        st.markdown("**Distribución híbrida del estilo**")

        pie_df = pd.DataFrame({
            "Estilo": ["Directo", "Llegadores / Vertical", "Posesión"],
            "Valor": [
                team_row["style_direct"],
                team_row["style_vertical"],
                team_row["style_possession"]
            ]
        }).sort_values("Valor", ascending=False)

        fig_pie = px.pie(
            pie_df,
            names="Estilo",
            values="Valor",
            title=None
        )
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # FORTALEZAS Y DEBILIDADES
    # --------------------------------------------------
    report_metric_cols = get_report_metric_cols(df_model)

    X_report = df_model[report_metric_cols].copy()
    X_report = X_report.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    percentiles = X_report.rank(pct=True) * 100
    percentiles.index = df_model["team_name"]

    team_percentiles = percentiles.loc[selected_team].sort_values(ascending=False)

    top_strengths = team_percentiles.head(5).round(1)
    top_weaknesses = team_percentiles.tail(5).sort_values().round(1)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Fortalezas del equipo")
        strengths_df = pd.DataFrame({
            "Métrica": [format_team_metric_name(m) for m in top_strengths.index],
            "Percentil": top_strengths.values
        })
        st.dataframe(strengths_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.subheader("Debilidades del equipo")
        weaknesses_df = pd.DataFrame({
            "Métrica": [format_team_metric_name(m) for m in top_weaknesses.index],
            "Percentil": top_weaknesses.values
        })
        st.dataframe(weaknesses_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # RESUMEN AUTOMÁTICO
    # --------------------------------------------------
    st.subheader("Resumen automático")

    strength_text = ", ".join([format_team_metric_name(m) for m in top_strengths.index[:3].tolist()])
    weakness_text = ", ".join([format_team_metric_name(m) for m in top_weaknesses.index[:3].tolist()])

    st.write(
        f"""
        **{selected_team}** ha sido clasificado dentro del estilo **{team_cluster_name}** en el momento con balón.
        Dentro de la muestra seleccionada, sus principales fortalezas relativas aparecen en
        **{strength_text}**, mientras que sus puntos menos destacados están en
        **{weakness_text}**.
        """
    )

# ==================================================
# MSB - MOMENTO SIN BALÓN
# ==================================================
if mode == "MSB - Momento sin balón":

    df_model = df_all.copy()

    st.sidebar.header("Filtros")

    leagues = sorted(df_model["league"].dropna().unique().tolist())
    league_options = ["Todas"] + leagues
    selected_league = st.sidebar.selectbox("Liga", league_options, key="league_def")

    seasons = sorted(df_model["season"].dropna().astype(str).unique().tolist())
    season_options = ["Todas"] + seasons
    selected_season = st.sidebar.selectbox("Temporada", season_options, key="season_def")

    df_view = df_model.copy()

    if selected_league != "Todas":
        df_view = df_view[df_view["league"] == selected_league].copy()

    if selected_season != "Todas":
        df_view = df_view[df_view["season"].astype(str) == selected_season].copy()

    df_view = df_view.dropna(subset=["ppda"]).reset_index(drop=True)

    st.header("Índice de presión (PPDA)")
    st.caption("PPDA: pases permitidos al rival por acción defensiva. Cuanto menor es el PPDA, mayor es la intensidad de presión.")

    ranking_df = (
        df_view[["team_name", "league", "season", "matches_played", "ppda"]]
        .sort_values("ppda", ascending=True)
        .reset_index(drop=True)
    )

    ranking_df["ranking_ppda"] = ranking_df.index + 1
    ranking_df["marker_size"] = 8

    # --------------------------------------------------
    # VISTA GLOBAL
    # --------------------------------------------------
    col1, col2 = st.columns([1.35, 1])

    with col1:
        st.subheader("Ranking de presión")

        st.dataframe(
            ranking_df[
                ["ranking_ppda", "team_name", "league", "season", "matches_played", "ppda"]
            ].rename(columns={
                "ranking_ppda": "Ranking",
                "team_name": "Equipo",
                "league": "Liga",
                "season": "Temporada",
                "matches_played": "Partidos",
                "ppda": "PPDA"
            }),
            use_container_width=True,
            hide_index=True,
            height=720
        )

    with col2:
        st.subheader("Distribución de PPDA")

        fig_hist = px.histogram(
            df_view,
            x="ppda",
            nbins=25,
            title="Distribución de PPDA"
        )
        fig_hist.update_layout(height=340)
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_box = px.box(
            df_view,
            y="ppda",
            points="all",
            hover_data=["team_name", "league", "season"],
            title="Distribución por equipo"
        )
        fig_box.update_layout(height=360)
        st.plotly_chart(fig_box, use_container_width=True)

    # --------------------------------------------------
    # INFORME INDIVIDUAL DE EQUIPO
    # --------------------------------------------------
    st.markdown("---")
    st.header("Informe individual de equipo")

    team_list = sorted(df_view["team_name"].dropna().unique().tolist())
    selected_team = st.selectbox("Selecciona un equipo", team_list, key="team_def")

    team_row = df_view[df_view["team_name"] == selected_team].iloc[0]

    team_rank = ranking_df.loc[
        ranking_df["team_name"] == selected_team, "ranking_ppda"
    ].iloc[0]

    n_teams = len(ranking_df)

    pressure_percentile = (
        ranking_df["ppda"].rank(pct=True, ascending=False)[
            ranking_df["team_name"] == selected_team
        ].iloc[0] * 100
    )

    ppda_value = team_row["ppda"]

    if ppda_value <= 5.5:
        pressure_label = "Presión muy alta"
    elif ppda_value <= 7.5:
        pressure_label = "Presión alta"
    elif ppda_value <= 9.5:
        pressure_label = "Presión media"
    elif ppda_value <= 11.5:
        pressure_label = "Presión baja"
    else:
        pressure_label = "Presión muy baja"

    head_col1, head_col2, head_col3 = st.columns([1, 3.2, 2.2])

    with head_col1:
        show_team_logo(selected_team, width=150)
        st.markdown("</div>", unsafe_allow_html=True)

    with head_col2:
        st.subheader(selected_team)
        st.write(f"**Liga:** {team_row['league']}")
        st.write(f"**Temporada:** {team_row['season']}")
        st.write(f"**Partidos:** {team_row['matches_played']}")
        st.write(f"**Interpretación del PPDA:** {pressure_label}")
        st.markdown("</div>", unsafe_allow_html=True)

    with head_col3:
        st.metric("PPDA", round(team_row["ppda"], 2))
        st.metric("Ranking de presión", f"{int(team_rank)} / {n_teams}")
        st.metric("Percentil de presión", f"{pressure_percentile:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # POSICIÓN EN LA MUESTRA
    # --------------------------------------------------
    st.subheader("Posición del equipo dentro de la muestra")

    rank_plot_df = ranking_df.copy()
    rank_plot_df["selected"] = rank_plot_df["team_name"] == selected_team
    rank_plot_df["marker_size"] = rank_plot_df["selected"].map({True: 16, False: 8})

    n_teams = len(rank_plot_df)
    rank_plot_df["ranking_ppda_inv"] = n_teams - rank_plot_df["ranking_ppda"] + 1

    fig_rank = px.scatter(
        rank_plot_df,
        x="ranking_ppda_inv",
        y="ppda",
        color="selected",
        size="marker_size",
        hover_data=["team_name", "league", "season"],
        title="Ranking de PPDA"
    )

    fig_rank.update_traces(opacity=0.8)
    fig_rank.update_layout(
        xaxis_title="Intensidad de presión (más a la derecha = más presión)",
        yaxis_title="PPDA",
        height=430,
        showlegend=False
    )

    fig_rank.update_xaxes(
        tickmode="array",
        tickvals=[1, n_teams],
        ticktext=["Menos presión", "Más presión"]
    )

    st.plotly_chart(fig_rank, use_container_width=True)

    # --------------------------------------------------
    # FORTALEZAS Y DEBILIDADES
    # --------------------------------------------------
    report_metric_cols = get_report_metric_cols(df_model)

    X_report = df_model[report_metric_cols].copy()
    X_report = X_report.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    percentiles = X_report.rank(pct=True) * 100
    percentiles.index = df_model["team_name"]

    team_percentiles = percentiles.loc[selected_team].sort_values(ascending=False)

    top_strengths = team_percentiles.head(5).round(1)
    top_weaknesses = team_percentiles.tail(5).sort_values().round(1)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Fortalezas del equipo")
        strengths_df = pd.DataFrame({
            "Métrica": [format_team_metric_name(m) for m in top_strengths.index],
            "Percentil": top_strengths.values
        })
        st.dataframe(strengths_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.subheader("Debilidades del equipo")
        weaknesses_df = pd.DataFrame({
            "Métrica": [format_team_metric_name(m) for m in top_weaknesses.index],
            "Percentil": top_weaknesses.values
        })
        st.dataframe(weaknesses_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # RESUMEN AUTOMÁTICO
    # --------------------------------------------------
    st.subheader("Resumen automático")

    strength_text = ", ".join([format_team_metric_name(m) for m in top_strengths.index[:3].tolist()])
    weakness_text = ", ".join([format_team_metric_name(m) for m in top_weaknesses.index[:3].tolist()])

    st.write(
        f"""
        **{selected_team}** registra un **PPDA de {team_row['ppda']:.2f}**, lo que lo sitúa en el puesto
        **{int(team_rank)} de {n_teams}** dentro de la muestra seleccionada en términos de intensidad de presión.
        Su interpretación general es de **{pressure_label.lower()}**.

        A nivel global, sus fortalezas relativas aparecen en **{strength_text}**,
        mientras que sus puntos menos destacados están en **{weakness_text}**.
        """
    )