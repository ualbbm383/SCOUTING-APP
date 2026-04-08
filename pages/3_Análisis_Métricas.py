import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_player_metrics
from src.auth import check_login

if not check_login():
    st.stop()

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


st.set_page_config(page_title="Rankings y comparación", layout="wide")

st.title("Análisis Exploratorio de Métricas")
st.markdown(
    """
    <div class="top-note">
        <div class="small-muted">
            Explora <b>rankings</b> de cualquier métrica, diseña <b>scatter plots</b> relacionando varias métricas
            (recomendado: volumen y % éxito) y realiza <b>comparaciones de radar</b> entre jugadores o con la media de su posición y perfil.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
) 

# --------------------------------------------------
# CONFIG LABELS
# --------------------------------------------------
PLAYER_METRIC_LABELS = {
    "minutes_total": "Minutos",
    "passes_90": "Pases/90",
    "successful_passes_90": "Pases exitosos/90",
    "short_passes_90": "Pases cortos/90",
    "medium_passes_90": "Pases medios/90",
    "long_passes_90": "Pases largos/90",
    "forward_passes_90":"Pases hacia delante/90",
    "backward_passes_90": "Pases hacia detrás/90",
    "lateral_passes_90": "Pases laterales/90",
    "passes_final_third_90": "Pases último tercio/90",
    "passes_final_third_pct": "% pases último tercio",
    "pass_accuracy": "% éxito pase",
    "short_pass_accuracy": "% éxito pase corto",
    "medium_pass_accuracy": "% éxito pase medio",
    "long_pass_accuracy": "% éxito pase largo",
    "forward_pass_accuracy": "% éxito pase hacia delante",
    "backward_pass_accuracy": "% éxito pase hacia atrás",
    "lateral_pass_accuracy": "% éxito pase lateral",
    "progressive_passes_90": "Pases progresivos/90",
    "progressive_passes_final_third_90": "Pases prog. últ. tercio/90",
    "progressive_pass_pct": "% pases progresivos",
    "key_passes_90": "Pases clave/90",
    "crosses_90": "Centros/90",
    "cross_accuracy": "% éxito centros",
    "shots_90": "Tiros/90",
    "goals_90": "Goles/90",
    "shots_on_target_90": "Tiros a puerta/90",
    "big_chances_90": "Ocasiones claras/90",
    "big_chances_created_90": "Ocasiones claras creadas/90",
    "shot_accuracy": "% éxito en tiro (A puerta/tiro)",
    "goal_conversion": "Ratio de conversión (Gol/tiro)",
    "inside_box_shot_pct": "% tiros dentro del área",
    "shots_inside_box_90": "Tiros dentro del área/90",
    "shots_outside_box_90": "Tiros fuera del área/90",
    "avg_shot_distance_m": "Distancia media de tiro",
    "recoveries_90": "Recuperaciones/90",
    "recoveries_def_third_90": "Recuperaciones tercio defensivo/90",
    "recoveries_mid_third_90": "Recuperaciones tercio medio/90",
    "recoveries_final_third_90": "Recuperaciones tercio final/90",
    "tackles_90": "Entradas/90",
    "successful_tackles_90": "Entradas exitosas/90",
    "tackle_success_pct": "% éxito entradas",
    "tackles_def_third_90": "Entradas primer tercio/90",
    "tackles_mid_third_90": "Entradas segundo tercio/90",
    "tackles_final_third_90": "Entradas últ. tercio/90",
    "interceptions_90": "Intercepciones/90",
    "clearances_90": "Despejes/90",
    "aerials_90": "Duelos aéreos/90",
    "successful_aerials_90": "Duelos aéreos exitosos/90",
    "aerial_win_pct": "% éxito aéreo",
    "blocked_passes_90": "Pases bloqueados/90",
    "fouls_90": "Faltas/90",
    "takeons_90": "Regates/90",
    "successful_takeons_90": "Regates exitosos/90",
    "takeon_success_pct": "% éxito regate",
    "dispossessed_90": "Pérdidas/90",
    "long_pass_pct": "% pases largos",
    "avg_pass_length": "Distancia media de pase",
    "avg_progressive_distance_m": "Distancia progresiva media",
    "cluster_name": "Perfil detectado",
    "position_group": "Posición",
    "age": "Edad",
    "market_value": "Valor de mercado",
}

DEFAULT_METRICS = [
    "passes_90",
    "progressive_passes_90",
    "key_passes_90",
    "crosses_90",
    "shots_90",
    "goals_90",
    "recoveries_90",
    "tackles_90",
    "interceptions_90",
    "aerials_90",
    "aerial_win_pct",
    "takeons_90",
    "takeon_success_pct",
    "big_chances_created_90",
    "inside_box_shot_pct",
]

POSITION_RADAR_METRICS = {
    "Midfielder": [
        "passes_90",
        "progressive_passes_90",
        "key_passes_90",
        "recoveries_90",
        "tackles_90",
        "shots_90",
    ],
    "Center Back": [
        "passes_90",
        "long_pass_pct",
        "interceptions_90",
        "clearances_90",
        "aerials_90",
        "aerial_win_pct",
    ],
    "Striker": [
        "shots_90",
        "goals_90",
        "inside_box_shot_pct",
        "aerials_90",
        "key_passes_90",
        "big_chances_created_90",
    ],
    "Winger": [
        "shots_90",
        "takeons_90",
        "takeon_success_pct",
        "key_passes_90",
        "crosses_90",
        "big_chances_created_90",
    ],
    "Full Back": [
        "progressive_pass_pct",
        "crosses_90",
        "key_passes_90",
        "takeons_90",
        "tackles_90",
        "interceptions_90",
    ],
}


POSITION_LABELS = {
    "Goalkeeper": "Portero",
    "Center Back": "Central",
    "Full Back": "Lateral",
    "Midfielder": "Centrocampista",
    "Winger": "Extremo",
    "Striker": "Delantero",
}

def format_position_name(position: str) -> str:
    return POSITION_LABELS.get(position, position)



def format_metric(metric: str) -> str:
    return PLAYER_METRIC_LABELS.get(metric, metric)

def build_player_label(row: pd.Series) -> str:
    return f"{row['player_name']} | {row['team_name']} | {row['season']}"

def get_numeric_metric_cols(df: pd.DataFrame) -> list[str]:
    excluded = {
        "Player ID",
        "cluster",
        "umap_x",
        "umap_y",
        "gmm_profile_1_pct",
        "gmm_profile_2_pct",
        "gmm_profile_3_pct",
        "profile_1_pct",
        "profile_2_pct",
        "profile_3_pct",
    }
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100

# --------------------------------------------------
# CARGA DATOS
# --------------------------------------------------
df_all = load_player_metrics().copy()

# etiqueta única
df_all["player_label"] = df_all.apply(build_player_label, axis=1)

numeric_metric_cols = get_numeric_metric_cols(df_all)
metric_options = [m for m in numeric_metric_cols if m in df_all.columns]

# --------------------------------------------------
# FILTROS
# --------------------------------------------------
st.sidebar.header("Filtros")

leagues = ["Todas"] + sorted(df_all["league"].dropna().astype(str).unique().tolist())
selected_league = st.sidebar.selectbox("Liga", leagues)

seasons = ["Todas"] + sorted(df_all["season"].dropna().astype(str).unique().tolist())
selected_season = st.sidebar.selectbox("Temporada", seasons)

position_values = sorted(df_all["position_group"].dropna().astype(str).unique().tolist())
position_options = ["Todas"] + position_values

selected_position_label = st.sidebar.selectbox(
    "Posición",
    position_options,
    format_func=lambda x: "Todas" if x == "Todas" else format_position_name(x)
)

profiles = ["Todos"] + sorted(df_all["cluster_name"].dropna().astype(str).unique().tolist())
selected_profile = st.sidebar.selectbox("Perfil", profiles)

min_minutes = int(df_all["minutes_total"].fillna(0).min())
max_minutes = int(df_all["minutes_total"].fillna(0).max())
selected_minutes = st.sidebar.slider(
    "Minutos mínimos",
    min_value=min_minutes,
    max_value=max_minutes,
    value=min(600, max_minutes),
    step=50
)

df_view = df_all.copy()

if selected_league != "Todas":
    df_view = df_view[df_view["league"].astype(str) == selected_league].copy()

if selected_season != "Todas":
    df_view = df_view[df_view["season"].astype(str) == selected_season].copy()

if selected_position_label != "Todas":
    df_view = df_view[df_view["position_group"].astype(str) == selected_position_label].copy()

if selected_profile != "Todos":
    df_view = df_view[df_view["cluster_name"].astype(str) == selected_profile].copy()

df_view = df_view[df_view["minutes_total"].fillna(0) >= selected_minutes].copy()
df_view = df_view.reset_index(drop=True)

st.caption(f"Jugadores visualizados: {len(df_view)}")

if df_view.empty:
    st.warning("No hay jugadores con esos filtros.")
    st.stop()

#st.markdown("---")

# --------------------------------------------------
# RANKING DINÁMICO
# --------------------------------------------------
st.header("Ranking dinámico")
st.caption("Selecciona una métrica y ordena la muestra para identificar los jugadores más destacados.")

rank_col1, rank_col2 = st.columns([1, 2])

with rank_col1:
    ranking_metric = st.selectbox(
        "Métrica del ranking",
        metric_options,
        index=metric_options.index("progressive_passes_90") if "progressive_passes_90" in metric_options else 0,
        format_func=format_metric,
        key="ranking_metric"
    )

    ranking_order = st.radio(
        "Orden",
        ["Mayor a menor", "Menor a mayor"],
        horizontal=True
    )

    top_n = st.slider("Número de jugadores", 5, 50, 20, 5)

ascending_rank = ranking_order == "Menor a mayor"

ranking_df = (
    df_view[
        ["player_name", "team_name", "league", "season", "minutes_total", "position_group", "cluster_name", ranking_metric]
    ]
    .dropna(subset=[ranking_metric])
    .sort_values(ranking_metric, ascending=ascending_rank)
    .head(top_n)
    .reset_index(drop=True)
)

ranking_df["position_group"] = ranking_df["position_group"].map(format_position_name)
ranking_df["ranking"] = ranking_df.index + 1

# etiqueta única para el eje Y
ranking_df["player_axis"] = (
    ranking_df["player_name"].astype(str)
    + " | "
    + ranking_df["team_name"].astype(str)
)

# para gráfico horizontal:
# si el ranking es de mayor a menor, invertimos para que el mejor salga arriba
if ascending_rank:
    ranking_plot_df = ranking_df.copy()
else:
    ranking_plot_df = ranking_df.iloc[::-1].copy()

with rank_col2:
    fig_rank = px.bar(
        ranking_plot_df,
        x=ranking_metric,
        y="player_axis",
        color="cluster_name" if "cluster_name" in ranking_plot_df.columns else None,
        hover_data=["player_name", "team_name", "league", "season", "minutes_total", "position_group"],
        orientation="h",
        title=f"Top {top_n} en {format_metric(ranking_metric)}"
    )

    fig_rank.update_layout(
        height=650,
        yaxis_title="",
        xaxis_title=format_metric(ranking_metric),
        legend_title_text="Perfil"
    )

    # fijar orden exacto del ranking global
    fig_rank.update_yaxes(
        categoryorder="array",
        categoryarray=ranking_plot_df["player_axis"].tolist()
    )

    st.plotly_chart(fig_rank, use_container_width=True)

st.dataframe(
    ranking_df.rename(columns={
        "player_name": "Jugador",
        "team_name": "Equipo",
        "league": "Liga",
        "season": "Temporada",
        "minutes_total": "Minutos",
        "position_group": "Posición",
        "cluster_name": "Perfil",
        ranking_metric: format_metric(ranking_metric),
        "ranking": "Ranking",
    }).drop(columns=["player_axis"], errors="ignore"),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# SCATTER LIBRE
# --------------------------------------------------
st.markdown("---")
st.header("Scatter libre")
st.caption("Especialmente útil para combinar métricas de volumen con métricas de eficiencia.")

scatter_controls_1, scatter_controls_2 = st.columns(2)

with scatter_controls_1:
    scatter_x = st.selectbox(
        "Eje X",
        metric_options,
        index=metric_options.index("takeons_90") if "takeons_90" in metric_options else 0,
        format_func=format_metric,
        key="scatter_x"
    )

with scatter_controls_2:
    scatter_y = st.selectbox(
        "Eje Y",
        metric_options,
        index=metric_options.index("takeon_success_pct") if "takeon_success_pct" in metric_options else min(1, len(metric_options)-1),
        format_func=format_metric,
        key="scatter_y"
    )

scatter_size_option = st.selectbox(
    "Tamaño de burbuja",
    ["Ninguno", "minutes_total", "market_value", "age"],
    format_func=lambda x: "Sin tamaño" if x == "Ninguno" else format_metric(x)
)

scatter_df = df_view.dropna(subset=[scatter_x, scatter_y]).copy()
scatter_df["position_label"] = scatter_df["position_group"].map(format_position_name)

fig_scatter = px.scatter(
    scatter_df,
    x=scatter_x,
    y=scatter_y,
    color="cluster_name" if "cluster_name" in scatter_df.columns else None,
    size=None if scatter_size_option == "Ninguno" else scatter_size_option,
    hover_name="player_name",
    hover_data=[
        "team_name",
        "league",
        "season",
        "minutes_total",
        "position_label",
        "cluster_name"
    ],
    title=f"{format_metric(scatter_x)} vs {format_metric(scatter_y)}"
)

fig_scatter.update_traces(marker=dict(opacity=0.8))
fig_scatter.update_layout(height=650)
st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------
# RADAR DE COMPARACIÓN
# --------------------------------------------------
st.markdown("---")
st.header("Radar de comparación")

radar_col1, radar_col2 = st.columns([1, 2])

player_options = sorted(df_view["player_label"].dropna().unique().tolist())

with radar_col1:
    selected_players = st.multiselect(
        "Selecciona jugadores",
        player_options,
        default=player_options[:2] if len(player_options) >= 2 else player_options,
        max_selections=4
    )

    if not selected_players:
        st.info("Selecciona al menos un jugador.")
        st.stop()

    radar_reference = st.radio(
        "Comparar también con",
        ["Nadie", "Media de la posición", "Media del perfil"],
        horizontal=False
    )

    first_player_row = df_view[df_view["player_label"] == selected_players[0]].iloc[0]
    first_position = first_player_row.get("position_group")

    default_radar_metrics = POSITION_RADAR_METRICS.get(first_position, DEFAULT_METRICS)
    default_radar_metrics = [m for m in default_radar_metrics if m in metric_options]

    radar_metrics = st.multiselect(
        "Métricas del radar",
        metric_options,
        default=default_radar_metrics[:6],
        format_func=format_metric
    )

    radar_scale = st.radio(
        "Escala del radar",
        ["Percentil", "Min-Max por métrica"],
        horizontal=True
    )

    st.markdown(
    """
    <span style='color: #9ca3af; font-size: 0.9rem;'>
    <b>Escala percentil:</b> muestra en qué posición relativa se encuentra el jugador respecto al resto.<br>
    <b>Escala Min-Max:</b> refleja mejor la diferencia cuantitativa real entre jugadores en cada métrica.
    </span>
    """,
    unsafe_allow_html=True)

if not radar_metrics:
    st.warning("Selecciona al menos una métrica para el radar.")
    st.stop()

radar_players_df = df_view[df_view["player_label"].isin(selected_players)].copy()

# --------------------------------------------------
# REFERENCIAS
# --------------------------------------------------
reference_rows = []

if radar_reference == "Media de la posición":
    ref_position = first_player_row.get("position_group")
    ref_df = df_view[df_view["position_group"] == ref_position]
    if not ref_df.empty:
        reference_rows.append((
            f"Media {format_position_name(ref_position)}",
            ref_df[radar_metrics].mean()
        ))

elif radar_reference == "Media del perfil":
    ref_profile = first_player_row.get("cluster_name")
    ref_df = df_view[df_view["cluster_name"] == ref_profile]
    if not ref_df.empty:
        reference_rows.append((
            f"Media {ref_profile}",
            ref_df[radar_metrics].mean()
        ))

# --------------------------------------------------
# ESCALADO DEL RADAR
# --------------------------------------------------
if radar_scale == "Percentil":
    radar_base = df_view[radar_metrics].copy()
    radar_base = radar_base.rank(pct=True) * 100

    radar_plot_df = radar_players_df.copy()
    radar_plot_df.loc[:, radar_metrics] = radar_base.loc[radar_players_df.index, radar_metrics].values

    processed_reference_rows = []
    for ref_name, ref_series in reference_rows:
        ref_percentiles = []
        for metric in radar_metrics:
            ref_percentiles.append(
                (df_view[metric] <= ref_series[metric]).mean() * 100
                if pd.notna(ref_series[metric]) else np.nan
            )
        processed_reference_rows.append(
            (ref_name, pd.Series(ref_percentiles, index=radar_metrics))
        )
    reference_rows = processed_reference_rows

    radial_range = [0, 100]

else:  # Min-Max por métrica
    radar_base = df_view[radar_metrics].copy()

    mins = radar_base.min()
    maxs = radar_base.max()
    denom = (maxs - mins).replace(0, 1)

    radar_scaled = (radar_base - mins) / denom

    radar_plot_df = radar_players_df.copy()
    radar_plot_df.loc[:, radar_metrics] = radar_scaled.loc[radar_players_df.index, radar_metrics].values

    processed_reference_rows = []
    for ref_name, ref_series in reference_rows:
        ref_scaled = (ref_series[radar_metrics] - mins) / denom
        processed_reference_rows.append((ref_name, ref_scaled))
    reference_rows = processed_reference_rows

    radial_range = [0, 1]

# --------------------------------------------------
# COLORES
# --------------------------------------------------
player_colors = [
    "#1f77b4",  # azul
    "#d62728",  # rojo
    "#2ca02c",  # verde
    "#ff7f0e",  # naranja
]

reference_colors = [
    "#111111",  # negro
    "#7f7f7f",  # gris
]

# --------------------------------------------------
# RADAR
# --------------------------------------------------
with radar_col2:
    fig_radar = go.Figure()

    for i, (_, row) in enumerate(radar_plot_df.iterrows()):
        color = player_colors[i % len(player_colors)]

        fig_radar.add_trace(go.Scatterpolar(
            r=[row[m] if pd.notna(row[m]) else 0 for m in radar_metrics],
            theta=[format_metric(m) for m in radar_metrics],
            fill="toself",
            name=f"{row['player_name']} | {row['team_name']}",
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.25
        ))

    for j, (ref_name, ref_series) in enumerate(reference_rows):
        ref_color = reference_colors[j % len(reference_colors)]

        fig_radar.add_trace(go.Scatterpolar(
            r=[ref_series[m] if pd.notna(ref_series[m]) else 0 for m in radar_metrics],
            theta=[format_metric(m) for m in radar_metrics],
            fill="toself",
            name=ref_name,
            line=dict(color=ref_color, width=2, dash="dash"),
            fillcolor=ref_color,
            opacity=0.18
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=radial_range
            )
        ),
        showlegend=True,
        height=700,
        title="Comparación radar"
    )

    st.plotly_chart(fig_radar, use_container_width=True)

# --------------------------------------------------
# TABLA DE COMPARACIÓN
# --------------------------------------------------
st.subheader("Tabla de comparación")

comparison_cols = ["player_name", "team_name", "league", "season", "minutes_total", "position_group", "cluster_name"] + radar_metrics
comparison_cols = [c for c in comparison_cols if c in radar_players_df.columns]

comparison_df = radar_players_df[comparison_cols].copy()
comparison_df["position_group"] = comparison_df["position_group"].map(format_position_name)

comparison_df = comparison_df.rename(columns={
    "player_name": "Jugador",
    "team_name": "Equipo",
    "league": "Liga",
    "season": "Temporada",
    "minutes_total": "Minutos",
    "position_group": "Posición",
    "cluster_name": "Perfil",
    **{m: format_metric(m) for m in radar_metrics}
})

st.dataframe(comparison_df, use_container_width=True, hide_index=True)