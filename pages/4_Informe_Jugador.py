import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from mplsoccer import Pitch, VerticalPitch

from src.data_loader import load_player_metrics, query_events
from src.player_clustering import plot_profile_pie
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


POSITION_KEY_METRICS = {
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
        "progressive_pass_pct",
        "clearances_90",
        "interceptions_90",
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


PLAYER_METRIC_LABELS = {
    "minutes_total": "Minutos",
    "passes_90": "Pases/90",
    "passes_final_third_90": "Pases último tercio/90",
    "passes_final_third_pct": "% pases último tercio",
    "pass_accuracy": "Precisión de pase",
    "progressive_passes_90": "Pases progresivos/90",
    "progressive_passes_final_third_90": "Pases prog. últ. tercio/90",
    "progressive_pass_pct": "% pases progresivos",
    "key_passes_90": "Pases clave/90",
    "crosses_90": "Centros/90",
    "cross_accuracy": "Precisión de centro",
    "shots_90": "Tiros/90",
    "goals_90": "Goles/90",
    "shots_on_target_90": "Tiros a puerta/90",
    "big_chances_90": "Ocasiones claras/90",
    "big_chances_created_90": "Ocasiones claras creadas/90",
    "shot_accuracy": "Precisión de tiro",
    "goal_conversion": "Conversión de gol",
    "inside_box_shot_pct": "% tiros dentro del área",
    "avg_shot_distance_m": "Distancia media de tiro",
    "recoveries_90": "Recuperaciones/90",
    "recoveries_def_third_90": "Recuperaciones tercio defensivo/90",
    "recoveries_mid_third_90": "Recuperaciones tercio medio/90",
    "recoveries_final_third_90": "Recuperaciones tercio final/90",
    "tackles_90": "Entradas/90",
    "tackle_success_pct": "% éxito en entradas",
    "interceptions_90": "Intercepciones/90",
    "clearances_90": "Despejes/90",
    "aerials_90": "Duelos aéreos/90",
    "aerial_win_pct": "% éxito aéreo",
    "blocked_passes_90": "Pases bloqueados/90",
    "fouls_90": "Faltas/90",
    "takeons_90": "Regates/90",
    "successful_takeons_90": "Regates exitosos/90",
    "takeon_success_pct": "% éxito regate",
    "dispossessed_90": "Pérdidas/90",
    "long_pass_pct": "% pases largos",
}

def format_player_metric_name(metric: str) -> str:
    return PLAYER_METRIC_LABELS.get(metric, metric)


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


EVENT_TYPE_LABELS = {
    "Pass": "Pases",
    "Shot": "Tiros",
    "Defensive actions": "Acciones defensivas",
    "BallRecovery": "Recuperaciones",
    "Tackle": "Entradas",
    "Interception": "Intercepciones",
    "Clearance": "Despejes",
    "BlockedPass": "Pases bloqueados",
    "Aerial": "Duelos aéreos",
    "TakeOn": "Regates",
}

def format_event_type_name(event_type: str) -> str:
    return EVENT_TYPE_LABELS.get(event_type, event_type)


PASS_VIEW_LABELS = {
    "All passes": "Todos los pases",
    "Progressive passes": "Pases progresivos",
    "Final-third passes": "Pases en último tercio",
    "Passes into final third": "Pases hacia último tercio",
    "Key passes": "Pases clave",
    "Crosses": "Centros",
    "Long passes": "Pases largos",
    "Forward passes": "Pases hacia delante",
    "Backward passes": "Pases hacia atrás",
    "Lateral passes": "Pases laterales",
}

def format_pass_view_name(value: str) -> str:
    return PASS_VIEW_LABELS.get(value, value)


SHOT_VIEW_LABELS = {
    "All shots": "Todos los tiros",
    "Goals": "Goles",
    "Shots on target": "Tiros a puerta",
    "Missed shots": "Tiros fallados",
    "Shot on post": "Tiros al palo",
}

def format_shot_view_name(value: str) -> str:
    return SHOT_VIEW_LABELS.get(value, value)


def get_profile_names_from_dataset(
    df_players: pd.DataFrame,
    position_group: str
) -> dict:
    """
    Reconstruye el mapping real cluster_id -> cluster_name
    usando los datos ya guardados en player_metrics_enriched.
    """
    df_pos = df_players[
        (df_players["position_group"] == position_group)
        & (df_players["cluster"].notna())
        & (df_players["cluster_name"].notna())
    ].copy()

    if df_pos.empty:
        return {}

    df_pos["cluster"] = df_pos["cluster"].astype(int)

    profile_names = {}

    for cluster_id in sorted(df_pos["cluster"].unique()):
        cluster_rows = df_pos[df_pos["cluster"] == cluster_id]

        if cluster_rows.empty:
            continue

        most_common_name = cluster_rows["cluster_name"].mode()
        if not most_common_name.empty:
            profile_names[cluster_id] = most_common_name.iloc[0]

    return profile_names


st.set_page_config(page_title="Informe del Jugador", layout="wide")

st.title("Informe del Jugador")
st.markdown(
    """
    <div class="top-note">
        <div class="small-muted">
            Creación de un perfil personal del jugador a partir de las métricas más importantes para su rol.
            Además, se puede visualizar la zona del campo de cada evento en cada partido para dar un mayor contexto.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# CARGA DE DATOS
# --------------------------------------------------
player_metrics = load_player_metrics().copy()

# --------------------------------------------------
# FILTRO DE JUGADOR
# --------------------------------------------------
st.subheader("Jugador")

df_players = player_metrics.copy()

df_players["player_label"] = (
    df_players["player_name"].astype(str)
    + " | "
    + df_players["team_name"].astype(str)
)

player_options = sorted(df_players["player_label"].dropna().unique().tolist())

selected_player = st.selectbox(
    "Selecciona jugador",
    player_options,
    key="player_selector"
)

if not selected_player:
    st.warning("Selecciona un jugador.")
    st.stop()

player_row = df_players[df_players["player_label"] == selected_player].iloc[0]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.subheader(f"{player_row['player_name']} — {player_row['team_name']}")
st.caption(
    f"{player_row['league']} | Temporada {player_row['season']} | "
    f"{player_row['minutes_total']} minutos"
)

# --------------------------------------------------
# RESUMEN RÁPIDO + ESTILO DE JUGADOR
# --------------------------------------------------
st.subheader("Resumen rápido")

left_col, right_col = st.columns([1.35, 1])

with left_col:
    st.markdown("**Información general**")

    info1, info2, info3 = st.columns(3)
    info1.metric("Posición", format_position_name(player_row.get("position_group", "-")))
    info2.metric("Estilo detectado", player_row.get("cluster_name", "-"))
    info3.metric("Minutos", int(player_row.get("minutes_total", 0)))

    st.markdown("**Métricas clave**")

    position_group = player_row.get("position_group")
    key_metrics = POSITION_KEY_METRICS.get(position_group, [
        "passes_90",
        "progressive_passes_90",
        "key_passes_90",
        "shots_90",
        "recoveries_90",
        "takeons_90",
    ])

    key_metrics = [m for m in key_metrics if m in player_metrics.columns]

    metric_cols = st.columns(3)
    for i, metric in enumerate(key_metrics):
        with metric_cols[i % 3]:
            value = player_row.get(metric)
            metric_value = round(float(value), 2) if pd.notna(value) else "-"
            st.metric(format_player_metric_name(metric), metric_value)

with right_col:
    st.markdown("**Distribución del perfil**")

    profile_names = get_profile_names_from_dataset(
        player_metrics,
        player_row["position_group"]
    )

    fig_pie = plot_profile_pie(player_row, profile_names)

    if fig_pie is not None:
        fig_pie.update_traces(
            textinfo="percent",
            hoverinfo="label+percent"
        )
        fig_pie.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            title=None
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.info("No hay información de estilo disponible para este jugador.")

# --------------------------------------------------
# TABLA DETALLADA
# --------------------------------------------------
st.subheader("Métricas detalladas")

detail_cols = [
    "minutes_total",
    "passes_90",
    "passes_final_third_90",
    "passes_final_third_pct",
    "pass_accuracy",
    "progressive_passes_90",
    "progressive_passes_final_third_90",
    "progressive_pass_pct",
    "key_passes_90",
    "crosses_90",
    "cross_accuracy",
    "shots_90",
    "goals_90",
    "shots_on_target_90",
    "big_chances_90",
    "big_chances_created_90",
    "shot_accuracy",
    "goal_conversion",
    "inside_box_shot_pct",
    "avg_shot_distance_m",
    "recoveries_90",
    "recoveries_def_third_90",
    "recoveries_mid_third_90",
    "recoveries_final_third_90",
    "tackles_90",
    "tackle_success_pct",
    "interceptions_90",
    "clearances_90",
    "aerials_90",
    "aerial_win_pct",
    "blocked_passes_90",
    "fouls_90",
    "takeons_90",
    "successful_takeons_90",
    "takeon_success_pct",
    "dispossessed_90",
]

detail_cols = [c for c in detail_cols if c in player_metrics.columns]

detail_df = pd.DataFrame({
    "Métrica": [format_player_metric_name(c) for c in detail_cols],
    "Valor": [player_row[c] for c in detail_cols]
})

with st.expander("Ver métricas detalladas"):
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------
# EVENTOS DEL JUGADOR DESDE DUCKDB
# --------------------------------------------------
final_third_x = 66.67
player_name_sql = str(player_row["player_name"]).replace("'", "''")
team_name_sql = str(player_row["team_name"]).replace("'", "''")
season_sql = str(player_row["season"]).replace("'", "''")

st.subheader("Campograma de eventos")
st.caption("Visualización espacial de los eventos del jugador para contextualizar su producción y comportamiento sobe el campo.")
# --------------------------------------------------
# PARTIDOS DEL JUGADOR
# --------------------------------------------------
matches_sql = f"""
WITH player_matches AS (
    SELECT DISTINCT
        matchId,
        match_date,
        team_name AS player_team
    FROM top5_events
    WHERE player_name = '{player_name_sql}'
      AND team_name = '{team_name_sql}'
      AND season = '{season_sql}'
),
opponents AS (
    SELECT
        pm.matchId,
        pm.match_date,
        pm.player_team,
        MIN(CASE WHEN e.team_name <> pm.player_team THEN e.team_name END) AS opponent
    FROM player_matches pm
    LEFT JOIN top5_events e
        ON pm.matchId = e.matchId
    GROUP BY pm.matchId, pm.match_date, pm.player_team
)
SELECT
    matchId,
    match_date,
    player_team,
    opponent
FROM opponents
ORDER BY match_date DESC
"""

player_matches = query_events(matches_sql)

if not player_matches.empty:
    player_matches["match_label"] = (
        "vs "
        + player_matches["opponent"].fillna("Unknown")
        + " | "
        + player_matches["match_date"].astype(str).str[:10]
    )
    match_options = ["Todos los partidos"] + player_matches["match_label"].tolist()
else:
    match_options = ["Todos los partidos"]

# --------------------------------------------------
# SELECTORES
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    selected_match = st.selectbox("Partido", match_options, key="match_selector")

if selected_match == "Todos los partidos":
    event_type_options = [
        "Pass",
        "Shot",
        "BallRecovery",
        "Tackle",
        "Interception",
        "Clearance",
        "BlockedPass",
        "Aerial",
        "TakeOn",
    ]
else:
    event_type_options = [
        "Pass",
        "Shot",
        "Defensive actions",
        "BallRecovery",
        "Tackle",
        "Interception",
        "Clearance",
        "BlockedPass",
        "Aerial",
        "TakeOn",
    ]

with col2:
    selected_event_type = st.selectbox(
        "Tipo de visualización",
        event_type_options,
        format_func=format_event_type_name,
        key="event_type_selector"
    )

# --------------------------------------------------
# FILTRO BASE
# --------------------------------------------------
if selected_event_type == "Shot":
    event_filter_sql = """
    "Event Type" IN ('Goal', 'MissedShots', 'SavedShot', 'ShotOnPost', 'ChanceMissed')
    """
elif selected_event_type == "Defensive actions":
    event_filter_sql = """
    "Event Type" IN ('BallRecovery', 'Tackle', 'Interception', 'Clearance', 'BlockedPass')
    """
else:
    event_filter_sql = f""""Event Type" = '{selected_event_type}'"""

match_filter_sql = ""
if selected_match != "Todos los partidos":
    selected_match_id = int(
        player_matches.loc[player_matches["match_label"] == selected_match, "matchId"].iloc[0]
    )
    match_filter_sql = f" AND matchId = {selected_match_id} "

sql = f"""
SELECT
    matchId,
    match_date,
    team_name,
    player_name,
    "Event Type",
    Outcome,
    CAST("Start X" AS DOUBLE) AS "Start X",
    CAST("Start Y" AS DOUBLE) AS "Start Y",
    CAST(PassEndX AS DOUBLE) AS PassEndX,
    CAST(PassEndY AS DOUBLE) AS PassEndY,
    CAST(Length AS DOUBLE) AS Length,
    CAST(Angle AS DOUBLE) AS Angle,
    "KeyPass",
    "Cross"
FROM top5_events
WHERE player_name = '{player_name_sql}'
  AND team_name = '{team_name_sql}'
  AND season = '{season_sql}'
  AND {event_filter_sql}
  AND "Start X" IS NOT NULL
  AND "Start Y" IS NOT NULL
  {match_filter_sql}
ORDER BY match_date
"""

player_events = query_events(sql)

coord_cols = ["Start X", "Start Y", "PassEndX", "PassEndY", "Length", "Angle"]
for col in coord_cols:
    if col in player_events.columns:
        player_events[col] = pd.to_numeric(player_events[col], errors="coerce")

# --------------------------------------------------
# SUBFILTROS
# --------------------------------------------------
show_arrows = False

if selected_event_type == "Pass":
    pass_view = st.selectbox(
        "Subtipo de pase",
        [
            "All passes",
            "Progressive passes",
            "Final-third passes",
            "Passes into final third",
            "Key passes",
            "Crosses",
            "Long passes",
            "Forward passes",
            "Backward passes",
            "Lateral passes",
        ],
        format_func=format_pass_view_name,
        key="pass_view_selector"
    )

    show_arrows = st.checkbox("Mostrar flechas", value=True, key="show_arrows_checkbox")

    player_events["dx"] = player_events["PassEndX"] - player_events["Start X"]
    player_events["dx_m"] = player_events["dx"] * 105 / 100
    player_events["angle_deg"] = (player_events["Angle"] * 180 / 3.141592653589793) % 360

    player_events["is_progressive"] = (
        ((player_events["Start X"] < 50) & (player_events["dx_m"] >= 20)) |
        ((player_events["Start X"] >= 50) & (player_events["dx_m"] >= 10))
    )
    player_events["is_final_third_pass"] = player_events["Start X"] >= final_third_x
    player_events["is_pass_into_final_third"] = (
        (player_events["Start X"] < final_third_x) &
        (player_events["PassEndX"] >= final_third_x)
    )
    player_events["is_key_pass"] = player_events["KeyPass"] == "Yes"
    player_events["is_cross"] = player_events["Cross"] == "Yes"
    player_events["is_long_pass"] = player_events["Length"] > 30
    player_events["is_forward_pass"] = (
        (player_events["angle_deg"] >= 300) | (player_events["angle_deg"] <= 60)
    )
    player_events["is_backward_pass"] = (
        (player_events["angle_deg"] >= 120) & (player_events["angle_deg"] <= 240)
    )
    player_events["is_lateral_pass"] = (
        ~player_events["is_forward_pass"] & ~player_events["is_backward_pass"]
    )

    if pass_view == "Progressive passes":
        player_events = player_events[player_events["is_progressive"]].copy()
    elif pass_view == "Final-third passes":
        player_events = player_events[player_events["is_final_third_pass"]].copy()
    elif pass_view == "Passes into final third":
        player_events = player_events[player_events["is_pass_into_final_third"]].copy()
    elif pass_view == "Key passes":
        player_events = player_events[player_events["is_key_pass"]].copy()
    elif pass_view == "Crosses":
        player_events = player_events[player_events["is_cross"]].copy()
    elif pass_view == "Long passes":
        player_events = player_events[player_events["is_long_pass"]].copy()
    elif pass_view == "Forward passes":
        player_events = player_events[player_events["is_forward_pass"]].copy()
    elif pass_view == "Backward passes":
        player_events = player_events[player_events["is_backward_pass"]].copy()
    elif pass_view == "Lateral passes":
        player_events = player_events[player_events["is_lateral_pass"]].copy()

elif selected_event_type == "Shot":
    shot_view = st.selectbox(
        "Subtipo de tiro",
        [
            "All shots",
            "Goals",
            "Shots on target",
            "Missed shots",
            "Shot on post",
        ],
        format_func=format_shot_view_name,
        key="shot_view_selector"
    )

    player_events["is_goal"] = player_events["Event Type"] == "Goal"
    player_events["is_on_target"] = player_events["Event Type"].isin(["Goal", "SavedShot"])
    player_events["is_missed"] = player_events["Event Type"] == "MissedShots"
    player_events["is_post"] = player_events["Event Type"] == "ShotOnPost"

    if shot_view == "Goals":
        player_events = player_events[player_events["is_goal"]].copy()
    elif shot_view == "Shots on target":
        player_events = player_events[player_events["is_on_target"]].copy()
    elif shot_view == "Missed shots":
        player_events = player_events[player_events["is_missed"]].copy()
    elif shot_view == "Shot on post":
        player_events = player_events[player_events["is_post"]].copy()

st.write(f"Eventos seleccionados: {len(player_events):,}")

# --------------------------------------------------
# CAMPO
# --------------------------------------------------
if player_events.empty:
    st.info("No hay eventos para mostrar con los filtros seleccionados.")
else:
    if selected_event_type == "Shot":
        pitch = VerticalPitch(
            pitch_type="opta",
            pitch_color="white",
            line_color="black",
            linewidth=1,
            half=True
        )
        fig, ax = pitch.draw(figsize=(6, 8))
    else:
        pitch = Pitch(
            pitch_type="opta",
            pitch_color="white",
            line_color="black",
            linewidth=1
        )
        fig, ax = pitch.draw(figsize=(7, 5))

    # ----------------------------------------------
    # PASS MAP
    # ----------------------------------------------
    if selected_event_type == "Pass":
        success_colors = player_events["Outcome"].map(
            lambda x: "green" if x == "Successful" else "red"
        )

        pitch.scatter(
            player_events["Start X"],
            player_events["Start Y"],
            ax=ax,
            s=28,
            c=success_colors,
            alpha=0.8
        )

        if show_arrows:
            pass_events = player_events.dropna(
                subset=["Start X", "Start Y", "PassEndX", "PassEndY"]
            ).copy()

            for _, row in pass_events.iterrows():
                arrow_color = "green" if row["Outcome"] == "Successful" else "red"
                pitch.arrows(
                    row["Start X"],
                    row["Start Y"],
                    row["PassEndX"],
                    row["PassEndY"],
                    ax=ax,
                    width=1.5,
                    headwidth=3.5,
                    headlength=3.5,
                    color=arrow_color,
                    alpha=0.4
                )

    # ----------------------------------------------
    # SHOT MAP
    # ----------------------------------------------
    elif selected_event_type == "Shot":
        shot_color_map = {
            "Goal": {"color": "green", "label": "Gol"},
            "SavedShot": {"color": "blue", "label": "Parado"},
            "MissedShots": {"color": "red", "label": "Fuera"},
            "ShotOnPost": {"color": "orange", "label": "Palo"},
            "ChanceMissed": {"color": "purple", "label": "Ocasión fallada"},
        }

        for event_name, info in shot_color_map.items():
            subset = player_events[player_events["Event Type"] == event_name].copy()
            if not subset.empty:
                pitch.scatter(
                    subset["Start X"],
                    subset["Start Y"],
                    ax=ax,
                    s=20,
                    color=info["color"],
                    alpha=0.6,
                    label=info["label"]
                )

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    # ----------------------------------------------
    # DEFENSIVE ACTIONS
    # ----------------------------------------------
    elif selected_event_type == "Defensive actions" and selected_match != "Todos los partidos":
        defensive_style = {
            "BallRecovery": {"marker": "o", "label": "Recuperación", "color": "blue"},
            "Tackle": {"marker": "X", "label": "Entrada", "color": "red"},
            "Interception": {"marker": "s", "label": "Intercepción", "color": "green"},
            "Clearance": {"marker": "^", "label": "Despeje", "color": "orange"},
            "BlockedPass": {"marker": "D", "label": "Pase bloqueado", "color": "purple"},
        }

        for event_name, style in defensive_style.items():
            subset = player_events[player_events["Event Type"] == event_name].copy()
            if not subset.empty:
                pitch.scatter(
                    subset["Start X"],
                    subset["Start Y"],
                    ax=ax,
                    s=45,
                    color=style["color"],
                    alpha=0.85,
                    marker=style["marker"],
                    label=style["label"]
                )

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    # ----------------------------------------------
    # EVENTO INDIVIDUAL
    # ----------------------------------------------
    else:
        colors = player_events["Outcome"].map(
            lambda x: "green" if x == "Successful" else "red"
        )

        pitch.scatter(
            player_events["Start X"],
            player_events["Start Y"],
            ax=ax,
            s=35,
            c=colors,
            alpha=0.8
        )

    title = f"{player_row['player_name']} | {selected_event_type}"
    if selected_event_type == "Pass" and "pass_view" in locals():
        title += f" | {pass_view}"
    if selected_event_type == "Shot" and "shot_view" in locals():
        title += f" | {shot_view}"
    if selected_match != "Todos los partidos":
        title += f" | {selected_match}"

    ax.set_title(title, fontsize=10)

    fig.tight_layout()

    left, center, right = st.columns([1, 3, 1])
    with center:
        st.pyplot(fig)