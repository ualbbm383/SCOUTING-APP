import pandas as pd
import streamlit as st

from src.player_clustering import (
    add_player_label,
    build_cluster_summary,
    build_single_cluster_metric_tables,
    get_player_row,
    get_supported_positions,
    plot_profile_pie,
    plot_umap_with_highlight,
    recalculate_and_update_position,
    update_player_position
)
from src.data_loader import load_player_metrics
from src.auth import check_login

if not check_login():
    st.stop()


st.set_page_config(page_title="Identificación de Perfiles de Jugadores", layout="wide")

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

st.title("Identificación de Perfiles de Jugadores")
st.markdown(
    """
    <div class="top-note">
        <div class="small-muted">
            Clasificación de jugadores por posición mediante técnicas de clustering.
            Los perfiles se generan a partir de métricas de juego y permiten identificar
            distintos roles dentro de cada posición. La distribución de perfiles se basa
            en la distancia relativa a los centroides del modelo, lo que permite reflejar
            la naturaleza híbrida de muchos jugadores.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
) 


def run_position_pipeline(position_group: str, min_minutes: int):
    # cargar siempre la versión más reciente del parquet enriquecido
    df_metrics = load_player_metrics().copy()

    result, _ = recalculate_and_update_position(
        df_full=df_metrics,
        position_group=position_group,
        min_minutes=min_minutes,
    )
    return result


def add_player_label_with_season(df_pos: pd.DataFrame) -> pd.DataFrame:
    df_pos = df_pos.copy()
    df_pos["player_label"] = (
        df_pos["player_name"].astype(str)
        + " | "
        + df_pos["team_name"].astype(str)
        + " | "
        + df_pos["season"].astype(str)
    )
    return df_pos


def get_player_row_with_label(df_pos: pd.DataFrame, player_label: str):
    res = df_pos[df_pos["player_label"] == player_label]
    if res.empty:
        return None
    return res.iloc[0]


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


# --------------------------------------------------
# FILTROS
# --------------------------------------------------

st.subheader("Configuración")

col1, col2 = st.columns([1, 1])

with col1:
    supported_positions = get_supported_positions()
    selected_position = st.selectbox(
        "Posición",
        supported_positions,
        index=2,
        format_func=format_position_name,
    )

with col2:
    min_minutes = st.number_input(
        "Minutos mínimos",
        min_value=0,
        value=600,
        step=50,
    )


# --------------------------------------------------
# ESTADO
# --------------------------------------------------
if "cluster_result" not in st.session_state:
    st.session_state["cluster_result"] = None
    st.session_state["cluster_position"] = None
    st.session_state["cluster_minutes"] = None

needs_refresh = (
    st.session_state["cluster_result"] is None
    or st.session_state["cluster_position"] != selected_position
    or st.session_state["cluster_minutes"] != min_minutes
)

# --------------------------------------------------
# PIPELINE CLUSTERING
# --------------------------------------------------
if needs_refresh:
    with st.spinner("Calculando clustering..."):
        result = run_position_pipeline(
            position_group=selected_position,
            min_minutes=min_minutes,
        )

    st.session_state["cluster_result"] = result
    st.session_state["cluster_position"] = selected_position
    st.session_state["cluster_minutes"] = min_minutes

result = st.session_state["cluster_result"]

# usar SIEMPRE la posición recién calculada para esta vista
df_pos = add_player_label_with_season(result["df_position"])
cluster_profile = result["cluster_profile"]
profile_names = result["profile_names"]

cluster_summary = build_cluster_summary(
    cluster_profile=cluster_profile,
    profile_names=profile_names,
    top_n=5,
)

st.divider()
# --------------------------------------------------
# RESUMEN DE ESTILOS
# --------------------------------------------------
st.subheader("Perfiles de jugador")

style_names = list(cluster_summary.keys())
n_styles = len(style_names)

style_cols = st.columns(n_styles)

for col, style_name in zip(style_cols, style_names):
    info = cluster_summary[style_name]

    top_df, bottom_df = build_single_cluster_metric_tables(
        cluster_summary=cluster_summary,
        cluster_name=style_name,
    )

    with col:
        st.markdown(f"### {style_name}")
        st.caption(f"Cluster {info['cluster_id']}")

        n_players = (df_pos["cluster_name"] == style_name).sum() if "cluster_name" in df_pos.columns else 0
        st.metric("Jugadores", int(n_players))

        st.markdown("**Métricas más altas**")
        if top_df is not None:
            st.dataframe(top_df, use_container_width=True, hide_index=True)

        st.markdown("**Métricas más bajas**")
        if bottom_df is not None:
            st.dataframe(bottom_df, use_container_width=True, hide_index=True)


st.divider()

# --------------------------------------------------
# SELECTOR DE JUGADOR
# --------------------------------------------------
st.subheader("Buscar jugador")

player_options = sorted(df_pos["player_label"].dropna().unique().tolist())

selected_player = st.selectbox(
    "Selecciona jugador",
    player_options,
    index=0 if player_options else None,
)

player_row = get_player_row_with_label(df_pos, selected_player) if selected_player else None


# --------------------------------------------------
# DETALLE DEL JUGADOR
# --------------------------------------------------
if player_row is not None:
    st.subheader("Perfil del jugador seleccionado")

    left, right = st.columns([1, 1])

    with left:
        cluster_name_value = player_row.get("cluster_name")
        if pd.isna(cluster_name_value) or str(cluster_name_value).strip() == "":
            cluster_name_value = "-"

        st.markdown(f"**Jugador:** {player_row['player_name']}")
        st.markdown(f"**Equipo:** {player_row['team_name']}")
        st.markdown(f"**Liga:** {player_row['league']}")
        st.markdown(f"**Temporada:** {player_row['season']}")
        st.markdown(f"**Posición:** {format_position_name(player_row['position_group'])}")
        st.markdown(f"**Minutos:** {int(player_row['minutes_total'])}")
        st.markdown(f"**Perfil principal:** {cluster_name_value}")

        cluster_value = player_row.get("cluster")
        if pd.notna(cluster_value):
            st.markdown(f"**ID de cluster:** {int(cluster_value)}")
        else:
            st.markdown("**ID de cluster:** -")

        prob_rows = []
        for cluster_id in sorted(profile_names.keys()):
            col = f"profile_{cluster_id+1}_pct"
            if col in player_row.index and pd.notna(player_row[col]):
                prob_rows.append(
                    {
                        "Perfil": profile_names[cluster_id],
                        "Probabilidad (%)": round(float(player_row[col]) * 100, 1),
                    }
                )

        if prob_rows:
            df_probs = pd.DataFrame(prob_rows).sort_values(
                "Probabilidad (%)", ascending=False
            )

            st.dataframe(df_probs, use_container_width=True, hide_index=True)
        else:
            st.info("No hay probabilidades disponibles para este jugador.")


    with right:
        pie_fig = plot_profile_pie(player_row, profile_names)
        if pie_fig is not None:
            pie_fig.update_traces(
                textinfo="percent",
                hoverinfo="label+percent"
            )

            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No hay probabilidades disponibles para este jugador.")


# --------------------------------------------------
# GRÁFICO GENERAL
# --------------------------------------------------
#st.subheader("Mapa del clustering")

scatter_fig = plot_umap_with_highlight(
    df_pos=df_pos,
    position_group=format_position_name(selected_position),
    player_label=selected_player,
)
st.plotly_chart(scatter_fig, use_container_width=True)

st.divider()

# CORRECCIÓN MANUAL DE LA POSICIÓN DE UN JUGADOR

IS_CLOUD = bool(st.secrets.get("IS_CLOUD", False))

if not IS_CLOUD:
    st.markdown("### Corrección manual de posición")

    df_all_players = load_player_metrics().copy()
    df_all_players = add_player_label_with_season(df_all_players)

    editable_players = sorted(df_all_players["player_label"].dropna().unique().tolist())

    edit_col1, edit_col2, edit_col3 = st.columns([2.2, 1.2, 1])

    with edit_col1:
        selected_player_to_edit = st.selectbox(
            "Jugador a corregir",
            editable_players,
            key="selected_player_to_edit"
        )

    player_edit_row = None
    if selected_player_to_edit:
        player_edit_row = df_all_players[df_all_players["player_label"] == selected_player_to_edit].iloc[0]

    with edit_col2:
        new_position_group = st.selectbox(
            "Nueva posición",
            get_supported_positions(),
            index=0,
            format_func=format_position_name,
            key="new_position_group"
        )

    with edit_col3:
        save_position_button = st.button("Guardar posición", width="stretch")

    if player_edit_row is not None:
        current_position = player_edit_row.get("position_group", None)
        current_position_label = format_position_name(current_position) if pd.notna(current_position) else "Sin asignar"
        st.caption(f"Posición actual: **{current_position_label}**")

    if save_position_button and selected_player_to_edit:
        try:
            update_player_position(
                player_label=selected_player_to_edit,
                new_position_group=new_position_group,
            )
            st.cache_data.clear()
            st.success(
                f"Posición guardada correctamente en metadata: {selected_player_to_edit} → {format_position_name(new_position_group)}. "
                "El cambio se aplicará al próximo recálculo de métricas enriquecidas y clusters."
            )
        except Exception as e:
            st.error(f"Error al guardar la posición: {e}")

else:
    st.info("La corrección manual de posición solo está disponible en el entorno local.")