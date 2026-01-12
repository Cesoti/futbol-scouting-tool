import streamlit as st
import pandas as pd
from mplsoccer import PyPizza
import matplotlib.pyplot as plt

from search_engine import (
    load_data,
    filter_players,
    build_feature_matrix,
    train_neighbors,
    get_similar_players,
    FEATURE_COLS,
)

st.set_page_config(page_title="Fútbol Scouting Tool", layout="wide")

CSV_PATH = "datos_laliga.csv"


@st.cache_data
def get_raw_data() -> pd.DataFrame:
    return load_data(CSV_PATH)


def make_comparison_pizza(df: pd.DataFrame, idx_a, idx_b):
    if idx_a not in df.index or idx_b not in df.index:
        st.error("Error: Índices de jugadores no válidos")
        return None

    player_a = df.loc[idx_a, "Player"]
    squad_a = df.loc[idx_a, "Squad"]
    player_b = df.loc[idx_b, "Player"]
    squad_b = df.loc[idx_b, "Squad"]

    values_a = df.loc[idx_a, FEATURE_COLS].astype(float).tolist()
    values_b = df.loc[idx_b, FEATURE_COLS].astype(float).tolist()

    max_range = [max(a, b, 1e-6) for a, b in zip(values_a, values_b)]
    min_range = [0.0] * len(FEATURE_COLS)

    params = [
        "Goles/90",
        "Asistencias/90",
        "G+A/90",
        "Cond. Prog/90",
        "Pases Prog/90",
        "Recep. Prog/90",
    ]

    background = "#101010"
    text = "#F2F2F2"
    color_a = "#A50044"
    color_b = "#00B4FF"

    baker = PyPizza(
        params=params,
        min_range=min_range,
        max_range=max_range,
        background_color=background,
        straight_line_color="#404040",
        straight_line_lw=1,
        last_circle_lw=1.5,
        last_circle_color="#404040",
        other_circle_lw=0.5,
        other_circle_color="#303030",
        inner_circle_size=10,
    )

    fig, ax = baker.make_pizza(
        values_a,
        compare_values=values_b,
        figsize=(9, 9),
        color_blank_space="same",
        blank_alpha=0.25,
        kwargs_slices=dict(
            facecolor=color_a,
            edgecolor="#222222",
            linewidth=1.2,
            zorder=2,
        ),
        kwargs_compare=dict(
            facecolor=color_b,
            edgecolor="#222222",
            linewidth=1.2,
            zorder=2,
        ),
        kwargs_params=dict(
            color=text,
            fontsize=11,
            va="center",
        ),
        kwargs_values=dict(
            color="#111111",
            fontsize=10,
            bbox=dict(
                edgecolor="none",
                facecolor=color_a,
                boxstyle="round,pad=0.25",
            ),
        ),
        kwargs_compare_values=dict(
            color="#111111",
            fontsize=10,
            bbox=dict(
                edgecolor="none",
                facecolor=color_b,
                boxstyle="round,pad=0.25",
            ),
        ),
    )

    title = f"{player_a} ({squad_a}) vs {player_b} ({squad_b})"
    fig.text(
        0.5,
        0.97,
        title,
        ha="center",
        va="center",
        fontsize=16,
        color=text,
    )

    fig.text(
        0.5,
        0.93,
        "Standard Stats LaLiga | per 90 mins",
        ha="center",
        va="center",
        fontsize=11,
        color="#BBBBBB",
    )

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=color_a, label=player_a),
        Patch(facecolor=color_b, label=player_b),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.03),
        frameon=False,
        fontsize=11,
    )

    fig.patch.set_facecolor(background)
    return fig


def main():
    st.sidebar.title("Fútbol Scouting Tool")

    if "similars_df" not in st.session_state:
        st.session_state.similars_df = None
    if "target_idx" not in st.session_state:
        st.session_state.target_idx = None
    if "df_aligned" not in st.session_state:
        st.session_state.df_aligned = None

    df_raw = get_raw_data()

    min_minutes = st.sidebar.slider(
        "Minutos mínimos",
        min_value=0,
        max_value=3000,
        value=400,
        step=50,
    )

    pos_options = ["ALL", "FW", "MF", "DF", "GK"]
    pos_choice = st.sidebar.selectbox("Posición", options=pos_options, index=1)

    df_filtered = filter_players(
        df_raw,
        min_minutes=min_minutes,
        position_prefix=pos_choice,
    )

    if df_filtered.empty:
        st.error("No hay jugadores que cumplan los filtros actuales.")
        return

    players = sorted(df_filtered["Player"].unique().tolist())
    selected_player = st.sidebar.selectbox("Elige un Jugador", options=players)

    st.header("Perfil del jugador seleccionado")

    row = df_filtered[df_filtered["Player"] == selected_player].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jugador", row["Player"])
    c2.metric("Equipo", row["Squad"])
    c3.metric("Posición", row["Pos"])
    c4.metric("Minutos", int(row["Min"]))

    c5, c6, c7 = st.columns(3)
    c5.metric("Goles totales", int(row["Gls"]))
    c6.metric("Asistencias totales", int(row["Ast"]))
    c7.metric("xG total", round(float(row["xG"]), 2))

    features_scaled, df_aligned, scaler = build_feature_matrix(df_filtered)

    if len(df_aligned) < 2:
        st.warning(
            "No hay suficientes jugadores con los filtros actuales para calcular similitudes."
        )
        return

    model = train_neighbors(features_scaled, metric="cosine")

    st.markdown("---")
    st.subheader("Jugadores similares en LaLiga (por 90 minutos)")

    if st.button("Encontrar parecidos"):
        target_idx, similars = get_similar_players(
            player_name=selected_player,
            df=df_aligned,
            features_scaled=features_scaled,
            model=model,
            top_k=5,
        )

        if target_idx is None or not similars:
            st.warning("No se encontraron jugadores similares con los filtros actuales.")
            st.session_state.similars_df = None
            st.session_state.target_idx = None
            st.session_state.df_aligned = None
        else:
            rows = []
            for idx, name, squad, dist in similars:
                sim = (1.0 - dist) * 100.0
                rows.append(
                    {
                        "Jugador": name,
                        "Equipo": squad,
                        "Distancia": round(dist, 4),
                        "Similitud (%)": round(sim, 1),
                        "Idx": idx,
                    },
                )
            st.session_state.similars_df = pd.DataFrame(rows)
            st.session_state.target_idx = target_idx
            st.session_state.df_aligned = df_aligned

    similars_df = st.session_state.similars_df
    target_idx = st.session_state.target_idx
    df_aligned = st.session_state.df_aligned

    if similars_df is not None and target_idx is not None:
        st.dataframe(
            similars_df[["Jugador", "Equipo", "Distancia", "Similitud (%)"]],
            use_container_width=True,
        )

    if similars_df is None or target_idx is None or df_aligned is None:
        return

    st.markdown("---")
    st.subheader("Comparación visual (Pizza Plot, métricas por 90)")

    options = similars_df["Jugador"].tolist()
    selected_comp = st.selectbox(
        "Elige un jugador de la lista para comparar",
        options=options,
    )

    selected_row = similars_df[similars_df["Jugador"] == selected_comp].iloc[0]
    idx_b = selected_row["Idx"]

    try:
        fig = make_comparison_pizza(df_aligned, idx_a=target_idx, idx_b=idx_b)
        if fig:
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al generar la pizza: {str(e)}")


if __name__ == "__main__":
    main()
