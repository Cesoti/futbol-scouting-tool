import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

CSV_PATH = "datos_laliga.csv"

# Nombres que vamos a asignar al leer el CSV sin cabeceras.
# Deben coincidir en número y orden con las columnas que tiene el archivo.
STANDARD_COLS = [
    "Player",      # 0
    "Nation",      # 1
    "Pos",         # 2
    "Squad",       # 3
    "Dummy",       # 4 (columna vacía / basura, se ignora luego)
    "Min",         # 5
    "90s",         # 6
    "Gls",         # 7
    "Ast",         # 8
    "G+A",         # 9
    "xG",          # 10
    "xAG",         # 11
    "npxG+xAG",    # 12
    "PrgC",        # 13
    "PrgP",        # 14
    "PrgR",        # 15
    "Gls_90",      # 16
    "Ast_90",      # 17
    "G+A_90",      # 18
]

FEATURE_COLS = [
    "Gls",
    "Ast",
    "xG",
    "xAG",
    "npxG+xAG",
    "PrgC",
    "PrgP",
    "PrgR",
]


def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Lee datos_laliga.csv sin cabeceras, separado por comas, y asigna nombres.
    """
    df = pd.read_csv(
        csv_path,
        sep=",",
        header=None,
        names=STANDARD_COLS,
        engine="python",
    )
    df.columns = df.columns.str.strip()

    # Nos quitamos la columna basura si existe
    if "Dummy" in df.columns:
        df = df.drop(columns=["Dummy"])

    return df


def filter_players(
    df: pd.DataFrame,
    min_minutes: int = 400,
    position_prefix: str | None = None,
) -> pd.DataFrame:
    """
    Filtra por minutos mínimos y por prefijo de posición (FW, MF, DF, GK).
    """
    df = df.copy()

    df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
    df = df[df["Min"] >= min_minutes]

    if position_prefix and position_prefix != "ALL":
        df = df[df["Pos"].astype(str).str.startswith(position_prefix)]

    return df[df["Player"].notna()]


def build_feature_matrix(df: pd.DataFrame):
    """
    Construye la matriz de características y la normaliza a [0, 1].
    """
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns found in dataframe.")

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    features_scaled = pd.DataFrame(
        scaled,
        columns=feature_cols,
        index=df.index,
    )
    return features_scaled, df, scaler


def train_neighbors(
    features_scaled: pd.DataFrame,
    metric: str = "cosine",
    n_neighbors: int = 6,
) -> NearestNeighbors:
    """
    Entrena un modelo NearestNeighbors con las features escaladas.
    """
    n_samples = len(features_scaled)
    if n_samples < 2:
        raise ValueError(
            f"Not enough samples to train NearestNeighbors (n_samples={n_samples})."
        )

    n_neighbors = min(n_neighbors, n_samples)
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="brute",
    )
    model.fit(features_scaled.values)
    return model


def get_similar_players(
    player_name: str,
    df: pd.DataFrame,
    features_scaled: pd.DataFrame,
    model: NearestNeighbors,
    top_k: int = 5,
):
    """
    Devuelve jugadores similares a player_name usando el modelo kNN.
    """
    mask = df["Player"].str.lower() == player_name.lower()
    candidates = df[mask]

    if candidates.empty:
        return None, []

    if len(df) < 2:
        return None, []

    target_label = candidates.index[0]
    target_pos = df.index.get_loc(target_label)

    target_vector = features_scaled.iloc[target_pos].values.reshape(1, -1)
    distances, indices = model.kneighbors(target_vector)

    positions = indices[0]
    dists = distances[0]

    results = []
    for pos, dist in zip(positions, dists):
        if pos == target_pos:
            continue

        idx = df.index[pos]
        results.append(
            (
                idx,
                df.loc[idx, "Player"],
                df.loc[idx, "Squad"],
                float(dist),
            ),
        )
        if len(results) >= top_k:
            break

    return target_label, results
