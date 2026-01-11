import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

CSV_PATH = "datos_laliga.csv"

STANDARD_COLS = [
    "Rk", "Player", "Nation", "Pos", "Squad", "CompID", "Born",
    "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt",
    "CrdY", "CrdR",
    "xG", "npxG", "xAG", "npxG+xAG",
    "PrgC", "PrgP", "PrgR",
    "Gls_90", "Ast_90", "G+A_90", "G-PK_90", "G+A-PK_90",
    "xG_90", "xAG_90", "xG+xAG_90", "npxG_90", "npxG+xAG_90",
    "Matches",
]

FEATURE_COLS = [
    "Gls",
    "Ast",
    "xG",
    "npxG",
    "xAG",
    "PrgC",
    "PrgP",
    "PrgR",
]


def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=";",
        thousands=",",
        header=None,
        names=STANDARD_COLS,
        engine="python",
    )
    df.columns = df.columns.str.strip()
    return df


def filter_players(
    df: pd.DataFrame,
    min_minutes: int = 400,
    position_prefix: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
    df = df[df["Min"] >= min_minutes]

    if position_prefix and position_prefix != "ALL":
        df = df[df["Pos"].astype(str).str.startswith(position_prefix)]

    return df[df["Player"].notna()]


def build_feature_matrix(df: pd.DataFrame):
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not feature_cols:
        msg = "No feature columns found in dataframe."
        raise ValueError(msg)

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
    n_samples = len(features_scaled)
    if n_samples < 2:
        msg = f"Not enough samples to train NearestNeighbors (n_samples={n_samples})."
        raise ValueError(msg)

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
