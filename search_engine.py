import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


CSV_PATH = "datos_laliga.csv"

STANDARD_COLS = [
    "Player",
    "Nation",
    "Pos",
    "Squad",
    "Age",
    "Min",
    "90s",
    "Gls",
    "Ast",
    "G+A",
    "xG",
    "xAG",
    "npxG+xAG",
    "PrgC",
    "PrgP",
    "PrgR",
    "Gls_90",
    "Ast_90",
    "G+A_90",
]

FEATURE_COLS = [
    "Gls_90",
    "Ast_90",
    "G+A_90",
    "PrgC_90",
    "PrgP_90",
    "PrgR_90",
]


def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=",",
        header=None,
        names=STANDARD_COLS,
        engine="python",
    )
    df.columns = df.columns.str.strip()

    # Convertir Age a numérico
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0)

    df["Min"] = pd.to_numeric(df["Min"], errors="coerce").fillna(0)
    df["90s"] = pd.to_numeric(df["90s"], errors="coerce")
    df["90s"] = df["90s"].replace(0, pd.NA)

    for base_col, per90_col in [
        ("PrgC", "PrgC_90"),
        ("PrgP", "PrgP_90"),
        ("PrgR", "PrgR_90"),
    ]:
        if base_col in df.columns:
            df[base_col] = pd.to_numeric(df[base_col], errors="coerce").fillna(0)
            if per90_col not in df.columns:
                df[per90_col] = df[base_col] / df["90s"]

    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def filter_players(
    df: pd.DataFrame,
    min_minutes: int = 400,
    position_prefix: str | None = None,
    max_age: int | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["Min"] = pd.to_numeric(df["Min"], errors="coerce").fillna(0)
    df = df[df["Min"] >= min_minutes]

    if position_prefix and position_prefix != "ALL":
        df = df[df["Pos"].astype(str).str.startswith(position_prefix)]

    if max_age is not None and "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(999)
        df = df[df["Age"] <= max_age]

    df = df[df["Player"].notna()]
    df = df.reset_index(drop=True)
    
    return df


def build_feature_matrix(df: pd.DataFrame):
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
    mask = df["Player"].str.lower() == player_name.lower()
    candidates = df[mask]

    if candidates.empty:
        return None, []

    if len(df) < 2:
        return None, []

    target_idx = candidates.index[0]
    target_vector = features_scaled.loc[target_idx].values.reshape(1, -1)
    
    distances, indices = model.kneighbors(target_vector)

    results = []
    for pos, dist in zip(indices[0], distances[0]):
        idx = df.index[pos]
        
        if idx == target_idx:
            continue

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

    return target_idx, results
