import pandas as pd
from typing import List

RAW_COLUMNS = [
    "Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born",
    "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt",
    "CrdY", "CrdR",
    "xG", "npxG", "xAG", "npxG+xAG",
    "PrgC", "PrgP", "PrgR",
    "Gls_90", "Ast_90", "G+A_90", "G-PK_90", "G+A-PK_90",
    "xG_90", "xAG_90", "xG+xAG_90", "npxG_90", "npxG+xAG_90"
]

OUTPUT_COLUMNS = [
    "Player", "Nation", "Pos", "Squad", "Age",
    "Min", "90s",
    "Gls", "Ast", "G+A", "xG", "xAG", "npxG+xAG",  
    "PrgC", "PrgP", "PrgR",
    "Gls_90", "Ast_90", "G+A_90"
]


def load_raw_fbref(csv_path: str) -> pd.DataFrame:
    """
    Lee el CSV bruto de FBref. Formato: separador ';' y encoding latin-1.
    Se salta líneas mal formadas.
    """
    print(f"Leyendo archivo bruto: {csv_path}")
    df = pd.read_csv(
        csv_path,
        sep=";",
        engine="python",
        encoding="latin-1",
        on_bad_lines="skip"
    )
    return df


def parse_age(age_str: str) -> int:
    """
    Convierte '25-347' → 25 (solo la edad en años).
    """
    if pd.isna(age_str):
        return 0
    if isinstance(age_str, (int, float)):
        return int(age_str)
    
    age_str = str(age_str).strip()
    if "-" in age_str:
        return int(age_str.split("-")[0])
    
    try:
        return int(age_str)
    except:
        return 0


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deja solo las columnas que nos interesan.
    """
    missing: List[str] = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        print("OJO: faltan columnas en el CSV bruto:")
        for c in missing:
            print(f"  - {c}")
        cols_present = [c for c in RAW_COLUMNS if c in df.columns]
        df = df[cols_present]
    else:
        df = df[RAW_COLUMNS]
    
    return df


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia la columna de posiciones (ej: 'FW,MF' -> 'FW').
    Si no existe 'Pos', devuelve el df tal cual.
    """
    if "Pos" not in df.columns:
        return df
    
    def _main_pos(pos: str) -> str:
        if isinstance(pos, str):
            return pos.split(",")[0].strip()
        return pos
    
    df["Pos"] = df["Pos"].apply(_main_pos)
    return df


def convert_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Convierte columnas a numéricas, forzando errores a NaN.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_per90(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura las métricas por 90 minutos si no vinieran en el CSV.
    """
    if "90s" not in df.columns and "Min" in df.columns:
        df["90s"] = df["Min"] / 90.0
    
    if "Gls_90" not in df.columns and "Gls" in df.columns and "90s" in df.columns:
        df["Gls_90"] = df["Gls"] / df["90s"]
    
    if "Ast_90" not in df.columns and "Ast" in df.columns and "90s" in df.columns:
        df["Ast_90"] = df["Ast"] / df["90s"]
    
    if "G+A_90" not in df.columns and "G+A" in df.columns and "90s" in df.columns:
        df["G+A_90"] = df["G+A"] / df["90s"]
    
    return df


def select_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona y ordena las columnas finales que usa la app.
    """
    cols_present = [c for c in OUTPUT_COLUMNS if c in df.columns]
    return df[cols_present]


def run_etl(input_path: str = "fbref_raw.csv",
            output_path: str = "datos_laliga.csv") -> None:
    """
    Pipeline completo: lee fbref_raw.csv, limpia y genera datos_laliga.csv.
    """
    df_raw = load_raw_fbref(input_path)
    df_clean = clean_columns(df_raw)
    
    if "Player" not in df_clean.columns:
        print("Columnas leídas del CSV:")
        print(list(df_clean.columns))
        raise SystemExit("El CSV no tiene encabezados esperados. Revisa fbref_raw.csv.")
    
    if "Age" in df_clean.columns:
        print("Parseando columna Age...")
        df_clean["Age"] = df_clean["Age"].apply(parse_age)
    
    df_clean = normalize_positions(df_clean)
    
    numeric_cols = [
        "Age", "Min", "90s",
        "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt",
        "CrdY", "CrdR",
        "xG", "npxG", "xAG", "npxG+xAG",
        "PrgC", "PrgP", "PrgR",
        "Gls_90", "Ast_90", "G+A_90", "G-PK_90", "G+A-PK_90",
        "xG_90", "xAG_90", "xG+xAG_90", "npxG_90", "npxG+xAG_90"
    ]
    
    df_clean = convert_numeric(df_clean, numeric_cols)
    df_clean = build_per90(df_clean)
    
    df_clean = df_clean[df_clean["Player"].astype(str).str.strip() != ""]
    df_clean = df_clean.reset_index(drop=True)
    
    df_final = select_output(df_clean)
    
    print(f"Guardando salida en: {output_path}")
    print(f"Columnas finales: {list(df_final.columns)}")
    print(f"Total de jugadores: {len(df_final)}")
    
    if "Age" in df_final.columns:
        print("\nMuestra de edades:")
        print(df_final[["Player", "Age"]].head(10))
    
    df_final.to_csv(
        output_path,
        sep=",",
        index=False,
        header=False  
    )
    
    print("ETL completado. 'datos_laliga.csv' actualizado.")


if __name__ == "__main__":
    run_etl()
