# --- IMPORTS ---
import pandas as pd
import numpy as np
import pandas as pd
import os

# --- NUEVAS FUNCIONES DE FILTRADO Y PREPARACIÓN ---
def desired_forms():
    """Return list of desired pulse forms in canonical lower-case."""
    return ["rectangular", "rombo", "triple rombo", "triple_rombo", "rampa ascendente", "rampa_ascendente"]

def extract_duration(df):
    """Ensure 'Duración (ms)' column exists and is integer, based on 'Duracion_ms'."""
    if "Duración (ms)" not in df.columns and "Duracion_ms" in df.columns:
        df.loc[:, "Duración (ms)"] = pd.to_numeric(df["Duracion_ms"], errors="coerce").round().astype("Int64")
    return df

def prep_for_anova(df):
    """
    Filtra y normaliza el DataFrame para mantener solo los ensayos válidos Gaussian-based
    con las formas y duraciones empleadas en los análisis de resultados.
    """
    df = df.copy()

    # 1️⃣ Filtrar modelo Gaussian-based
    if "MovementType" in df.columns:
        df = df[df["MovementType"].astype(str).str.lower() == "gaussian-based"]

    # 2️⃣ Normalizar nombres de forma del pulso
    if "Forma_del_Pulso" in df.columns:
        df.loc[:, "Forma_del_Pulso"] = df["Forma_del_Pulso"].astype(str).str.strip().str.lower()
        valid_forms = ["rectangular", "rombo", "triple rombo", "triple_rombo", "rampa ascendente", "rampa_ascendente"]
        df = df[df["Forma_del_Pulso"].isin(valid_forms)]
        map_pulse = {
            "rectangular": "Rectangular",
            "rombo": "Rombo",
            "triple rombo": "Triple rombo",
            "triple_rombo": "Triple rombo",
            "rampa ascendente": "Rampa ascendente",
            "rampa_ascendente": "Rampa ascendente",
        }
        df.loc[:, "Forma del estímulo"] = df["Forma_del_Pulso"].map(map_pulse).fillna(df["Forma_del_Pulso"].str.title())

    # 3️⃣ Duración en ms
    if "Duracion_ms" in df.columns:
        df.loc[:, "Duración (ms)"] = pd.to_numeric(df["Duracion_ms"], errors="coerce").round().astype("Int64")

    # 4️⃣ Quitar filas sin métricas cinemáticas
    metricas = [
        "lat_inicio_ms", "lat_pico_mayor_ms", "valor_pico_max",
        "dur_total_ms", "delta_t_pico", "num_movs"
    ]
    for c in metricas:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in metricas if c in df.columns], how="all")

    # 5️⃣ Limpiar duplicados
    df = df.drop_duplicates()

    return df
# --- FUNCIONES AUXILIARES PARA NORMALIZACIÓN DE TABLAS ---

# --- NUEVA FUNCIÓN: ORDENAR POR COORDENADAS ---
def ordena_por_coordenadas(df):
    """Ordena las filas según la columna 'Coordenada (x, y)' en orden fijo (6,3) → (10,3) → (12,1)."""
    orden = ["(6, 3)", "(10, 3)", "(12, 1)"]
    if "Coordenada (x, y)" in df.columns:
        df["__orden_coord"] = pd.Categorical(df["Coordenada (x, y)"], categories=orden, ordered=True)
        df = df.sort_values("__orden_coord").drop(columns="__orden_coord")
    return df

# --- NUEVA FUNCIÓN: ORDENAR POR ESTÍMULO ---
def ordena_por_estimulo(df):
    """Ordena las filas según la columna 'Forma del estímulo' en el orden fijo Rectangular → Rombo → Triple rombo → Rampa ascendente."""
    orden = ["Rectangular", "Rombo", "Triple rombo", "Rampa ascendente"]
    if "Forma del estímulo" in df.columns:
        df["__orden_estimulo"] = pd.Categorical(df["Forma del estímulo"], categories=orden, ordered=True)
        df = df.sort_values("__orden_estimulo").drop(columns="__orden_estimulo")
    return df

base_dir = "/Users/brunobustos/Documents/GitHub/stimulationb15/data/plot_trials"

for fname in ["aggregated_df_enriched.csv", "summary_sample_sizes.csv", "descriptive_statistics_table.csv"]:
    path = os.path.join(base_dir, fname)
    print(f"\n📄 Archivo: {fname}")
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    print("Columnas:", list(df.columns))
    print(df.dtypes)
    print(df.head(3))

# --- CONFIGURACIÓN DE RUTAS ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
base_dir = os.path.join(project_root, "data", "plot_trials")

path_agg = os.path.join(base_dir, "aggregated_df_enriched.csv")
path_n = os.path.join(base_dir, "summary_sample_sizes.csv")
desc_stats_path = os.path.join(base_dir, "descriptive_statistics_table.csv")

print("📂 Ruta base detectada:", base_dir)
# --- CARGA DE DATOS ---
# Chequeo de existencia de archivos antes de leer
if not os.path.exists(path_agg):
    raise FileNotFoundError(f"No se encontró el archivo esperado: {path_agg}\nVerifica que los archivos CSV estén en la carpeta 'data/plot_trials'.")
if not os.path.exists(path_n):
    raise FileNotFoundError(f"No se encontró el archivo esperado: {path_n}\nVerifica que los archivos CSV estén en la carpeta 'data/plot_trials'.")

def load_csv_safely(path):
    """Carga un CSV con tolerancia a errores de codificación y corrige columnas numéricas mal exportadas."""
    import re

    # 1️⃣ Cargar con tolerancia de codificación
    df = pd.read_csv(path, encoding='latin1', low_memory=False)

    # 2️⃣ Limpiar columnas numéricas exportadas como texto con puntos intermedios
    for col in df.columns:
        if df[col].dtype == object:
            # Eliminar espacios, comas y puntos que actúan como separadores
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .apply(lambda x: re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", x))  # elimina puntos entre miles
                .str.replace(",", ".", regex=False)  # usa punto decimal
            )
            # 3️⃣ Intentar conversión a número
            converted = pd.to_numeric(df[col], errors='ignore')
            if converted.dtype == object:
                print(f"Columna '{col}' no convertida a numérico, dtype: {converted.dtype}")
                print(f"Valores de muestra: {df[col].head(5).tolist()}")
            else:
                df[col] = converted

    return df

# Procesar aggregated_df_enriched.csv
df = load_csv_safely(path_agg)
df = prep_for_anova(df)

# --- MÉTRICAS CINEMÁTICAS PREFERIDAS ---
metricas_preferidas = [
    "lat_inicio_ms", "lat_pico_mayor_ms", "valor_pico_max",
    "dur_total_ms", "delta_t_pico", "num_movs"
]
# Solo mantener las que existan realmente en el DataFrame
metricas_preferidas = [m for m in metricas_preferidas if m in df.columns]
if not metricas_preferidas:
    raise ValueError("❌ No se encontraron columnas de métricas preferidas en aggregated_df_enriched.csv")

# === Filtrar solo las 6 articulaciones principales ===
# Solo se incluyen las seis articulaciones principales; se excluyen nudillo y dedo.
#valid_body_parts = ['Frente', 'Hombro', 'Bicep', 'Braquiradial', 'Codo', 'Muneca']
valid_body_parts = ['Codo', 'Hombro', 'Muneca']
if 'body_part' in df.columns:
    df = df[df['body_part'].isin(valid_body_parts)].copy()

# --- AGRUPACIÓN POR ENSAYO PARA REPRESENTAR CONDICIONES ANALIZADAS ---
if "Ensayo_Key" in df.columns:
    group_cols_ensayo = ["Ensayo_Key", "body_part", "Coordenada_x", "Coordenada_y", "Forma_del_Pulso", "Duracion_ms"]
    df_ensayo = df.groupby(group_cols_ensayo)[metricas_preferidas].mean().reset_index()
else:
    df_ensayo = df.copy()

# --- FILTROS CONSISTENTES CON 'resumenes_finales.py' ---
# (El filtrado por articulaciones principales ya fue aplicado arriba)
# Solo modelo Gaussian-based y body parts de interés (seis articulaciones principales)
# parts_of_interest = ['Frente', 'Hombro', 'Bíceps', 'Braquiorradial', 'Codo', 'Muñeca']
# df = df[df['MovementType'].astype(str).str.lower() == 'gaussian-based']
# if 'body_part' in df.columns:
#     df = df[df['body_part'].isin(parts_of_interest)]

# Filtrar formas de pulso válidas y duraciones estándar
valid_forms = ['rectangular', 'rombo', 'triple rombo', 'triple_rombo', 'rampa ascendente', 'rampa_ascendente']
valid_durations = [500, 750, 1000]
df['Forma_del_Pulso'] = df['Forma_del_Pulso'].astype(str).str.strip().str.lower()
df = df[df['Forma_del_Pulso'].isin(valid_forms)]
if 'Duracion_ms' in df.columns:
    df['Duracion_ms'] = pd.to_numeric(df['Duracion_ms'], errors='coerce').round().astype('Int64')
    df = df[df['Duracion_ms'].isin(valid_durations)]

# Procesar summary_sample_sizes.csv
df_n = load_csv_safely(path_n)

# Procesar descriptive_statistics_table.csv si existe
if os.path.exists(desc_stats_path):
    desc_stats_df = pd.read_csv(desc_stats_path)
else:
    desc_stats_df = None

# --- NORMALIZAR NOMBRES DE COLUMNAS ---
df.columns = [c.replace("Coord_x", "Coordenada_x").replace("Coord_y", "Coordenada_y") for c in df.columns]
df_n.columns = [c.replace("Coord_x", "Coordenada_x").replace("Coord_y", "Coordenada_y") for c in df_n.columns]
if desc_stats_df is not None:
    desc_stats_df.columns = [c.strip() for c in desc_stats_df.columns]



print("\n🔹 Procesando 'aggregated_df_enriched.csv' para generar tabla de distribución no paramétrica (percentiles).")

# Filtrar solo el modelo que usamos en resultados (Gaussian-based)
# Ya hemos filtrado arriba con los mismos criterios que en 'resumenes_finales.py'
df_pct = df_ensayo.copy()

# Selección robusta de métricas: usa preferidas y si faltan, busca alternativas *_tf
metricas_preferidas = ["lat_inicio_ms", "lat_pico_mayor_ms", "valor_pico_max", "dur_total_ms", "delta_t_pico", "num_movs"]
alternativas = {
    "lat_inicio_ms": "lat_inicio_ms_tf",
    "lat_pico_mayor_ms": "lat_pico_mayor_ms_tf",
    "valor_pico_max": "valor_pico_max_tf",
    "delta_t_pico": "delta_t_pico_tf",
}
metricas = []
for m in metricas_preferidas:
    if m in df_pct.columns:
        metricas.append(m)
    elif m in alternativas and alternativas[m] in df_pct.columns:
        metricas.append(alternativas[m])

if not metricas:
    raise ValueError("❌ No se encontraron columnas de métricas esperadas (ni sus alternativas *_tf) en aggregated_df_enriched.csv")

print(f"✅ Métricas detectadas automáticamente: {metricas}")

# Agregación no paramétrica por condición
def agg_q(x):
    return pd.Series({
        "q1": np.nanpercentile(x, 25),
        "mediana": np.nanpercentile(x, 50),
        "q3": np.nanpercentile(x, 75),
        "n": x.count()
    })
# --- FUNCIONES AUXILIARES PARA NORMALIZACIÓN DE TABLAS ---

def normaliza_forma_del_pulso(df):
    """Normaliza la columna 'Forma_del_Pulso' → 'Forma del estímulo' con nombres capitalizados."""
    if "Forma_del_Pulso" in df.columns:
        df.loc[:, "Forma_del_Pulso"] = df["Forma_del_Pulso"].astype(str).str.strip().str.lower()
        map_pulse = {
            "rectangular": "Rectangular",
            "rombo": "Rombo",
            "triple rombo": "Triple rombo",
            "triple_rombo": "Triple rombo",
            "rampa ascendente": "Rampa ascendente",
            "rampa_ascendente": "Rampa ascendente",
        }
        df.loc[:, "Forma del estímulo"] = df["Forma_del_Pulso"].map(map_pulse).fillna(df["Forma_del_Pulso"].str.title())
        df.drop(columns=["Forma_del_Pulso"], inplace=True)
    return df

def normaliza_duracion_ms(df):
    """Convierte Duracion_ms → Duración (ms)"""
    if "Duracion_ms" in df.columns:
        df.loc[:, "Duración (ms)"] = pd.to_numeric(df["Duracion_ms"], errors="coerce").round().astype("Int64")
        df.drop(columns=["Duracion_ms"], inplace=True)
    return df

def unifica_coordenada(df):
    """Combina Coordenada_x y Coordenada_y en una sola columna textual."""
    if "Coordenada_x" in df.columns and "Coordenada_y" in df.columns:
        df.loc[:, "Coordenada (x, y)"] = df.apply(
            lambda r: f"({int(r['Coordenada_x'])}, {int(r['Coordenada_y'])})", axis=1
        )
    return df

def renombra_columnas_formales(df):
    """Renombra métricas a nombres formales en español, usando px/s para la velocidad."""
    rename_map = {
        "lat_inicio_ms_mean±sd": "Latencia al inicio (ms)",
        "lat_pico_mayor_ms_mean±sd": "Latencia al pico mayor (ms)",
        "valor_pico_max_mean±sd": "Amplitud del pico mayor de velocidad (px/s)",
        "dur_total_ms_mean±sd": "Duración total del movimiento (ms)",
        "delta_t_pico_mean±sd": "Diferencia primer–pico mayor (ms)",
        "num_movs_mean±sd": "Número de submovimientos",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df
group_cols = ["Coordenada_x", "Coordenada_y", "Forma del estímulo", "Duración (ms)"]
for col in ["Coordenada_x", "Coordenada_y"]:
    if col not in df_pct.columns:
        raise ValueError(f"Falta la columna requerida '{col}' en aggregated_df_enriched.csv")

# construir dict de agregadores para .agg con nombres estables
agg_dict = {m: [("q1", lambda s: np.nanpercentile(s, 25)),
                ("mediana", lambda s: np.nanpercentile(s, 50)),
                ("q3", lambda s: np.nanpercentile(s, 75)),
                ("n", "count")] for m in metricas}

# Asegurar formato correcto antes de agrupar
df_pct = normaliza_forma_del_pulso(df_pct)
df_pct = normaliza_duracion_ms(df_pct)
df_pct = unifica_coordenada(df_pct)
tabla_raw = df_pct.groupby(group_cols).agg(agg_dict)

# Aplana columnas MultiIndex -> 'm__q1', 'm__mediana', etc.
tabla_raw.columns = [f"{m}__{stat}" for (m, stat) in tabla_raw.columns]
tabla_raw = tabla_raw.reset_index()

# Mapeo a nombres formales para la tabla final
formal_map = {
    "lat_inicio_ms": "Latencia al inicio (ms)",
    "lat_inicio_ms_tf": "Latencia al inicio (ms)",
    "lat_pico_mayor_ms": "Latencia al pico mayor (ms)",
    "lat_pico_mayor_ms_tf": "Latencia al pico mayor (ms)",
    "valor_pico_max": "Amplitud del pico de velocidad (px/s)",
    "valor_pico_max_tf": "Amplitud del pico de velocidad (px/s)",
    "dur_total_ms": "Duración total (ms)",
    "delta_t_pico": "Diferencia inicio–pico (ms)",
    "delta_t_pico_tf": "Diferencia inicio–pico (ms)",
    "num_movs": "Número de submovimientos",
}

from collections import OrderedDict
rows = []
for _, r in tabla_raw.iterrows():
    out = OrderedDict()
    out["Coordenada (x, y)"] = f"({int(r['Coordenada_x'])}, {int(r['Coordenada_y'])})"
    out["Forma del estímulo"] = r["Forma del estímulo"]
    out["Duración (ms)"] = int(r["Duración (ms)"])
    # cada métrica en formato: mediana [q1–q3]
    for m in metricas:
        q1_col = f"{m}__q1"
        med_col = f"{m}__mediana"
        q3_col = f"{m}__q3"
        q1 = r[q1_col] if q1_col in r else np.nan
        med = r[med_col] if med_col in r else np.nan
        q3 = r[q3_col] if q3_col in r else np.nan
        out[formal_map.get(m, m)] = f"{med:.1f} [{q1:.1f}–{q3:.1f}]"
    # n: número de ensayos únicos para la condición
    if "Ensayo_Key" in df_ensayo.columns:
        mask = (
            (df_ensayo["Coordenada_x"] == r["Coordenada_x"])
            & (df_ensayo["Coordenada_y"] == r["Coordenada_y"])
            & (df_ensayo["Forma_del_Pulso"].astype(str).str.lower() == str(r["Forma del estímulo"]).strip().lower())
            & (df_ensayo["Duracion_ms"] == r["Duración (ms)"])
        )
        n_val = df_ensayo.loc[mask, "Ensayo_Key"].nunique()
    else:
        # fallback a conteo de filas
        n_val = None
        for m in metricas:
            n_tmp = r.get(f"{m}__n", np.nan)
            if pd.notna(n_tmp):
                n_val = int(n_tmp)
                break
    out["n"] = n_val
    rows.append(out)

tabla_percentiles_final = pd.DataFrame(rows)


# --- Nuevo orden jerárquico: por coordenada y dentro de cada una por estímulo ---
orden_coord = ["(6, 3)", "(10, 3)", "(12, 1)"]
orden_estim = ["Rectangular", "Rombo", "Triple rombo", "Rampa ascendente"]

if "Coordenada (x, y)" in tabla_percentiles_final.columns and "Forma del estímulo" in tabla_percentiles_final.columns:
    tabla_percentiles_final["__orden_coord"] = pd.Categorical(tabla_percentiles_final["Coordenada (x, y)"], categories=orden_coord, ordered=True)
    tabla_percentiles_final["__orden_estim"] = pd.Categorical(tabla_percentiles_final["Forma del estímulo"], categories=orden_estim, ordered=True)
    tabla_percentiles_final = tabla_percentiles_final.sort_values(["__orden_coord", "__orden_estim", "Duración (ms)"]).drop(columns=["__orden_coord", "__orden_estim"])

# Exportar CSV y LaTeX (solo el entorno tabular, sin \begin{table}), con nombres consistentes
csv_path_percentiles_sd = os.path.join(base_dir, "tabla_percentiles_sd.csv")
tabla_percentiles_final.to_csv(csv_path_percentiles_sd, index=False)

latex_path_percentiles_sd = os.path.join(base_dir, "tabla_percentiles_sd.tex")
latex_str = tabla_percentiles_final.to_latex(index=False, escape=False)
latex_str = latex_str.split("\\begin{tabular}")[1]
with open(latex_path_percentiles_sd, "w", encoding="utf-8") as f:
    f.write("\\begin{tabular}" + latex_str)

print(f"✅ Tabla no paramétrica (percentiles) exportada correctamente:\n  - CSV: {csv_path_percentiles_sd}\n  - TEX: {latex_path_percentiles_sd}")

# --- PROCESAMIENTO DE summary_sample_sizes.csv ---
print("\n🔹 Resumen de 'summary_sample_sizes.csv': Total de ensayos (n_trials) por condición y modelo.")

# --- PROCESAMIENTO DE summary_sample_sizes.csv ---
print("\n🔹 Resumen de 'summary_sample_sizes.csv': Total de ensayos (n_trials) por condición y modelo.")

if 'n_trials' in df_n.columns:
    # Intentar agrupar por las columnas que parezcan relevantes para condición y modelo
    group_cols_n = [col for col in ['Condition', 'Model', 'Forma_del_Pulso', 'Duracion_ms'] if col in df_n.columns]
    if group_cols_n:
        summary_n = df_n.groupby(group_cols_n)['n_trials'].sum().reset_index()
        print(summary_n)
    else:
        print("No se encontraron columnas adecuadas para agrupar en 'summary_sample_sizes.csv'. Mostrando primeras filas:")
        print(df_n.head())
else:
    print("La columna 'n_trials' no se encontró en 'summary_sample_sizes.csv'. Mostrando primeras filas:")
    print(df_n.head())

summary_out_path = os.path.join(base_dir, "tabla_summary_n.csv")
df_n.to_csv(summary_out_path, index=False)
print(f"📁 Tabla independiente de summary guardada en: {summary_out_path}")

# --- PROCESAMIENTO DE descriptive_statistics_table.csv ---
if desc_stats_df is not None:
    print("\n🔹 Análisis independiente de 'descriptive_statistics_table.csv': columnas numéricas principales y sus promedios generales.")
    numeric_cols = desc_stats_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        means = desc_stats_df[numeric_cols].mean()
        print("Columnas numéricas encontradas:", numeric_cols)
        print("Promedios generales:")
        print(means)
    else:
        print("No se encontraron columnas numéricas en 'descriptive_statistics_table.csv'.")
else:
    print("\n⚠️ El archivo 'descriptive_statistics_table.csv' no fue encontrado, no se realizó análisis independiente.")


# --- NUEVA SECCIÓN: TABLAS AGRUPADAS EXTENDIDAS ---
print("\n🔹 Generando tablas agrupadas extendidas según lógica de otros scripts...")

# Definir columnas de agrupación extendida
group_cols_extendido = [
    'Dia experimental', 'Coordenada_x', 'Coordenada_y', 'body_part', 'Forma_del_Pulso', 'Duracion_ms', 'MovementType'
]
# Solo usar columnas que existan en df_ensayo
group_cols_extendido = [col for col in group_cols_extendido if col in df_ensayo.columns]

# Agrupar y calcular mean, std, count
tabla_grouped = (
    df_ensayo.groupby(group_cols_extendido)[metricas]
      .agg(['mean', 'std', 'count'])
)
tabla_grouped = tabla_grouped.round(2)
tabla_grouped.columns = ["_".join(col).strip() for col in tabla_grouped.columns.values]
tabla_grouped.reset_index(inplace=True)

# Formatear mean±sd y agregar columna n (n = número de Ensayo_Key únicos)
for m in metricas:
    tabla_grouped[f"{m}_mean±sd"] = tabla_grouped.apply(
        lambda r: f"{r[f'{m}_mean']:.1f} ± {r[f'{m}_std']:.1f}", axis=1
    )
if "Ensayo_Key" in df_ensayo.columns:
    # n = número de Ensayo_Key únicos por combinación
    def get_n(row):
        mask = np.ones(len(df_ensayo), dtype=bool)
        for col in group_cols_extendido:
            mask &= df_ensayo[col] == row[col]
        return df_ensayo.loc[mask, "Ensayo_Key"].nunique()
    tabla_grouped['n'] = tabla_grouped.apply(get_n, axis=1)
else:
    tabla_grouped['n'] = tabla_grouped[f"{metricas[0]}_count"].astype(int)

# Seleccionar columnas finales
cols_finales_ext = group_cols_extendido + [f"{m}_mean±sd" for m in metricas] + ['n']
tabla_grouped_final = tabla_grouped[cols_finales_ext]

# Normaliza columnas clave
tabla_grouped_final = normaliza_forma_del_pulso(tabla_grouped_final)
tabla_grouped_final = normaliza_duracion_ms(tabla_grouped_final)
tabla_grouped_final = unifica_coordenada(tabla_grouped_final)
tabla_grouped_final = renombra_columnas_formales(tabla_grouped_final)

# Ordena columnas finales coherentes
cols_export_ext = []
if "Dia experimental" in tabla_grouped_final.columns:
    cols_export_ext.append("Dia experimental")
if "body_part" in tabla_grouped_final.columns:
    cols_export_ext.append("body_part")
if "Coordenada (x, y)" in tabla_grouped_final.columns:
    cols_export_ext.append("Coordenada (x, y)")
cols_export_ext += [c for c in ["Forma del estímulo", "Duración (ms)"] if c in tabla_grouped_final.columns]
cols_export_ext += [
    "Latencia al inicio (ms)",
    "Latencia al pico mayor (ms)",
    "Amplitud del pico mayor de velocidad (px/s)",
    "Duración total del movimiento (ms)",
    "Diferencia primer–pico mayor (ms)",
    "Número de submovimientos",
    "n"
]
cols_export_ext = [c for c in cols_export_ext if c in tabla_grouped_final.columns]
tabla_grouped_final = tabla_grouped_final[cols_export_ext]

# --- Nuevo orden jerárquico: por coordenada y dentro de cada una por estímulo ---
if "Coordenada (x, y)" in tabla_grouped_final.columns and "Forma del estímulo" in tabla_grouped_final.columns:
    tabla_grouped_final["__orden_coord"] = pd.Categorical(tabla_grouped_final["Coordenada (x, y)"], categories=orden_coord, ordered=True)
    tabla_grouped_final["__orden_estim"] = pd.Categorical(tabla_grouped_final["Forma del estímulo"], categories=orden_estim, ordered=True)
    tabla_grouped_final = tabla_grouped_final.sort_values(["__orden_coord", "__orden_estim", "Duración (ms)"]).drop(columns=["__orden_coord", "__orden_estim"])
csv_path_promedios_sd = os.path.join(base_dir, "tabla_promedios_sd.csv")
tabla_grouped_final.to_csv(csv_path_promedios_sd, index=False)
latex_path_promedios_sd = os.path.join(base_dir, "tabla_promedios_sd.tex")
latex_str = tabla_grouped_final.to_latex(index=False, escape=False)
latex_str = latex_str.split("\\begin{tabular}")[1]
with open(latex_path_promedios_sd, "w", encoding="utf-8") as f:
    f.write("\\begin{tabular}" + latex_str)
print(f"Tabla agrupada principal CSV guardada en: {csv_path_promedios_sd}")
print(f"Tabla agrupada principal LaTeX guardada en: {latex_path_promedios_sd}")

# Guardar tabla extendida original (tabla_promedios_filtrada.csv/tex) como antes
csv_path_ext = os.path.join(base_dir, "tabla_promedios_filtrada.csv")
tabla_grouped_final.to_csv(csv_path_ext, index=False)
latex_path_ext = os.path.join(base_dir, "tabla_promedios_filtrada.tex")
with open(latex_path_ext, "w", encoding="utf-8") as f:
    latex_str = tabla_grouped_final.to_latex(index=False, escape=False,
                                             caption="Resumen de métricas cinemáticas (media ± DE) por agrupación extendida",
                                             label="tab:promedios_filtrada")
    latex_str = latex_str.split("\\begin{table}")[1]
    f.write("\\begin{table}" + latex_str)
print(f"Tabla agrupada extendida CSV guardada en: {csv_path_ext}")
print(f"Tabla agrupada extendida LaTeX guardada en: {latex_path_ext}")


# --- TABLA PROMEDIOS POR DÍA (CORREGIDA Y CONSISTENTE) ---
# Agrupar por coordenadas (x, y), forma y duración
group_cols_dia = [col for col in ['Coordenada_x', 'Coordenada_y', 'Forma_del_Pulso', 'Duracion_ms'] if col in df_ensayo.columns]
tabla_dia = (
    df_ensayo.groupby(group_cols_dia)[metricas]
      .agg(['mean', 'std', 'count'])
)
tabla_dia = tabla_dia.round(2)
tabla_dia.columns = ["_".join(col).strip() for col in tabla_dia.columns.values]
tabla_dia.reset_index(inplace=True)
# Formatear mean±sd y agregar columna n (n = número de Ensayo_Key únicos)
for m in metricas:
    tabla_dia[f"{m}_mean±sd"] = tabla_dia.apply(
        lambda r: f"{r[f'{m}_mean']:.1f} ± {r[f'{m}_std']:.1f}", axis=1
    )
if "Ensayo_Key" in df_ensayo.columns:
    def get_n(row):
        mask = np.ones(len(df_ensayo), dtype=bool)
        for col in group_cols_dia:
            mask &= df_ensayo[col] == row[col]
        return df_ensayo.loc[mask, "Ensayo_Key"].nunique()
    tabla_dia['n'] = tabla_dia.apply(get_n, axis=1)
else:
    tabla_dia['n'] = tabla_dia[f"{metricas[0]}_count"].astype(int)
cols_finales_dia = group_cols_dia + [f"{m}_mean±sd" for m in metricas] + ['n']
tabla_dia_final = tabla_dia[cols_finales_dia]

# Normaliza columnas clave
tabla_dia_final = normaliza_forma_del_pulso(tabla_dia_final)
tabla_dia_final = normaliza_duracion_ms(tabla_dia_final)
tabla_dia_final = unifica_coordenada(tabla_dia_final)
tabla_dia_final = renombra_columnas_formales(tabla_dia_final)

# Aplicar orden categórico a "Forma del estímulo"
orden_estimulos = pd.CategoricalDtype(categories=["Rectangular", "Rombo", "Triple rombo", "Rampa ascendente"], ordered=True)
if "Forma del estímulo" in tabla_dia_final.columns:
    tabla_dia_final["Forma del estímulo"] = tabla_dia_final["Forma del estímulo"].astype(orden_estimulos)

# Ordenar filas por "Coordenada (x, y)", "Forma del estímulo", "Duración (ms)"
sort_cols = []
if "Coordenada (x, y)" in tabla_dia_final.columns:
    sort_cols.append("Coordenada (x, y)")
if "Forma del estímulo" in tabla_dia_final.columns:
    sort_cols.append("Forma del estímulo")
if "Duración (ms)" in tabla_dia_final.columns:
    sort_cols.append("Duración (ms)")
if sort_cols:
    tabla_dia_final = tabla_dia_final.sort_values(sort_cols).reset_index(drop=True)

# Ordenar columnas exactamente igual que las otras tablas
cols_orden = [
    "Coordenada (x, y)",
    "Forma del estímulo",
    "Duración (ms)",
    "Latencia al inicio (ms)",
    "Latencia al pico mayor (ms)",
    "Amplitud del pico mayor de velocidad (px/s)",
    "Duración total del movimiento (ms)",
    "Diferencia primer–pico mayor (ms)",
    "Número de submovimientos",
    "n"
]
cols_export_dia_corr = [c for c in cols_orden if c in tabla_dia_final.columns]
tabla_dia_final_corr = tabla_dia_final[cols_export_dia_corr]

# --- Nuevo orden jerárquico: por coordenada y dentro de cada una por estímulo ---
orden_coord = ["(6, 3)", "(10, 3)", "(12, 1)"]
orden_estim = ["Rectangular", "Rombo", "Triple rombo", "Rampa ascendente"]

if "Coordenada (x, y)" in tabla_dia_final_corr.columns and "Forma del estímulo" in tabla_dia_final_corr.columns:
    tabla_dia_final_corr["__orden_coord"] = pd.Categorical(tabla_dia_final_corr["Coordenada (x, y)"], categories=orden_coord, ordered=True)
    tabla_dia_final_corr["__orden_estim"] = pd.Categorical(tabla_dia_final_corr["Forma del estímulo"], categories=orden_estim, ordered=True)
    tabla_dia_final_corr = tabla_dia_final_corr.sort_values(["__orden_coord", "__orden_estim", "Duración (ms)"]).drop(columns=["__orden_coord", "__orden_estim"])

# Exportar a nuevo archivo LaTeX (solo entorno tabular, sin table/caption)
latex_path_dia_corr = os.path.join(base_dir, "tabla_promedios_por_dia_corr.tex")
latex_str = tabla_dia_final_corr.to_latex(index=False, escape=False)
latex_str = latex_str.split("\\begin{tabular}")[1]
with open(latex_path_dia_corr, "w", encoding="utf-8") as f:
    f.write("\\begin{tabular}" + latex_str)
print(f"✅ Tabla promedios por día corregida exportada correctamente en formato LaTeX:\n  - TEX: {latex_path_dia_corr}")

# --- TABLA PROMEDIOS POR BODY PART ---
# Agrupar por body_part e incluir coordenadas para evitar filas repetidas
group_cols_bp = [col for col in ['body_part', 'Coordenada_x', 'Coordenada_y', 'Forma_del_Pulso', 'Duracion_ms'] if col in df_ensayo.columns]
tabla_bp = (
    df_ensayo.groupby(group_cols_bp)[metricas]
      .agg(['mean', 'std', 'count'])
)
tabla_bp = tabla_bp.round(2)
tabla_bp.columns = ["_".join(col).strip() for col in tabla_bp.columns.values]
tabla_bp.reset_index(inplace=True)
for m in metricas:
    tabla_bp[f"{m}_mean±sd"] = tabla_bp.apply(
        lambda r: f"{r[f'{m}_mean']:.1f} ± {r[f'{m}_std']:.1f}", axis=1
    )
if "Ensayo_Key" in df_ensayo.columns:
    def get_n(row):
        mask = np.ones(len(df_ensayo), dtype=bool)
        for col in group_cols_bp:
            mask &= df_ensayo[col] == row[col]
        return df_ensayo.loc[mask, "Ensayo_Key"].nunique()
    tabla_bp['n'] = tabla_bp.apply(get_n, axis=1)
else:
    tabla_bp['n'] = tabla_bp[f"{metricas[0]}_count"].astype(int)
cols_finales_bp = group_cols_bp + [f"{m}_mean±sd" for m in metricas] + ['n']
tabla_bp_final = tabla_bp[cols_finales_bp]

# Normaliza columnas clave
tabla_bp_final = normaliza_forma_del_pulso(tabla_bp_final)
tabla_bp_final = normaliza_duracion_ms(tabla_bp_final)
tabla_bp_final = unifica_coordenada(tabla_bp_final)
tabla_bp_final = renombra_columnas_formales(tabla_bp_final)

# Ordena coordenadas y elimina duplicados antes de exportar archivos
tabla_bp_final = ordena_por_coordenadas(tabla_bp_final)
# Ordenar por estímulo en orden fijo después de coordenada
tabla_bp_final = ordena_por_estimulo(tabla_bp_final)
tabla_bp_final = tabla_bp_final.drop_duplicates()

# Ordena columnas coherentes
cols_export_bp = []
if "body_part" in tabla_bp_final.columns:
    cols_export_bp.append("body_part")
if "Coordenada (x, y)" in tabla_bp_final.columns:
    cols_export_bp.append("Coordenada (x, y)")
cols_export_bp += [c for c in ["Forma del estímulo", "Duración (ms)"] if c in tabla_bp_final.columns]
cols_export_bp += [
    "Latencia al inicio (ms)",
    "Latencia al pico mayor (ms)",
    "Amplitud del pico mayor de velocidad (px/s)",
    "Duración total del movimiento (ms)",
    "Diferencia primer–pico mayor (ms)",
    "Número de submovimientos",
    "n"
]
cols_export_bp = [c for c in cols_export_bp if c in tabla_bp_final.columns]
tabla_bp_final = tabla_bp_final[cols_export_bp]

csv_path_bp = os.path.join(base_dir, "tabla_promedios_por_bodypart.csv")
tabla_bp_final.to_csv(csv_path_bp, index=False)
latex_path_bp = os.path.join(base_dir, "tabla_promedios_por_bodypart.tex")
with open(latex_path_bp, "w", encoding="utf-8") as f:
    latex_str = tabla_bp_final.to_latex(index=False, escape=False,
                                        caption="Resumen de métricas cinemáticas (media ± DE) por body part",
                                        label="tab:promedios_por_bodypart")
    latex_str = latex_str.split("\\begin{table}")[1]
    f.write("\\begin{table}" + latex_str)
print(f"Tabla promedios por body part CSV guardada en: {csv_path_bp}")
print(f"Tabla promedios por body part LaTeX guardada en: {latex_path_bp}")

print("\n✅ Procesamiento completo. Archivos generados correctamente.")