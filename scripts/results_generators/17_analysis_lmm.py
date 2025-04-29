# ------------------------------------
# utilidades básicas
# ------------------------------------
import pandas as pd, numpy as np, re
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import os
DESIRED_FORMS = ("rectangular", "rombo", "triple rombo", "rampa ascendente")

def extract_duration(stim):
    try:
        return float(re.split(r',\s*', stim)[1].lower().replace('ms',''))
    except Exception:
        return np.nan

def prepare_df(raw):
    df = raw.copy()

    # -- desglosa estímulo en forma y duración ------------------------
    df["Forma"]       = (df["Estímulo"]
                         .str.split(',', 1).str[0].str.lower().str.strip())
    df["Duracion_ms"] = df["Estímulo"].apply(extract_duration)

    # -- filtra estímulos de interés ----------------------------------
    df = df[df["Forma"].isin(DESIRED_FORMS)]

    # -- id único de sitio: Día · X · Y -------------------------------
    df["Sitio"] = (df["Dia experimental"].astype(str) + "_" +
                   df["Coordenada_x"].astype(str)     + "_" +
                   df["Coordenada_y"].astype(str))

    # -- convierte a categoría para que MixedLM lo trate como nominal --
    df["Forma"]       = pd.Categorical(df["Forma"], ordered=False)
    df["Duracion_ms"] = pd.Categorical(df["Duracion_ms"], ordered=True)

    return df
# métricas a estudiar
METRICAS = ["lat_inicio_ms", "lat_pico_mayor_ms", "valor_pico_max",
            "dur_total_ms",  "delta_t_pico",      "num_movs"]

# transformaciones suaves para aproximar normalidad
TRANSFORM = {
    "valor_pico_max"    : np.log1p,
    "lat_pico_mayor_ms" : np.sqrt,
    "delta_t_pico"      : np.sqrt,
}

def fit_lmm(df, metric):
    # -- aplica transformación si procede ----------------------------
    y = metric + "_tf" if metric in TRANSFORM else metric
    if metric in TRANSFORM:
        df = df.copy()
        df[y] = TRANSFORM[metric](df[metric])

    # -- fórmula fija + intercepto aleatorio por 'Sitio' --------------
    form = f"{y} ~ C(Forma) * C(Duracion_ms)"
    md   = smf.mixedlm(form, data=df, groups="Sitio")
    m    = md.fit(reml=False, method="lbfgs")

    return m

def tidy_result(res, metric):
    coefs = res.summary().tables[1]                # Wald z & p
    sel   = coefs.index.str.contains("C\\(Forma\\)|C\\(Duracion_ms\\)")
    out   = (coefs.loc[sel, ["Coef.", "P>|z|"]]
                   .rename(columns={"Coef.":"beta", "P>|z|":"p"}))
    out["metric"] = metric
    return out.reset_index(names="term")

def run_analysis(agg):
    dfs = []
    for metric in METRICAS:
        res  = fit_lmm(agg, metric)
        dfs += [tidy_result(res, metric)]
    df_all = pd.concat(dfs, ignore_index=True)

    # --- corrección FDR (Benjamini-Hochberg) -------------------------
    reject, p_adj, *_ = multipletests(df_all["p"], method="fdr_bh")
    df_all["p_adj"] = p_adj
    df_all["sig"]   = reject
    return df_all

# -------------------------------------------------------------------
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\plot_trials4mad'

submov_path = os.path.join(output_comparisons_dir, 'aggregated_df_enriched.csv')
raw   = pd.read_csv(submov_path)
ready = prepare_df(raw)

# 3-A  ·  análisis global (todos los días)
global_stats = run_analysis(ready)
global_stats.to_csv("stats_global_lmm.csv", index=False)

# 3-B  ·  análisis por día
out = []
for d, g in ready.groupby("Dia experimental"):
    tmp = run_analysis(g)
    tmp["dia"] = d
    out.append(tmp)
pd.concat(out).to_csv("stats_by_day_lmm.csv", index=False)
