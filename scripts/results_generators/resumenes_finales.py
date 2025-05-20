import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, least_squares
from math import sqrt
import matplotlib
matplotlib.use('Agg')
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from typing import Optional            # agrega arriba
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from scipy.interpolate import griddata
#from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import re
import shutil
import glob
import textwrap
import math

import plotly.graph_objects as go
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from patsy.contrasts import Sum  

from matplotlib.ticker import MultipleLocator, FormatStrFormatter, NullFormatter
import matplotlib.gridspec as gridspec

# ——— Tema global para todos los gráficos ———
# sns.set_theme(style="whitegrid")


# --- CONFIGURACIÓN DEL LOGGING
log_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\filtered_processing_log.txt'
logging.basicConfig(filename=log_path, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# --- DIRECTORIOS
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\plot_trials'
if not os.path.exists(output_comparisons_dir):
    os.makedirs(output_comparisons_dir)

# Diccionarios de colores y etiquetas
body_parts_specific_colors = {
    'Frente': 'blue',
    'Hombro': 'orange',
    'Codo': 'green',
    'Muneca': 'red',
    'Braquiradial': 'grey',
    'Bicep': 'brown'
}
body_parts = list(body_parts_specific_colors.keys())

# -------------------------------
# Formas de pulso a incluir en TODOS los análisis
# -------------------------------
desired_forms = ["rectangular", "rombo", "triple rombo", "rampa ascendente"]
key_forms = ["rectangular", "rombo"]

shape_colors = {
    "rectangular": "orange",
    "rombo": "blue",
    "rampa ascendente": "green",
    "triple rombo": "red"
}

metric_labels = {
    "lat_inicio_ms": "Latencia al Inicio",
    "lat_pico_mayor_ms": "Latencia al Pico Mayor",
    "valor_pico_max": "Amplitud del Pico Mayor",
    "dur_total_ms": "Duración Total",
    "delta_t_pico": "Diferencia Primer-Pico Mayor",
    "num_movs": "Número de Submovimientos"
}

# ─── NUEVO bloque global ───────────────────────────────────────────────
TRANSFORMS = {
    'valor_pico_max'     : np.log1p,          # log(1+x)
    'lat_pico_mayor_ms'  : np.sqrt,
    'delta_t_pico'       : np.sqrt
}
# ----------------------------------------------------------------------

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

# -------------------------------
# Función para extraer la duración
# -------------------------------
def extract_duration(s):
    try:
        if not isinstance(s, str):
            return np.nan
        parts = s.split(',')
        if len(parts) < 2:
            return np.nan
        dur_str = parts[1].strip().lower().replace("ms", "").replace(",", ".")
        return float(dur_str)
    except Exception as e:
        logging.error(f"Error al extraer duración de '{s}': {e}")
        return np.nan
    
# ─── utilitario ───────────────────────────────────────────────────────
def _maybe_transform(df, metric):
    """Aplica la transformación definida en TRANSFORMS y devuelve:
       (serie transformada, nombre_columna_nueva)"""
    if metric in TRANSFORMS:
        new_col = metric + '_tf'
        df[new_col] = TRANSFORMS[metric](df[metric])
        return df, new_col
    return df, metric
# ----------------------------------------------------------------------
# ─── 1) PRE-ANÁLISIS ÚNICO ─────────────────────────────────────────────────────
def prep_for_anova(df, *,
                   metric: str = None,
                   model:  str = 'Gaussian-based',
                   day=None, coord_x=None, coord_y=None, body_part=None):
    """
    Pipeline único:
      • Filtra por modelo, día, coord, parte corporal
      • Extrae Forma_del_Pulso y Duracion_ms (como category)
      • Filtra solo desired_forms
      • Dropna en metric si se pide
    """
    d = df.copy()

    # filtros contextuales
    if model     is not None: d = d[d['MovementType'] == model]
    if day       is not None: d = d[d['Dia experimental'] == day]
    if coord_x   is not None: d = d[d['Coordenada_x']    == coord_x]
    if coord_y   is not None: d = d[d['Coordenada_y']    == coord_y]
    if body_part is not None: d = d[d['body_part']       == body_part]

    # forma y duración
    d['Forma_del_Pulso'] = (
        d['Estímulo']
         .str.split(',', 1).str[0]
         .str.lower().str.strip()
    )
    d['Duracion_ms'] = (
        d['Estímulo']
         .apply(extract_duration)
         .astype('Int64')      # permite NaN
         .astype('category')
    )

    # filtrar solo las formas que queremos
    d = d[d['Forma_del_Pulso'].isin(desired_forms)]

    # si pedimos un metric, dropna
    if metric is not None:
        d = d.dropna(subset=[metric, 'Forma_del_Pulso', 'Duracion_ms'])

    return d


# ─── 2) ANOVA SIN HC3, TIPO II ─────────────────────────────────────────────────
from statsmodels.stats.anova import anova_lm

def run_anova(df, formula, typ=2):
    """
    Ajusta OLS sin cov_type y devuelve tabla ANOVA tipo II o III.
    """
    mod = ols(formula, data=df).fit()
    return anova_lm(mod, typ=typ)


def dynamic_anova(df, formula, typ=2,
                  data_metric=None, group_var=None, test_id=None):
    """
    Si typ==3, chequea Levene y activa HC3 sólo si Levene_p<.05.
    """
    mod = ols(formula, data=df).fit()

    if typ == 3 and data_metric is not None:
        # 1) guardamos Shapiro & Levene
        check_assumptions(mod, data_metric, group_var, prefix=test_id)
        lev_p = ASSUMPTION_RESULTS[-1]['Levene_p']
        # 2) elegimos cov_type
        if lev_p < 0.05:
            return anova_lm(mod, typ=3, cov_type='hc3')
    # fallback sin robustez
    return anova_lm(mod, typ=typ)


# ─── 3) CÁLCULO DE TABLAS ANOVA (tipo II) ───────────────────────────────────────
def _calc_anova_table(agg: pd.DataFrame) -> pd.DataFrame:
    agg = prep_for_anova(agg)  # ya incluye Forma_del_Pulso y Duracion_ms
    metrics = selected_metrics
    grouping = ['Dia experimental', 'Coordenada_x', 'Coordenada_y', 'body_part', 'MovementType']
    formulas = {
        'factorial': lambda m: f"{m} ~ C(Forma_del_Pulso, Sum) * C(Duracion_ms, Sum)"
    }

    rows = []
    for (dia, x, y, bp, mtype), group in agg.groupby(grouping):
        for met in metrics:
            sub = group.dropna(subset=[met])
            if sub.shape[0] < 3: 
                continue
            # aplicamos transform si toca
            sub, m_used = _maybe_transform(sub, met)
            # Sólo factorial SS Tipo III con contrastes Sum
            form = formulas['factorial'](m_used)
            aov = anova_lm(ols(form, data=sub).fit(), typ=3)
            for factor in ['C(Forma_del_Pulso, Sum)',
               'C(Duracion_ms, Sum)',
               'C(Forma_del_Pulso, Sum):C(Duracion_ms, Sum)']:

                rows.append({
                    'Dia experimental': dia,
                    'Coordenada_x'    : x,
                    'Coordenada_y'    : y,
                    'body_part'       : bp,
                    'MovementType'    : mtype,
                    'Metric'          : met,
                    'Factor'          : factor,
                    'sum_sq'          : aov.loc[factor, 'sum_sq'],
                    'df'              : aov.loc[factor, 'df'],
                    'F'               : aov.loc[factor, 'F'],
                    'PR(>F)'          : aov.loc[factor, 'PR(>F)'],
                    'Partial_Eta_Sq'  : calc_partial_eta_sq(aov, factor, 'Residual'),
                    'N'               : len(sub)
                })

    return pd.DataFrame(rows)


# ─── 4) ANOVA 2-VIAS CONTROLADA (opcional) ─────────────────────────────────────
def run_factorial_anova(df, metric):
    # 1) filtrado sólo Rectangular vs Rombo y duraciones 500/1000
    df = prep_for_anova(df, metric=metric)
    df = df[
        df['Forma_del_Pulso'].isin(key_forms) &
        df['Duracion_ms'].astype(int).isin([500, 1000])
    ]
    df, y = _maybe_transform(df, metric)
    if df.shape[0] < 4:
        return dict.fromkeys(
            ['forma_p','duracion_p','interaccion_p',
             'forma_F','duracion_F','interaccion_F'], np.nan)

    # 2) un único modelo factorial SS Tipo III con contrastes Sum
    formula = f"{y} ~ C(Forma_del_Pulso, Sum) * C(Duracion_ms, Sum)"

    mod = ols(formula, data=df).fit()
    aov = anova_lm(mod, typ=3)

    # 3) extraer los tres términos
    return {
        'forma_p'       : aov.loc['C(Forma_del_Pulso, Sum)',            'PR(>F)'],
        'duracion_p'    : aov.loc['C(Duracion_ms, Sum)',               'PR(>F)'],
        'interaccion_p' : aov.loc['C(Forma_del_Pulso, Sum):C(Duracion_ms, Sum)', 'PR(>F)'],
        'forma_F'       : aov.loc['C(Forma_del_Pulso, Sum)',            'F'],
        'duracion_F'    : aov.loc['C(Duracion_ms, Sum)',               'F'],
        'interaccion_F' : aov.loc['C(Forma_del_Pulso, Sum):C(Duracion_ms, Sum)', 'F']
    }




# -------------------------------
# ETAPA 1: Agregación de métricas
# -------------------------------
def aggregate_trial_metrics_extended(submovements_df):
    fs = 100.0  # frames/s

    def agg_threshold(group):
        inicio_idx = group["Latencia al Inicio (s)"].idxmin()
        pico_idx = group["Latencia al Pico (s)"].idxmin()
        max_idx  = group["Valor Pico (velocidad)"].idxmax()
        return pd.Series({
            "lat_inicio_ms": group.loc[inicio_idx, "Latencia al Inicio (s)"] * 1000,
            "lat_primer_pico_ms": group.loc[pico_idx, "Latencia al Pico (s)"] * 1000,
            "lat_pico_ultimo_ms": group["Latencia al Pico (s)"].max() * 1000,
            "dur_total_ms": ((group["Fin Movimiento (Frame)"].max() - group["Inicio Movimiento (Frame)"].min()) / fs) * 1000,
            "valor_pico_inicial": group.loc[inicio_idx, "Valor Pico (velocidad)"],
            "valor_pico_max": group["Valor Pico (velocidad)"].max(),
            "num_movs": group.shape[0],
            "lat_inicio_mayor_ms": group.loc[max_idx, "Latencia al Inicio (s)"] * 1000,
            "lat_pico_mayor_ms": group.loc[max_idx, "Latencia al Pico (s)"] * 1000,
            "delta_t_pico": (group.loc[max_idx, "Latencia al Pico (s)"] - group.loc[pico_idx, "Latencia al Pico (s)"]) * 1000
        })

    def agg_gaussian(group):
        return pd.Series({
            "lat_inicio_ms": (group.apply(lambda row: (row["mu_gauss"] - 2 * row["sigma_gauss"]) - 1, axis=1).min()) * 1000,
            "lat_primer_pico_ms": (group["mu_gauss"].min() - 1) * 1000,
            "lat_pico_ultimo_ms": (group["mu_gauss"].max() - 1) * 1000,
            "dur_total_ms": (group.apply(lambda row: (row["mu_gauss"] + 2 * row["sigma_gauss"]) - 1, axis=1).max() -
                             group.apply(lambda row: (row["mu_gauss"] - 2 * row["sigma_gauss"]) - 1, axis=1).min()) * 1000,
            "valor_pico_inicial": group["A_gauss"].min(),
            "valor_pico_max": group["A_gauss"].max(),
            "num_movs": group.shape[0],
            "lat_inicio_mayor_ms": ((group.apply(lambda row: row["mu_gauss"] - 2 * row["sigma_gauss"], axis=1)
                                     .loc[group["A_gauss"].idxmax()] - 1) * 1000),
            "lat_pico_mayor_ms": ((group.loc[group["A_gauss"].idxmax(), "mu_gauss"] - 1) * 1000),
            "delta_t_pico": (group.loc[group["A_gauss"].idxmax(), "mu_gauss"] - 
                                 group.loc[group["mu_gauss"].idxmin(), "mu_gauss"]) * 1000
        })

    def agg_minjerk(group):
        return pd.Series({
            "lat_inicio_ms": (group["t_start"].min() - 1) * 1000,
            "lat_primer_pico_ms": (group["t_peak"].min() - 1) * 1000,
            "lat_pico_ultimo_ms": (group["t_peak"].max() - 1) * 1000,
            "dur_total_ms": (group["t_end"].max() - group["t_start"].min()) * 1000,
            "valor_pico_inicial": group["valor_pico"].min(),
            "valor_pico_max": group["valor_pico"].max(),
            "num_movs": group.shape[0],
            "lat_inicio_mayor_ms": (group.loc[group["valor_pico"].idxmax(), "t_start"] - 1) * 1000,
            "lat_pico_mayor_ms": (group.loc[group["valor_pico"].idxmax(), "t_peak"] - 1) * 1000,
            "delta_t_pico": (group.loc[group["valor_pico"].idxmax(), "t_peak"] -
                                 group.loc[group["t_peak"].idxmin(), "t_peak"]) * 1000
        })

    grouping_cols = ['Ensayo_Key', 'Estímulo', 'MovementType',
                     'Dia experimental', 'body_part', 'Coordenada_x', 'Coordenada_y',
                     'Distancia Intracortical']
    agg_list = []
    for name, group in submovements_df.groupby(grouping_cols):
        ens_key, estimulo, movement_type, dia, bp, cx, cy, ic_dist = name

        if movement_type == 'Threshold-based':
            agg_metrics = agg_threshold(group)
        elif movement_type == 'Gaussian-based':
            agg_metrics = agg_gaussian(group)
        elif movement_type == 'MinimumJerk':
            agg_metrics = agg_minjerk(group)
        else:
            agg_metrics = pd.Series({k: np.nan for k in [
                "lat_inicio_ms", "lat_primer_pico_ms", "lat_pico_ultimo_ms",
                "dur_total_ms", "valor_pico_inicial", "valor_pico_max",
                "num_movs", "lat_inicio_mayor_ms", "lat_pico_mayor_ms", "delta_t_pico"
            ]})
        for i, col in enumerate(grouping_cols):
            agg_metrics[col] = name[i]
        agg_list.append(agg_metrics)
    aggregated_df = pd.DataFrame(agg_list)
    aggregated_df = aggregated_df.loc[:, ~aggregated_df.columns.duplicated()]

    aggregated_df["Sitio"] = (
        aggregated_df["Dia experimental"].astype(str) + "_" +
        aggregated_df["Coordenada_x"].astype(str)     + "_" +
        aggregated_df["Coordenada_y"].astype(str)
    )

    # 🔽  NUEVO: enriquecemos antes de devolver
    aggregated_df = enrich_aggregated(aggregated_df)
    return aggregated_df
# ──────────────────────────────────────────────────────────────
# NUEVO helper  ➜   lo pones justo después de aggregate_…()
# ──────────────────────────────────────────────────────────────
def enrich_aggregated(df):
    """
    • Calcula Forma_del_Pulso y Duracion_ms
    • Aplica las TRANSFORMS y deja las columnas *_tf
    • Garantiza que 'Sitio' está bien formado (por si llega de otra fuente)
    """
    # ----- estímulo → forma + duración ------------------------
    df['Forma_del_Pulso'] = (df['Estímulo']
                             .str.split(',', 1).str[0]
                             .str.lower().str.strip())
    df['Duracion_ms'] = df['Estímulo'].apply(extract_duration)

    # ----- transforms ----------------------------------------
    for m, f in TRANSFORMS.items():
        if m in df.columns and m + '_tf' not in df.columns:
            df[m + '_tf'] = f(df[m])

    # ----- sitio (día_x_y)  ----------------------------------
    if 'Sitio' not in df.columns:
        df['Sitio'] = (df['Dia experimental'].astype(str) + '_' +
                       df['Coordenada_x'].astype(str)     + '_' +
                       df['Coordenada_y'].astype(str))

    # elimina posibles duplicados de columnas
    return df.loc[:, ~df.columns.duplicated()]

# -------------------------------
# ETAPA 2: Pruebas de hipótesis (ANOVA, Post-hoc, Friedman)
# (Se mantiene la estructura original)
# -------------------------------
def calc_partial_eta_sq(anova_table, factor_row='C(Q("Forma del Pulso"))', resid_row='Residual'):
    try:
        ss_factor = anova_table.loc[factor_row, 'sum_sq']
        ss_resid = anova_table.loc[resid_row, 'sum_sq']
        return ss_factor / (ss_factor + ss_resid)
    except:
        return np.nan

def build_significance_matrix_from_arrays(factor_levels, tukey_df):
    n = len(factor_levels)
    pval_matrix = pd.DataFrame(np.ones((n, n)), index=factor_levels, columns=factor_levels)
    for row in tukey_df.itertuples():
        g1 = getattr(row, 'group1')
        g2 = getattr(row, 'group2')
        p_adj = getattr(row, 'p_adj')
        if g1 in pval_matrix.index and g2 in pval_matrix.columns:
            pval_matrix.loc[g1, g2] = p_adj
            pval_matrix.loc[g2, g1] = p_adj
    return pval_matrix

def do_posthoc_tests(df, metric, factor_name):
    df, m_used = _maybe_transform(df.copy(), metric)      # 👈
    df_clean = df.dropna(subset=[m_used, factor_name])
    if df_clean[factor_name].nunique() < 2:
        return None, None
    try:
        tukey_res = pairwise_tukeyhsd(endog=df_clean[m_used].values,
                                  groups=df_clean[factor_name].values,
                                  alpha=0.05)
        tk_df = pd.DataFrame(data=tukey_res._results_table.data[1:], 
                             columns=tukey_res._results_table.data[0])
        tk_df = tk_df.rename(columns={'p-adj': 'p_adj'})
        factor_levels = sorted(df_clean[factor_name].unique())
        pval_matrix = build_significance_matrix_from_arrays(factor_levels, tk_df)
        return tk_df, pval_matrix
    except Exception as e:
        logging.warning(f"Error en post-hoc para {factor_name}, {metric}: {e}")
        return None, None
    

ASSUMPTION_RESULTS = []         #  <<-- NUEVO  (renombramos para diferenciar)

def check_assumptions(model, data_metric, group_var, prefix):
    # ── intentar obtener los residuos ──────────────────────────────
    try:
        resid = model.resid                      # MixedLMResults
    except ValueError:                           # cov(RE) singular
        #   y  = Xβ               (ignora los efectos aleatorios)
        y  = model.model.endog
        X  = model.model.exog
        β  = model.fe_params
        resid = y - X @ β

    # ---------- Shapiro y Levene como antes ------------------------
    sw_stat, sw_p = shapiro(resid)

    levels  = data_metric[group_var[0]].dropna().unique()
    samples = [data_metric.loc[data_metric[group_var[0]] == lev,
                               model.model.endog_names]
               for lev in levels
               if len(data_metric.loc[data_metric[group_var[0]] == lev]) >= 2]
    lev_p = levene(*samples)[1] if len(samples) >= 2 else np.nan

    ASSUMPTION_RESULTS.append(
        {"Test_ID": prefix, "Shapiro_p": sw_p, "Levene_p": lev_p}
    )


# ---------- wrapper que guarda ----------
def do_significance_tests_aggregated(
                aggregated_df: pd.DataFrame,
                *,
               output_dir: Optional[str] = None
        ) -> pd.DataFrame:
    anova_df = _calc_anova_table(aggregated_df)
    if output_dir:
        csv = os.path.join(output_dir, 'anova_results_aggregated.csv')
        anova_df.to_csv(csv, index=False)
        print(f"Resultados ANOVA guardados en: {csv}")
    return anova_df

def format_stat(label, p, F):
    if pd.isna(p) or pd.isna(F):
        return f"{label}: NA"
    else:
        prefix = "*" if p < 0.05 else ""
        return f"{label}: {prefix}p={p:.3g} (F={F:.2f})"



def get_tukey_pvals_for_stimulus(agg_df, stim, metric):
    df_sub      = agg_df[agg_df['Estímulo'] == stim].copy()
    df_sub, m_used = _maybe_transform(df_sub, metric)
    if df_sub['MovementType'].nunique() < 2:
        return None
    tukey_res = pairwise_tukeyhsd(endog=df_sub[m_used].values,
                                  groups=df_sub['MovementType'].values,
                                  alpha=0.05)
    tk_df = pd.DataFrame(data=tukey_res._results_table.data[1:], 
                         columns=tukey_res._results_table.data[0])
    tk_df = tk_df.rename(columns={'p-adj': 'p_adj'})
    groups = sorted(df_sub['MovementType'].unique())
    pval_matrix = build_significance_matrix_from_arrays(groups, tk_df)
    return pval_matrix

def add_significance_brackets(ax, pairs, p_values, box_positions, y_offset=0.02, line_height=0.03, font_size=10):
    y_lim = ax.get_ylim()
    h_base = y_lim[1] + (y_lim[1] - y_lim[0]) * y_offset
    step = 0
    for (lv1, lv2) in pairs:
        key1 = (lv1, lv2)
        key2 = (lv2, lv1)
        if key1 in p_values:
            pval = p_values[key1]
        elif key2 in p_values:
            pval = p_values[key2]
        else:
            continue
        x1 = box_positions[lv1]
        x2 = box_positions[lv2]
        if x1 > x2:
            x1, x2 = x2, x1
        h = h_base + step * (y_lim[1] - y_lim[0]) * line_height
        step += 1
        ax.plot([x1, x1, x2, x2],
                [h, h+0.001, h+0.001, h],
                lw=1.5, c='k')
        p_text = f"* p={pval:.3g}" if pval < 0.05 else f"p={pval:.3g}"
        ax.text((x1+x2)*0.5, h+0.001, p_text, ha='center', va='bottom', fontsize=font_size)

def format_stats_rows(model_stats, etiqueta):
    p_vals = []
    F_vals = []
    for abbr, key in [('for', 'forma'), ('dur', 'duracion'), ('int', 'interaccion')]:
        p = model_stats.get(f"{key}_p", np.nan)
        F = model_stats.get(f"{key}_F", np.nan)
        p_str = f"{abbr}: {'*' if not pd.isna(p) and p < 0.05 else ''}p={p:.3g}" if not pd.isna(p) else f"{abbr}: NA"
        F_str = f"{abbr}: F={F:.2f}" if not pd.isna(F) else f"{abbr}: NA"
        p_vals.append(p_str)
        F_vals.append(F_str)
    return f"{etiqueta} (p): " + ", ".join(p_vals), f"{etiqueta} (F): " + ", ".join(F_vals)

# Definición de las seis métricas seleccionadas
selected_metrics = [
    'lat_inicio_ms',       # Latencia al inicio
    'lat_pico_mayor_ms',   # Latencia al pico mayor
    'valor_pico_max',      # Amplitud del pico máximo
    'dur_total_ms',        # Duración total
    'delta_t_pico',    # Diferencia entre primer y pico mayor
    'num_movs'             # Número de movimientos
]
# Nuevo diccionario para etiquetas de unidades en el eje Y
yaxis_units = {
    "lat_inicio_ms": "ms",
    "lat_primer_pico_ms": "ms",
    "lat_pico_ultimo_ms": "ms",
    "lat_inicio_mayor_ms": "ms",
    "lat_pico_mayor_ms": "ms",
    "dur_total_ms": "ms",
    "valor_pico_inicial": "px/s",
    "valor_pico_max": "px/s",
    "delta_t_pico": "ms",
    "num_movs": ""  # En número de movimientos, no es necesaria unidad
}

def _has_variation(df, col):
    # al menos 2 niveles distintos Y cada nivel con ≥2 observaciones
    vc = df[col].value_counts()
    return (vc.index.size >= 2) and (vc.min() >= 2)

def compute_lmm_stats(df, metric):
    df = df[df["MovementType"] == "Gaussian-based"].copy()

    # --- chequeos básicos de variación ---------------------------
    if df["Sitio"].nunique() < 2 \
       or not _has_variation(df, "Forma_del_Pulso") \
       or not _has_variation(df, "Duracion_ms"):
        return dict.fromkeys(["forma_p","duracion_p","interaccion_p"], np.nan)

    df, y = _maybe_transform(df, metric)

    # ---------- intento 1: modelo completo -----------------------
    try:
        md  = smf.mixedlm(f"{y} ~ C(Forma_del_Pulso)*Duracion_ms",
                          data=df, groups=df["Sitio"])
        res = md.fit(method="lbfgs", reml=False)
    except (np.linalg.LinAlgError, ValueError):
        # ---------- intento 2: sólo interacción significativa -----
        try:
            md = smf.mixedlm(
                f"{y} ~ C(Forma_del_Pulso)*C(Duracion_ms)",
                data=df, groups=df["Sitio"]
            )

            res = md.fit(method="lbfgs", reml=False)
        except Exception:
            return dict.fromkeys(["forma_p","duracion_p","interaccion_p"], np.nan)

    prefix = (f"{df['Dia experimental'].iloc[0] if 'Dia experimental' in df else 'GLOBAL'}_"
                  f"{metric}")
    check_assumptions(
            model       = res,
            data_metric = df[[y, 'Forma_del_Pulso']].dropna(),
            group_var   = ['Forma_del_Pulso'],
            prefix      = prefix
        )
    
    pvals = res.pvalues
    return {
        "forma_p":        pvals.filter(like="C(Forma_del_Pulso)").min(),
        "duracion_p":     pvals.filter(like="C(Duracion_ms)").min(),
        "interaccion_p":  pvals.filter(like=":").min()            # NaN si no hubo interacción
    }


def stars(p):
    if   p < 0.001: return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "NS"

# ─────────────────────────────────────────────────────────────────────────────
# NUEVA VARIABLE GLOBAL  –  lista donde iremos apilando los p‑values
# ─────────────────────────────────────────────────────────────────────────────
SUMMARY_PVALS = []        # ← se llena dentro de plot_summary_by_filters
SUMMARY_SAMPLE_SIZES = []

# -------------------------------
# Funciones de graficación de resumen
# -------------------------------
def plot_summary_by_filters(aggregated_df, output_dir, day=None, coord_x=None, coord_y=None, body_part=None, 
                              title_prefix="Global_Summary", model_filter=None):
    """
    Genera boxplots de las 6 métricas seleccionadas agrupados por 'Estímulo' usando nombres descriptivos.
    Construye el título del gráfico de forma automática a partir de la información de agrupación, en el formato:
       "Resumen de métricas modelo [Modelo] Coordenada: ([coord_x], [coord_y]) Fecha: [day] [body_part]"
    Ejemplo: "Resumen de métricas modelo Gaussiano Coordenada: (6, 3) Fecha: 28/05 Braquiradial"
    NOTA: El nombre del archivo se mantiene igual que en versiones anteriores (usando title_prefix).
    """
    df = prep_for_anova(aggregated_df, day=day, coord_x=coord_x, coord_y=coord_y, body_part=body_part)

    # Si no se pasan los parámetros, se infieren a partir del DataFrame (si hay un único valor en la columna)
    if day is None:
        unique_days = df["Dia experimental"].dropna().unique()
        day = unique_days[0] if len(unique_days) == 1 else None
    if coord_x is None:
        unique_coord_x = df["Coordenada_x"].dropna().unique()
        coord_x = unique_coord_x[0] if len(unique_coord_x) == 1 else None
    if coord_y is None:
        unique_coord_y = df["Coordenada_y"].dropna().unique()
        coord_y = unique_coord_y[0] if len(unique_coord_y) == 1 else None
    if body_part is None:
        unique_bp = df["body_part"].dropna().unique()
        body_part = unique_bp[0] if len(unique_bp) == 1 else None

    # Normalizamos las columnas necesarias
    df['Forma_del_Pulso'] = df['Estímulo'].apply(lambda s: s.split(',')[0].strip().lower() if isinstance(s, str) and ',' in s else np.nan)
    # desired_forms = ["rectangular", "rombo", "triple rombo", "rampa ascendente"]
    df = df[df["Forma_del_Pulso"].isin(desired_forms)]
    df['Forma_del_Pulso'] = pd.Categorical(df['Forma_del_Pulso'], categories=desired_forms, ordered=True)
    df['Duracion_ms'] = df['Estímulo'].apply(extract_duration)
    df = df.loc[:, ~df.columns.duplicated()]

    # Construir el título usando la información de agrupación inferida
    title_parts = ["Resumen de métricas"]
    # Definir el modelo
    if model_filter is not None and len(model_filter) == 1:
        if model_filter[0] == "Gaussian-based":
            model_text = "modelo Gaussiano"
        elif model_filter[0] == "Threshold-based":
            model_text = "modelo Umbral"
        else:
            model_text = "modelo " + model_filter[0]
    else:
        model_text = title_prefix  # cuando no se especifica, se usa el title_prefix
    title_parts.append(model_text)
    # Agregar coordenadas si se tienen ambos valores
    if coord_x is not None and coord_y is not None:
        title_parts.append(f"Coordenada: ({coord_x}, {coord_y})")
    # Agregar fecha si hay un único valor
    if day is not None:
        title_parts.append(f"Fecha: {day}")
    # Agregar body_part si existe
    if body_part is not None:
        title_parts.append(body_part)
    final_title = " ".join(title_parts)

    # (Se mantiene el nombre del archivo usando el title_prefix sin modificar)
    file_title = sanitize_filename(title_prefix.replace(' ', '_'))

    # Continuar con la construcción de los gráficos (se usa la misma lógica para boxplots, límites de ejes, etc.)
    metrics_order = selected_metrics  
    ordered_stimuli = df.sort_values(["Forma_del_Pulso", "Duracion_ms"])["Estímulo"].unique().tolist()

    latency_metrics = ['lat_inicio_ms', 'lat_pico_mayor_ms']
    peak_metrics = ['valor_pico_max']
    # global_latency_max = df[latency_metrics].max().max() if not df[latency_metrics].empty else None
    # global_peak_max = df[peak_metrics].quantile(0.95).max() if not df[peak_metrics].empty else None

    if model_filter is not None and model_filter == ["Gaussian-based"]:
        box_width = 0.6 / 0.5  # cajas más anchas
        median_line_color = 'silver'  # mediana en plateado
        median_line_width = 3.0
    else:
        box_width = 0.6 / 2.2
        median_line_color = 'black'
        median_line_width = 4.0

    # 1) inicializamos el diccionario de límites
    global_ylim = {}

    # 2) para cada métrica de latencia usamos percentil 95 + 10 % de margen
    for m in latency_metrics:
        vals = df[m].dropna()
        if len(vals):
            p95 = np.percentile(vals, 95)
            global_ylim[m] = (0, p95 * 1.10)
        else:
            global_ylim[m] = (0, 1)

    """
    # 3) idem para amplitud de pico
    for m in peak_metrics:
        vals = df[m].dropna()
        if len(vals):
            p95 = np.percentile(vals, 95)
            global_ylim[m] = (0, p95 * 1.10)
        else:
            global_ylim[m] = (0, 1)
    """
    
    # ────────────────────────────────────
    # límites unificados
    # ────────────────────────────────────
    # para latencias todas con el mismo máximo
    shared_latency_max = max(global_ylim[m][1] for m in latency_metrics)
    shared_latency_ylim = (0, shared_latency_max)
    # para amplitud de pico: calculamos p95 de todas las métricas de pico SIN margen
    raw_peak_max = max(
        np.percentile(df[m].dropna(), 95)
        for m in peak_metrics
        if len(df[m].dropna()) > 0
    )
    # aplicamos un 10% de margen
    shared_peak_ylim = (0, raw_peak_max * 0.8)
    
    tick_settings = {}
    # latencias + dur_total_ms + delta_t_pico cada 500ms (major)/250ms (minor)
    for m in latency_metrics + ['dur_total_ms', 'delta_t_pico']:
        tick_settings[m] = (MultipleLocator(500), MultipleLocator(250))
    # pico máximo: cada 100px/s (major)/50px/s (minor)
    for m in peak_metrics:
        tick_settings[m] = (MultipleLocator(100), MultipleLocator(50))

    n_cols = math.ceil(len(metrics_order) / 2)
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 5, 2 * 5), squeeze=False)
    positions_by_stim = {}

    # Se crea cada boxplot para las métricas seleccionadas (código no modificado en esta parte)
    for idx, metric in enumerate(metrics_order):
        ax = axs[idx // n_cols, idx % n_cols]
        boxplot_data = []
        x_positions = []
        group_centers = []
        labels = []
        current_pos = 0
        for stim in ordered_stimuli:
            df_stim = df[df['Estímulo'] == stim]
            if df_stim.empty:
                continue
            model_positions = {}
            for mtype in ['Threshold-based', 'Gaussian-based', 'MinimumJerk']:
                if model_filter is not None and mtype not in model_filter:
                    continue
                n_trials = len(df[(df['Estímulo']==stim) & (df['MovementType']==mtype)])
                SUMMARY_SAMPLE_SIZES.append({
                    'Day':        day or 'GLOBAL',
                    'Coord_x':    coord_x,
                    'Coord_y':    coord_y,
                    'Body_part':  body_part or 'ALL',
                    'Stimulus':   stim,
                    'Model':      mtype,
                    'n_trials':   n_trials
                })
                data = df_stim[df_stim['MovementType'] == mtype][metric].dropna().values
                if len(data) == 0:
                    continue
                boxplot_data.append(data)
                if mtype == 'Threshold-based':
                    pos = current_pos
                elif mtype == 'Gaussian-based':
                    pos = current_pos + 0.5
                else:
                    pos = current_pos + 0.75
                x_positions.append(pos)
                model_positions[mtype] = pos
            if model_positions:
                center = np.mean(list(model_positions.values()))
                group_centers.append(center)
                labels.append(stim)
                positions_by_stim[stim] = model_positions
                current_pos = max(x_positions) + 1.5
            else:
                current_pos += 0.6 * 3 + 1.5

        if not boxplot_data:
            ax.text(0.5, 0.5, "Sin datos", ha='center', va='center')
            continue

        bp_obj = ax.boxplot(
            boxplot_data,
            positions=x_positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            boxprops     = dict(linewidth=0),   # ← sin borde en las cajas
            whiskerprops = dict(linewidth=0),   # ← sin bigotes
            capprops     = dict(linewidth=0),   # ← sin “capas” en los extremos
            medianprops  = {'color': median_line_color, 'linewidth': median_line_width}
        )

        for element in ['whiskers', 'caps']:
            for line in bp_obj[element]:
                line.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for pos, patch in zip(x_positions, bp_obj['boxes']):
            diffs = {stim: abs((positions_by_stim[stim]['Threshold-based'] if 'Threshold-based' in positions_by_stim[stim]
                                 else list(positions_by_stim[stim].values())[0]) - pos)
                     for stim in positions_by_stim}
            stim_key = min(diffs, key=diffs.get)
            shape = stim_key.split(',')[0].strip().lower()
            try:
                float(stim_key.split(',')[1].replace(' ms','').strip())
            except:
                pass
            cmap = shape_colors.get(shape, 'gray')
            patch.set_facecolor(cmap)
            patch.set_edgecolor('black')

        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel("")
        ax.set_title(metric_labels.get(metric, metric), fontsize=18)
        ax.set_ylabel(yaxis_units.get(metric, ""), fontsize=10)

        # ── fijar rango unificado ─────────────────────────────────────────
        if metric in latency_metrics:
            ax.set_ylim(*shared_latency_ylim)
        elif metric in peak_metrics:
            ax.set_ylim(*shared_peak_ylim)
        else:
            vals = np.concatenate(boxplot_data)
            p95 = np.percentile(vals, 95)
            ax.set_ylim(0, p95 * 1.10)

        # ── aplicar ticks según lo definido arriba ────────────────────────
        major, minor = tick_settings.get(metric, (None, None))
        if major and minor:
            ax.yaxis.set_major_locator(major)
            ax.yaxis.set_minor_locator(minor)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='y', which='major', length=6)
            ax.tick_params(axis='y', which='minor', length=3)



        # Se anota una línea con los p-valores del modelo Gaussiano (como ejemplo)
        # Se anota una caja en la esquina superior izquierda con los p-valores del modelo Gaussiano (dividido en tres líneas)
        # ── filtro para tests: sólo las formas clave ───────────
        # ANOVA factorial sólo rectangular vs rombo a 500/1000 ms
        df_tests = df[
            (df.MovementType == "Gaussian-based") &
            (df["Forma_del_Pulso"].isin(key_forms)) &
            (df["Duracion_ms"].astype(int).isin([500, 1000]))
        ]
        stats = run_factorial_anova(df_tests, metric)
        # Bonferroni para los 3 efectos
        # extraemos raw p’s
        # Asegurarnos de tener floats puros
        # Bonferroni para los 3 efectos
        raw = [ float(stats.get(k, np.nan)) for k in ('forma_p','duracion_p','interaccion_p') ]
        _, p_adj, _, _ = multipletests(raw, alpha=0.05, method='bonferroni')
        adj = dict(zip(['forma_p','duracion_p','interaccion_p'], p_adj))

        # Definimos las etiquetas limpias
        labels = ['Forma', 'Duración', 'Forma x Duración']

        # Creamos lista de (texto, es_significativo)
        texts = []
        for key, lbl in zip(['forma_p','duracion_p','interaccion_p'], labels):
            p = adj[key]
            s = stars(p)
            texts.append((f"{lbl}: {s}", s != "NS"))

        # Dibujamos
        y = 0.99; dy = 0.10
        for txt, sig in texts:
            ax.text(0.01, y, txt,
                    transform=ax.transAxes,
                    fontsize=12, va='top', ha='left',
                    color='red' if sig else 'black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))
            y -= dy



        # Opcional: guardarlo en SUMMARY_PVALS si lo necesitas
        SUMMARY_PVALS.append({
            "Dia": day or "GLOBAL", "Coord_x": coord_x, "Coord_y": coord_y,
            "Body_part": body_part, "Metric": metric,
            **dict(zip(
            ['forma_p','duracion_p','interaccion_p'],
            p_adj
        ))
        })


        # 1) Llamas a Tukey para ESTE estímulo y ESTA métrica:
        tk_df, pval_matrix = do_posthoc_tests(
            df[df['Estímulo']==stim],  # sub-DataFrame de ESTE estímulo
            metric,                    # la métrica actual
            'MovementType'             # factor que quieres comparar
        )

        # 2) Si tienes resultados, anotas:
        if pval_matrix is not None:
            # todas las parejas de niveles de MovementType
            for pair in itertools.combinations(positions_by_stim[stim].keys(), 2):
                add_significance_brackets(
                    ax,
                    [pair],                        # solo esta pareja
                    pval_matrix,                  # matriz de p-valores
                    positions_by_stim[stim],      # posiciones x de cada nivel
                    y_offset=0.03, line_height=0, font_size=6
                )

    # Establecer el título final del gráfico usando final_title
    fig.suptitle(final_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Se conserva el nombre del archivo original (usando title_prefix sin modificaciones)
    fname = sanitize_filename(title_prefix.replace(' ', '_'))
    out_path = os.path.join(output_dir, fname + ".png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] {final_title} guardado en: {out_path}")




# -------------------------------
# Funciones de graficación 3D (se mantienen)
# -------------------------------
def plot_3d_gaussian_boxplots_by_bodypart(aggregated_df, output_dir, day=None, coord_x=None, coord_y=None):
    import plotly.graph_objects as go
    from scipy.interpolate import griddata
    df = aggregated_df[aggregated_df['MovementType'] == 'Gaussian-based'].copy()
    if day is not None:
        df = df[df["Dia experimental"] == day]
    if coord_x is not None:
        df = df[df["Coordenada_x"] == coord_x]
    if coord_y is not None:
        df = df[df["Coordenada_y"] == coord_y]
    if df.empty:
        print("No hay datos Gaussian-based para el grupo seleccionado.")
        return
    metrics = ["lat_inicio_ms", "lat_primer_pico_ms", "lat_pico_ultimo_ms",
               "dur_total_ms", "valor_pico_inicial", "valor_pico_max",
               "num_movs", "lat_inicio_mayor_ms", "lat_pico_mayor_ms", "delta_t_pico"]
    df['Forma'] = df['Estímulo'].apply(lambda s: s.split(',')[0].strip().lower() if isinstance(s, str) and ',' in s else 'unknown')
    df['Duración (ms)'] = df['Estímulo'].apply(extract_duration)
    shape_order = ["rectangular", "rombo", "rampa descendente", "triple rombo", "rampa ascendente"]
    df["Forma"] = pd.Categorical(df["Forma"], categories=shape_order, ordered=True)
    ordered_stimuli = df.sort_values(["Forma", "Duración (ms)"])["Estímulo"].unique().tolist()
    ordered_bodyparts = body_parts
    stim_to_x = {stim: i for i, stim in enumerate(ordered_stimuli)}
    bp_to_y = {bp: i for i, bp in enumerate(ordered_bodyparts)}
    stats_list = []
    for stim, bp, forma in df[['Estímulo', 'body_part', 'Forma']].drop_duplicates().values:
        sub = df[(df['Estímulo'] == stim) & (df['body_part'] == bp)]
        if sub.empty:
            continue
        stat_entry = {'Estímulo': stim, 'body_part': bp, 'Forma': forma}
        for metric in metrics:
            vals = sub[metric].dropna().values
            if len(vals) == 0:
                continue
            stat_entry[f"{metric}_vmin"] = np.min(vals)
            stat_entry[f"{metric}_q1"] = np.percentile(vals, 25)
            stat_entry[f"{metric}_med"] = np.median(vals)
            stat_entry[f"{metric}_q3"] = np.percentile(vals, 75)
            stat_entry[f"{metric}_vmax"] = np.max(vals)
        stats_list.append(stat_entry)
    stats_df = pd.DataFrame(stats_list)
    if stats_df.empty:
        print("No hay datos para construir la interpolación de las medianas.")
        return
    for metric in metrics:
        metric_label = metric_labels.get(metric, metric)
        subdf = stats_df.dropna(subset=[f"{metric}_med", f"{metric}_vmin", f"{metric}_vmax"])
        if subdf.empty:
            continue
        x_vals = subdf['Estímulo'].map(stim_to_x).values
        y_vals = subdf['body_part'].map(bp_to_y).values
        med_vals = subdf[f"{metric}_med"].values
        q1_vals = subdf[f"{metric}_q1"].values
        q3_vals = subdf[f"{metric}_q3"].values
        cmin = np.percentile(med_vals, 5)
        cmax = np.percentile(med_vals, 95)
        grid_x, grid_y = np.mgrid[min(x_vals):max(x_vals):100j, min(y_vals):max(y_vals):100j]
        grid_med = griddata((x_vals, y_vals), med_vals, (grid_x, grid_y), method='cubic')
        surface_med = go.Surface(
            x=grid_x, y=grid_y, z=grid_med,
            colorscale='Viridis', opacity=0.6, cmin=cmin, cmax=cmax,
            showscale=True, name='Mediana'
        )
        global_zmin = np.min(q1_vals)
        global_zmax = np.max(q3_vals) * 1.05
        traces = []
        for idx, row in subdf.iterrows():
            x = stim_to_x[row['Estímulo']]
            y = bp_to_y.get(row['body_part'], 0)
            width = 0.4
            depth = 0.4
            vertices_center = np.array([
                [x - width/2, y - depth/2, row[f"{metric}_q1"]],
                [x + width/2, y - depth/2, row[f"{metric}_q1"]],
                [x + width/2, y + depth/2, row[f"{metric}_q1"]],
                [x - width/2, y + depth/2, row[f"{metric}_q1"]],
                [x - width/2, y - depth/2, row[f"{metric}_q3"]],
                [x + width/2, y - depth/2, row[f"{metric}_q3"]],
                [x + width/2, y + depth/2, row[f"{metric}_q3"]],
                [x - width/2, y + depth/2, row[f"{metric}_q3"]]
            ])
            center_box_trace = go.Mesh3d(
                x=vertices_center[:, 0],
                y=vertices_center[:, 1],
                z=vertices_center[:, 2],
                opacity=0.8,
                color=shape_colors.get(row['Forma'], 'gray'),
                i=[0, 0, 0, 4, 4, 4],
                j=[1, 2, 3, 5, 6, 7],
                k=[2, 3, 1, 6, 7, 5],
                showscale=False,
                name=f"Center {row['Estímulo']} - {row['body_part']}"
            )
            traces.append(center_box_trace)
            center_line = go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[row[f"{metric}_q1"], row[f"{metric}_q3"]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            )
            traces.append(center_line)
            median_trace = go.Scatter3d(
                x=[x - width/2, x + width/2],
                y=[y, y],
                z=[row[f"{metric}_med"], row[f"{metric}_med"]],
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False
            )
            traces.append(median_trace)
        layout = go.Layout(
            title=f"3D Gaussian-based Boxes for {metric_label} (Day: {day}, Coord: {coord_x}, {coord_y})",
            scene=dict(
                xaxis=dict(
                    title=dict(text="", font=dict(size=14)),
                    tickvals=list(stim_to_x.values()),
                    ticktext=list(stim_to_x.keys()),
                    range=[min(stim_to_x.values()) - 0.5, max(stim_to_x.values()) + 0.5]
                ),
                yaxis=dict(
                    title="",
                    tickvals=list(bp_to_y.values()),
                    ticktext=list(bp_to_y.keys())
                ),
                zaxis=dict(
                    title=metric_label,
                    range=[min(global_zmin - 0.05 * abs(global_zmax - global_zmin), -0.05 * global_zmax), global_zmax]
                )
            )
        )
        fig = go.Figure(data=[surface_med] + traces, layout=layout)
        filename = sanitize_filename(f"3D_Gaussian_Boxplot_{metric_label}_Day_{day}_Coord_{coord_x}_{coord_y}.html")
        out_path = os.path.join(output_dir, filename)
        fig.write_html(out_path)
        print(f"3D Gaussian Boxplot for {metric_label} saved at: {out_path}")

# -------------------------------
# Funciones de agrupación: llamadas para distintos grupos
# -------------------------------

def plot_summary_by_day_coord(aggregated_df, output_dir):
    for (dia, coord_x, coord_y), df_sub in aggregated_df.groupby(['Dia experimental', 'Coordenada_x', 'Coordenada_y']):
        title = f"Summary_{dia}_Coord_{coord_x}_{coord_y}"
        plot_summary_by_filters(df_sub, output_dir, title_prefix=title, model_filter=["Gaussian-based"])
def reset_summary_pvals():
    global SUMMARY_PVALS
    SUMMARY_PVALS = []

def plot_global_summary(aggregated_df, output_dir):
    reset_summary_pvals()
    title = "Global_Summary"
    plot_summary_by_filters(aggregated_df, output_dir, title_prefix=title, model_filter=["Gaussian-based"])

def plot_summary_by_bodypart(aggregated_df, output_dir):
    for (dia, bp), df_sub in aggregated_df.groupby(['Dia experimental', 'body_part']):
        title = f"Summary_{dia}_{bp}"
        plot_summary_by_filters(df_sub, output_dir, title_prefix=title, model_filter=["Gaussian-based"])

def plot_model_only_validation(aggregated_df, group_by, metrics=None):
    """
    Igual que antes, pero compara únicamente Gaussian‑based vs MinimumJerk.
    """
    if metrics is None:
        metrics = selected_metrics

    for name, grp in aggregated_df.groupby(group_by):
        # Filtrar solo los dos modelos que nos interesan
        sub_grp = grp[grp['MovementType'].isin(['Gaussian-based','MinimumJerk'])]

        resultados = []
        for m in metrics:
            sub = sub_grp.dropna(subset=[m, 'MovementType'])
            # comprobamos que haya al menos una observación en cada grupo
            tipos = sub['MovementType'].value_counts()
            if len(sub) < 3 or any(tipos.get(t,0) < 2 for t in ['Gaussian-based','MinimumJerk']):
                p = np.nan; F = np.nan
            else:
                model = ols(f"{m} ~ C(MovementType)", data=sub).fit()
                an = anova_lm(model, typ=2)
                p = an.loc['C(MovementType)', 'PR(>F)']
                F = an.loc['C(MovementType)', 'F']
            resultados.append({'Metric': m, 'p': p, 'F': F})

        df_r = pd.DataFrame(resultados)
        df_r['-log10(p)'] = -np.log10(df_r['p'].fillna(1))
        df_r['sig'] = df_r['p'] < 0.05
        df_r['Label'] = df_r['sig'].map({True: 'S', False: 'NS'})

        # Gráfico
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(
            [metric_labels[m] for m in df_r['Metric']],
            df_r['-log10(p)'],
            edgecolor='black'
        )
        ax.axhline(-np.log10(0.05), ls='--', color='red')
        for rect, label in zip(bars, df_r['Label']):
            ax.text(
                rect.get_x() + rect.get_width()/2,
                rect.get_height() + 0.1,
                label,
                ha='center', va='bottom'
            )

        ax.set_ylabel('-log10(p)')
        # al principio de plot_model_only_validation:
        label = ", ".join(group_by)  # p.ej. "Dia experimental, Coordenada_x, Coordenada_y"
        
        ax.set_title(f"Comparación Gaussian vs Minjerk – {label} = {name}")

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        grp_name = name if not isinstance(name, tuple) else "_".join(map(str,name))
        fname = sanitize_filename(f"model_compare_gm_{'_'.join(group_by)}_{grp_name}.png")
        fig.savefig(os.path.join(output_comparisons_dir, fname), dpi=150)
        plt.close()
        print(f"[Model compare G vs M] guardado en: {fname}")

def plot_model_compare_gauss_min(aggregated_df, by='global'):
    groups = []
    df = aggregated_df.copy()
    if by=='global':
        groups = [('Global','Global', df)]
    else:
        for dia, x, y in df[['Dia experimental','Coordenada_x','Coordenada_y']].drop_duplicates().values:
            sub = df[(df['Dia experimental']==dia)&
                     (df['Coordenada_x']==x)&
                     (df['Coordenada_y']==y)]
            name = f"{dia}_{x}_{y}"
            groups.append((name, name, sub))

    for name, title, sub in groups:
        regs = []
        sub = sub[sub['MovementType'].isin(['Gaussian-based','MinimumJerk'])]
        for metric in selected_metrics:
            clean = sub.dropna(subset=[metric,'MovementType'])
            if clean['MovementType'].value_counts().min() < 2:
                p = np.nan; F = np.nan
            else:
                m = ols(f"{metric} ~ C(MovementType)", data=clean).fit()
                an = anova_lm(m, typ=2)
                p = an.loc['C(MovementType)','PR(>F)']
                F = an.loc['C(MovementType)','F']
            regs.append({'Metric':metric_labels[metric], '-log10(p)': -np.log10(p if p>0 else 1), 'sig': p<0.05})

        cmp_df = pd.DataFrame(regs).set_index('Metric')
        plt.figure(figsize=(8,5))
        ax = cmp_df['-log10(p)'].plot(kind='bar', edgecolor='black')
        ax.axhline(-np.log10(0.05), ls='--', color='red')

        # anotamos “S”/“NS” usando las barras, no el índice (strings)
        for bar, is_sig in zip(ax.patches, cmp_df['sig']):
            x = bar.get_x() + bar.get_width()/2
            y = bar.get_height()
            ax.text(
                x, 
                y + 0.05,
                'S' if is_sig else 'NS',
                ha='center', va='bottom'
            )

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()


        fname = sanitize_filename(f"model_compare_gauss_min_{by}_{name}.png")
        plt.savefig(os.path.join(output_comparisons_dir, fname), dpi=150)
        plt.close()
        print(f"[ModelCmp G-M] guardado en: {fname}")


def plot_model_validation_full_by_group(aggregated_df, output_dir, group_by, metrics=None):
    """
    Para cada grupo, calcula p-valor de:
      - ANOVA solo Forma
      - ANOVA solo Duración
      - ANOVA interacción (forma*duración)
    y plotea un bar chart con “S”/“NS”.
    """
    if metrics is None:
        metrics = selected_metrics

    # define los tres factores ahí
    factors = {
        'Forma':             "C(Forma_del_Pulso)",
        'Duración':          "C(Duracion_ms)",
        'Interacción':       "C(Forma_del_Pulso):C(Duracion_ms)"
    }

    for name, grp in aggregated_df.groupby(group_by):
        resultados = []
        for m in metrics:
            sub = grp.dropna(subset=[m,'Forma_del_Pulso','Duracion_ms'])
            for lbl, term in factors.items():
                vars_ = re.findall(r"C\(([^)]+)\)", term)
                tmp = sub.dropna(subset=[m]+vars_)
                if len(tmp)<3 or any(tmp[v].nunique()<2 for v in vars_):
                    p = np.nan
                else:
                    an = anova_lm(ols(f"{m} ~ {term}", data=tmp).fit(), typ=2)
                    p = an.loc[term,'PR(>F)'] if term in an.index else np.nan
                resultados.append({'Metric':m,'Factor':lbl,'p':p})

        df_r = pd.DataFrame(resultados)
        df_r['sig'] = df_r['p']<0.05
        df_r['Label'] = df_r['sig'].map({True:'S',False:'NS'})
        df_r['-log10(p)'] = -np.log10(df_r['p'].fillna(1))

        pivot = (df_r.pivot(index='Metric',columns='Factor',values='-log10(p)')
                   .reindex(index=metrics,columns=factors.keys()))
        pivot.index = [metric_labels[m] for m in pivot.index]

        fig, ax = plt.subplots(figsize=(8,5))
        pivot.plot(kind='bar', ax=ax, color='gray')
        for i,rects in enumerate(ax.containers):
            for j,rect in enumerate(rects):
                label = df_r.iloc[i*len(factors)+j]['Label']
                ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.1,
                        label, ha='center', va='bottom')
        ax.axhline(-np.log10(0.05), color='red', linestyle='--')
        ax.set_ylabel('-log10(p)')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Model Validation – {group_by}={name}")
        plt.tight_layout()

        fname = sanitize_filename(f"model_validation_full_{group_by}_{name}.png")
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
        print("Guardado:", fname)


def generate_descriptive_stats_table(aggregated_df, output_dir):
    """
    Genera una tabla descriptiva (mean, std, median, min, max y count) para cada métrica,
    agrupada por 'Day', 'Body_Part' y 'Movement_Model'. La tabla resultante se guarda en CSV.
    """
    # Aseguramos que las columnas de agrupación estén en el formato adecuado
    if 'Dia experimental' not in aggregated_df.columns:
        aggregated_df['Dia experimental'] = aggregated_df['Estímulo'].apply(lambda s: s.split(',')[0].strip().lower())
    # Usamos los mismos nombres de columnas que en el resumen ANOVA
    descriptive_df = aggregated_df.groupby(['Dia experimental', 'body_part', 'MovementType'])[selected_metrics] \
                                   .agg(['mean', 'std', 'median', 'min', 'max', 'count'])
    # Aplanamos los MultiIndex de columnas
    descriptive_df.columns = ['_'.join(col).strip() for col in descriptive_df.columns.values]
    descriptive_df = descriptive_df.reset_index()
    out_csv = os.path.join(output_dir, 'descriptive_statistics_table.csv')
    descriptive_df.to_csv(out_csv, index=False)
    print(f"Tabla descriptiva de estadísticas guardada en: {out_csv}")
    return descriptive_df

def _row(anova_df, pattern):
    """
    Devuelve el índice cuya versión ‘limpia’ contiene la versión ‘limpia’
    de *pattern*.  Elimina espacios, comas, paréntesis y la palabra 'sum'.
    """
    def clean(s):
        # quita 'sum', espacios, paréntesis, comas y dos puntos
        return re.sub(r'sum|[ (),:]', '', s.lower())

    pat = clean(pattern)
    for r in anova_df.index:
        if pat in clean(r):
            return r

    raise KeyError(f"fila '{pattern}' no hallada en ANOVA\nÍndices = {list(anova_df.index)}")




# ---------- 1. Cálculo único de ANOVA factorial ----------
# ---------- ANOVA factorial único (forma × duración) ----------
import statsmodels.api as sm
def balance_forms_durations(df):
    forms = df['Forma_del_Pulso'].unique()
    dur_por_forma = {
        f: set(df[df['Forma_del_Pulso']==f]['Duracion_ms'].unique())
        for f in forms
    }
    common = set.intersection(*dur_por_forma.values())
    return df[df['Duracion_ms'].isin(common)]
# 1) ANOVA 2-vías SS Tipo III, HC3
import statsmodels.api as sm




def balance_forms_durations(df, forms=('rectangular','rombo')):
    # Duraciones presentes en todas las formas clave
    common = set.intersection(*[
        set(df[df['Forma_del_Pulso']==f]['Duracion_ms'].unique())
        for f in forms
    ])
    return df[df['Duracion_ms'].isin(common)]





def preprocess_estimulo(df):
    df['Forma_del_Pulso'] = df['Estímulo'].apply(
        lambda s: s.split(',')[0].strip().lower() if isinstance(s, str) and ',' in s else np.nan)
    df['Duracion_ms'] = df['Estímulo'].apply(extract_duration)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def plot_heatmap_gauss_from_anova(aggregated_df, output_dir, title):

    # --- claves que identifican el grupo ------------------------------
    days      = aggregated_df['Dia experimental'].dropna().unique()
    day_key   = days[0] if len(days) == 1 else 'GLOBAL'

    coords_x  = aggregated_df['Coordenada_x'].dropna().unique()
    coord_x_k = coords_x[0] if len(coords_x) == 1 else None

    coords_y  = aggregated_df['Coordenada_y'].dropna().unique()
    coord_y_k = coords_y[0] if len(coords_y) == 1 else None

    # --- p‑values ya guardados en los summaries -----------------------
    global SUMMARY_PVALS
    pval_df = pd.DataFrame(SUMMARY_PVALS)

    results = []
    for metric in selected_metrics:

        # ---------- MASK corregido -----------------------------------
        mask = (
            (pval_df['Dia']    == day_key) &
            (pval_df['Metric'] == metric)  &
            (
                (pval_df['Coord_x'].isna() if coord_x_k is None
                 else pval_df['Coord_x'] == coord_x_k)
            ) &
            (
                (pval_df['Coord_y'].isna() if coord_y_k is None
                 else pval_df['Coord_y'] == coord_y_k)
            )
        )
        # --------------------------------------------------------------

        if mask.any():
            # ya están bonferroni-correctos
            row   = pval_df.loc[mask].iloc[0]
            p_f, p_d, p_int = row[['forma_p','duracion_p','interaccion_p']].values

        else:
            # fallback (muy rara vez se ejecutará ya)
            sub = aggregated_df.dropna(subset=[metric,
                                               'Forma_del_Pulso',
                                               'Duracion_ms'])
            if sub['Forma_del_Pulso'].nunique() < 2 or sub['Duracion_ms'].nunique() < 2:
                p_f = p_d = p_int = np.nan
            else:
                # ── filtro para tests: sólo las formas clave ───────────
                sub_tests = sub[sub["Forma_del_Pulso"].isin(key_forms)]
                # recalculamos crudos y aplicamos bonferroni
                stats = run_factorial_anova(sub_tests, metric)
                # Asegurarnos de tener floats puros
                raw = [
                    float(stats.get('forma_p', np.nan)),
                    float(stats.get('duracion_p', np.nan)),
                    float(stats.get('interaccion_p', np.nan))
                ]
                # Ahora sí
                _, p_adj, _, _ = multipletests(raw, alpha=0.05, method='bonferroni')

                p_f, p_d, p_int = p_adj


        results.append({
            'Metric':      metric_labels[metric],
            'Forma':       p_f,
            'Duración':    p_d,
            'Interacción': p_int
        })


    heat_df = pd.DataFrame(results).set_index('Metric')

    # --- anotaciones y dibujo (sin cambios) ---------------------------
        # --- anotaciones con asteriscos en lugar de p=… -----------------
    annot = heat_df.copy().astype(object)
    for m in annot.index:
        for c in annot.columns:
            p = heat_df.loc[m, c]
            if pd.isna(p):
                annot.loc[m, c] = "NA"
            else:
                annot.loc[m, c] = stars(p)


    plt.figure(figsize=(6, 0.6 * len(heat_df)))
    ax = sns.heatmap(
        heat_df, cmap="Greys_r", vmin=0, vmax=0.05,
        annot=annot, fmt="", linewidths=0.5,
        cbar_kws={'label': 'p‑value'}
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Efecto')
    ax.set_ylabel('Métrica')
    ax.set_title(title)
    plt.tight_layout()

    out_path = os.path.join(output_dir, sanitize_filename(title) + '.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Heatmap ANOVA] {title} guardado en: {out_path}")

def run_ttest_simple_by_site(df, metrics=None, output_dir=output_comparisons_dir):
    if metrics is None:
        metrics = selected_metrics.copy()

    # 1) filtramos Gaussian-based
    sub = (df[df.MovementType=='Gaussian-based']
           .assign(
               Forma=lambda d: d['Estímulo'].str.split(',',1).str[0].str.lower().str.strip(),
               Dur_ms=lambda d: d['Estímulo'].apply(extract_duration).round().astype(int)
           ))

    # 2) filtramos hombro/codo/muñeca, rectangular vs rombo a 500 ms
    desired_parts = ['Hombro','Codo','Muneca','Braquiradial','Bicep','Frente']

    mask = (
        (sub['Dur_ms'] == 500) &
        (sub['Forma'].isin(['rectangular','rombo'])) &
        (sub['body_part'].isin(desired_parts))
    )
    filtered = sub[mask]
    # Verificación rápida de que sólo salen esas seis partes
    assert set(filtered['body_part'].unique()) <= set(desired_parts), \
        f"Hay partes fuera de {desired_parts}: {set(filtered['body_part'].unique())}"

    logging.info(f"T-test: {len(filtered)} ensayos tras filtrar formas y bodyparts {desired_parts}")
    sub = sub[mask]

    all_results = []
    for (day, x, y), grp in sub.groupby(['Dia experimental','Coordenada_x','Coordenada_y']):
        site_stats = []
        for m in metrics:
            g1 = grp.loc[grp.Forma=='rectangular', m].dropna()
            g2 = grp.loc[grp.Forma=='rombo',       m].dropna()

            # aplicamos transform si toca
            if m in TRANSFORMS:
                g1 = TRANSFORMS[m](g1)
                g2 = TRANSFORMS[m](g2)

            # test de varianzas y normalidad
            p_lev  = levene(g1, g2).pvalue if len(g1)>1 and len(g2)>1 else np.nan
            p_sw1  = shapiro(g1).pvalue if len(g1)>2 else np.nan
            p_sw2  = shapiro(g2).pvalue if len(g2)>2 else np.nan

            # igualamos tamaños
            n = min(len(g1), len(g2))
            g1, g2 = g1.sample(n, random_state=1), g2.sample(n, random_state=1)

            if (p_sw1 > .05 and p_sw2 > .05) and n>=2:
                stat, p_raw = ttest_ind(g1, g2, equal_var=(p_lev>.05))
            else:
                stat, p_raw = mannwhitneyu(g1, g2)

            site_stats.append({'metric': m, 'p_raw': p_raw})
            all_results.append({
                'day': day, 'coord_x': x, 'coord_y': y,
                'metric': m, 'stat': stat, 'p_raw': p_raw,
                'p_levene': p_lev, 'p_shapiro1': p_sw1, 'p_shapiro2': p_sw2
            })

        # 3) dibujamos el heatmap
        heat_df = pd.DataFrame(site_stats).set_index('metric') \
                     .rename(index=lambda m: metric_labels[m])
        annot   = heat_df['p_raw'].apply(lambda p: stars(p) if pd.notna(p) else 'NA').to_frame()

        plt.figure(figsize=(6, len(metrics)*0.5+1))
        ax = sns.heatmap(
            heat_df, cmap="Greys_r", vmin=0, vmax=0.05,
            annot=annot, fmt='', linewidths=0.5,
            cbar_kws={'label':'p-value'}
        )
        ax.set_xticks([0.5])
        ax.set_xticklabels(['Rectangular vs Rombo (500 ms)'], ha='center')
        ax.set_title(f"Coordenada ({x},{y})")
        ax.set_ylabel('Métrica')
        plt.tight_layout()

        fn = sanitize_filename(f"ttest_rect_vs_rombo_{day}_{x}_{y}.png")
        plt.savefig(os.path.join(output_dir, fn), dpi=150, bbox_inches='tight')
        plt.close()

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    logging.info("Ejecutando el análisis completo")
    ASSUMPTION_RESULTS.clear()

    # 1) Carga y agregación
    submov_path = os.path.join(output_comparisons_dir, 'submovement_detailed_valid.csv')
    if not os.path.exists(submov_path):
        print("No se encontró submovement_detailed_valid.csv → saliendo.")
        sys.exit(1)

    submovements_df = pd.read_csv(submov_path)
    aggregated_df   = aggregate_trial_metrics_extended(submovements_df)
    # Filtramos solo hombro, codo y muñeca
    parts_of_interest = ['Hombro', 'Codo', 'Muneca','Braquiradial','Bicep','Frente']
    aggregated_df = aggregated_df[aggregated_df['body_part'].isin(parts_of_interest)]

    tt_df = run_ttest_simple_by_site(aggregated_df, selected_metrics, output_comparisons_dir)
    tt_df.to_csv(os.path.join(output_comparisons_dir, 'ttest_simple_by_site.csv'), index=False)

    
    # 2) Filtrado global Gaussian-based
    df_global = prep_for_anova(aggregated_df, model='Gaussian-based', metric=None)

    # 3) Cálculo centralizado de ANOVAs (tipo II, por día/coord/parte)
    anova_full = do_significance_tests_aggregated(
        df_global,
        output_dir=output_comparisons_dir
    )
    anova_full.to_csv(
        os.path.join(output_comparisons_dir, 'anova_gaussian_full_by_group.csv'),
        index=False
    )

    # 4) Gráficos globales
    plot_global_summary(aggregated_df, output_comparisons_dir)
    plot_heatmap_gauss_from_anova(df_global, output_comparisons_dir, title="ANOVA Gauss – Global")

    # 5) Por día+coordenada
    combos = (
        aggregated_df.query("MovementType=='Gaussian-based'")
                     .dropna(subset=['Dia experimental','Coordenada_x','Coordenada_y'])
                     [['Dia experimental','Coordenada_x','Coordenada_y']]
                     .drop_duplicates()
    )
    for dia, x, y in combos.itertuples(index=False):

        reset_summary_pvals()                             # ← aquí

        df_grp = prep_for_anova(
            aggregated_df,
            model='Gaussian-based',
            day=dia,
            coord_x=x,
            coord_y=y
        )
        plot_summary_by_filters(
            df_grp,
            output_comparisons_dir,
            title_prefix=f"Summary_{dia}_Coord_{x}_{y}",
            model_filter=["Gaussian-based"]
        )
        plot_heatmap_gauss_from_anova(
            df_grp,
            output_comparisons_dir,
            title=f"ANOVA Gauss Día {dia} – X={x},Y={y}"
        )
        plot_3d_gaussian_boxplots_by_bodypart(df_grp, output_comparisons_dir,
                                              day=dia, coord_x=x, coord_y=y)

    # 6) Por body_part
    combos_bp = (
        aggregated_df.query("MovementType=='Gaussian-based'")
                     .dropna(subset=['Dia experimental','body_part'])
                     [['Dia experimental','body_part']]
                     .drop_duplicates()
    )
    for dia, bp in combos_bp.itertuples(index=False):
        reset_summary_pvals()                             # ← y aquí

        df_bp = prep_for_anova(
            aggregated_df,
            model='Gaussian-based',
            day=dia,
            body_part=bp
        )
        plot_summary_by_filters(
            df_bp,
            output_comparisons_dir,
            title_prefix=f"Summary_{dia}_{bp}",
            model_filter=["Gaussian-based"]
        )

    # 7) Guardar p-values y supuestos

    pd.DataFrame(SUMMARY_PVALS)\
    .to_csv(os.path.join(output_comparisons_dir, 'summary_gaussian_pvals.csv'),
            index=False)
    pd.DataFrame(SUMMARY_SAMPLE_SIZES)\
    .to_csv(os.path.join(output_comparisons_dir,
                        'summary_sample_sizes.csv'),
            index=False)
    print("Tamaños de muestra guardados en: summary_sample_sizes.csv")

    # Sólo grabamos ASSUMPTION_RESULTS si no está vacío
    assump_df = pd.DataFrame(ASSUMPTION_RESULTS)
    if not assump_df.empty:
        assump_df.sort_values('Test_ID')\
            .to_csv(os.path.join(output_comparisons_dir,
                                    'assumption_tests_gaussian.csv'),
                    index=False)
    else:
        print("No se generaron tests de supuestos.")

    # 8) Guardar agregado final
    aggregated_df.to_csv(
        os.path.join(output_comparisons_dir, 'aggregated_df_enriched.csv'),
        index=False
    )
    
    plot_model_compare_gauss_min(aggregated_df, by='global')

    
    
    print("¡Análisis completo!")
