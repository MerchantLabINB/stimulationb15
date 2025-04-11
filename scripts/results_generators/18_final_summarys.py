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
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import friedmanchisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import re
import shutil
import glob
import textwrap
import math

import plotly.graph_objects as go

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

shape_colors = {
    "rectangular": "orange",
    "rombo": "blue",
    "rampa ascendente": "green",
    "rampa descendente": "purple",
    "triple rombo": "red"
}

metric_labels = {
    "lat_inicio_ms": "Latencia al Inicio",
    "lat_primer_pico_ms": "Latencia al Primer Pico",
    "lat_pico_ultimo_ms": "Latencia al Último Pico",
    "lat_inicio_mayor_ms": "Latencia del Movimiento de Mayor Amplitud",
    "lat_pico_mayor_ms": "Latencia del Pico Mayor",
    "valor_pico_inicial": "Amplitud del Pico Inicial",
    "valor_pico_max": "Amplitud del Pico Máximo",
    "dur_total_ms": "Duración Total",
    "delta_t_pico": "Diferencia entre Primer y Pico Mayor",
    "num_movs": "Número de Movimientos"
}

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

    grouping_cols = ['Ensayo_Key', 'Ensayo', 'Estímulo', 'MovementType',
                     'Dia experimental', 'body_part', 'Coordenada_x', 'Coordenada_y',
                     'Distancia Intracortical']
    agg_list = []
    for name, group in submovements_df.groupby(grouping_cols):
        movement_type = name[3]
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
    return aggregated_df

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
    df_clean = df.dropna(subset=[metric, factor_name])
    if df_clean[factor_name].nunique() < 2:
        return None, None
    try:
        tukey_res = pairwise_tukeyhsd(endog=df_clean[metric].values,
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

def do_significance_tests_aggregated(aggregated_df, output_dir=None):
    if output_dir is None:
        output_dir = output_comparisons_dir

    if 'Forma_del_Pulso' not in aggregated_df.columns:
        aggregated_df['Forma_del_Pulso'] = aggregated_df['Estímulo'].apply(
            lambda s: s.split(',')[0].strip().lower() if isinstance(s, str) and ',' in s else np.nan)
    else:
        aggregated_df['Forma_del_Pulso'] = aggregated_df['Forma_del_Pulso'].str.lower().str.strip()

    if 'Duracion_ms' not in aggregated_df.columns:
        aggregated_df['Duracion_ms'] = aggregated_df['Estímulo'].apply(extract_duration)
    aggregated_df['Duracion_ms'] = aggregated_df['Duracion_ms'].apply(lambda x: str(int(x)) if not pd.isna(x) else x)
    aggregated_df = aggregated_df.loc[:, ~aggregated_df.columns.duplicated()]

    metrics = ['lat_inicio_ms', 'lat_primer_pico_ms', 'lat_pico_ultimo_ms',
               'dur_total_ms', 'valor_pico_inicial', 'valor_pico_max',
               'num_movs', 'lat_inicio_mayor_ms', 'lat_pico_mayor_ms', 'delta_t_pico']
    grouping = ['Dia experimental', 'body_part', 'MovementType']
    results_anova = []

    formulas = {
        "forma": lambda m: f"{m} ~ C(Forma_del_Pulso)",
        "duracion": lambda m: f"{m} ~ C(Duracion_ms)",
        "combinado": lambda m: f"{m} ~ C(Forma_del_Pulso) * C(Duracion_ms)"
    }

    for (dia, bp, movtype), df_sub in aggregated_df.groupby(grouping):
        if len(df_sub) < 3:
            continue
        for metric in metrics:
            data_metric = df_sub.dropna(subset=[metric])
            if len(data_metric) < 3:
                continue
            if data_metric['Duracion_ms'].nunique() < 2:
                logging.warning(f"En {dia}, {bp}, {movtype} para {metric} la variable Duracion_ms es constante. Se omite el efecto de duración.")
                try:
                    model = ols(f"{metric} ~ C(Forma_del_Pulso)", data=data_metric).fit()
                    anova_res = anova_lm(model, typ=2)
                    for row in anova_res.index:
                        if row == 'Residual':
                            continue
                        results_anova.append({
                            'Dia experimental': dia,
                            'body_part': bp,
                            'MovementType': movtype,
                            'Metric': metric,
                            'Modelo': "solo forma",
                            'Factor': row,
                            'sum_sq': anova_res.loc[row, 'sum_sq'],
                            'df': anova_res.loc[row, 'df'],
                            'F': anova_res.loc[row, 'F'],
                            'PR(>F)': anova_res.loc[row, 'PR(>F)'],
                            'Partial_Eta_Sq': calc_partial_eta_sq(anova_res, factor_row=row, resid_row='Residual'),
                            'Num_observations': len(data_metric)
                        })
                except Exception as e:
                    logging.warning(f"Fallo ANOVA (solo forma) en {dia}, {bp}, {movtype}, {metric}: {e}")
                continue

            for model_key, formula_func in formulas.items():
                formula = formula_func(metric)
                try:
                    model = ols(formula, data=data_metric).fit()
                    anova_res = anova_lm(model, typ=2)
                    for row in anova_res.index:
                        if row == 'Residual':
                            continue
                        results_anova.append({
                            'Dia experimental': dia,
                            'body_part': bp,
                            'MovementType': movtype,
                            'Metric': metric,
                            'Modelo': model_key,
                            'Factor': row,
                            'sum_sq': anova_res.loc[row, 'sum_sq'],
                            'df': anova_res.loc[row, 'df'],
                            'F': anova_res.loc[row, 'F'],
                            'PR(>F)': anova_res.loc[row, 'PR(>F)'],
                            'Partial_Eta_Sq': calc_partial_eta_sq(anova_res, factor_row=row, resid_row='Residual'),
                            'Num_observations': len(data_metric)
                        })
                except Exception as e:
                    logging.warning(f"Fallo ANOVA en {dia}, {bp}, {movtype}, {metric} ({model_key}): {e}")
                    continue
    anova_df = pd.DataFrame(results_anova)
    out_csv = os.path.join(output_dir, 'anova_results_aggregated.csv')
    anova_df.to_csv(out_csv, index=False)
    print(f"Resultados ANOVA guardados en: {out_csv}")

def do_significance_tests_aggregated_return(aggregated_df):
    """
    Ejecuta las pruebas de hipótesis (ANOVA, post‑hoc y Friedman) sobre la tabla de métricas
    agregadas y retorna un DataFrame (anova_df) con los resultados de ANOVA.
    Se agrupa por 'Dia experimental', 'body_part' y 'MovementType' y para cada métrica se calcula:
       - Sum of Squares
       - Degrees of Freedom
       - F Value
       - P Value
       - Partial Eta Squared
       - Número de observaciones
    """
    # Aseguramos que 'Forma_del_Pulso' y 'Duracion_ms' estén creadas y en el formato correcto.
    if 'Forma_del_Pulso' not in aggregated_df.columns:
        aggregated_df['Forma_del_Pulso'] = aggregated_df['Estímulo'].apply(
            lambda s: s.split(',')[0].strip().lower() if isinstance(s, str) and ',' in s else np.nan)
    else:
        aggregated_df['Forma_del_Pulso'] = aggregated_df['Forma_del_Pulso'].str.lower().str.strip()

    if 'Duracion_ms' not in aggregated_df.columns:
        aggregated_df['Duracion_ms'] = aggregated_df['Estímulo'].apply(extract_duration)
    aggregated_df['Duracion_ms'] = aggregated_df['Duracion_ms'].apply(lambda x: str(int(x)) if not pd.isna(x) else x)
    aggregated_df = aggregated_df.loc[:, ~aggregated_df.columns.duplicated()]

    metrics = ['lat_inicio_ms', 'lat_primer_pico_ms', 'lat_pico_ultimo_ms',
               'dur_total_ms', 'valor_pico_inicial', 'valor_pico_max',
               'num_movs', 'lat_inicio_mayor_ms', 'lat_pico_mayor_ms', 'delta_t_pico']
    grouping = ['Dia experimental', 'body_part', 'MovementType']
    results_anova = []
    formulas = {
        "forma": lambda m: f"{m} ~ C(Forma_del_Pulso)",
        "duracion": lambda m: f"{m} ~ C(Duracion_ms)",
        "combinado": lambda m: f"{m} ~ C(Forma_del_Pulso) * C(Duracion_ms)"
    }

    for (dia, bp, movtype), df_sub in aggregated_df.groupby(grouping):
        if len(df_sub) < 3:
            continue
        for metric in metrics:
            data_metric = df_sub.dropna(subset=[metric])
            if len(data_metric) < 3:
                continue
            # Si Duracion_ms es constante, se ajusta solo con Forma_del_Pulso
            if data_metric['Duracion_ms'].nunique() < 2:
                try:
                    model = ols(f"{metric} ~ C(Forma_del_Pulso)", data=data_metric).fit()
                    anova_res = anova_lm(model, typ=2)
                    for row in anova_res.index:
                        if row == 'Residual':
                            continue
                        results_anova.append({
                            'Dia experimental': dia,
                            'Body Part': bp,
                            'Movement Model': movtype,
                            'Metric': metric,
                            'ANOVA Model': "Solo Forma",
                            'Factor': row,
                            'Sum of Squares': anova_res.loc[row, 'sum_sq'],
                            'Degrees of Freedom': anova_res.loc[row, 'df'],
                            'F Value': anova_res.loc[row, 'F'],
                            'P Value': anova_res.loc[row, 'PR(>F)'],
                            'Partial Eta Squared': calc_partial_eta_sq(anova_res, factor_row=row, resid_row='Residual'),
                            'N Observations': len(data_metric)
                        })
                except Exception as e:
                    logging.warning(f"Fallo ANOVA (solo forma) en {dia}, {bp}, {movtype}, {metric}: {e}")
                continue

            for model_key, formula_func in formulas.items():
                formula = formula_func(metric)
                try:
                    model = ols(formula, data=data_metric).fit()
                    anova_res = anova_lm(model, typ=2)
                    for row in anova_res.index:
                        if row == 'Residual':
                            continue
                        results_anova.append({
                            'Dia experimental': dia,
                            'Body Part': bp,
                            'Movement Model': movtype,
                            'Metric': metric,
                            'ANOVA Model': model_key,
                            'Factor': row,
                            'Sum of Squares': anova_res.loc[row, 'sum_sq'],
                            'Degrees of Freedom': anova_res.loc[row, 'df'],
                            'F Value': anova_res.loc[row, 'F'],
                            'P Value': anova_res.loc[row, 'PR(>F)'],
                            'Partial Eta Squared': calc_partial_eta_sq(anova_res, factor_row=row, resid_row='Residual'),
                            'N Observations': len(data_metric)
                        })
                except Exception as e:
                    logging.warning(f"Fallo ANOVA en {dia}, {bp}, {movtype}, {metric} ({model_key}): {e}")
                    continue
    anova_df = pd.DataFrame(results_anova)
    return anova_df

def compute_model_pvals(agg_df, metric):
    models = ['Threshold-based', 'Gaussian-based', 'MinimumJerk']
    pvals = {}
    for mt in models:
        df_model = agg_df[agg_df['MovementType'] == mt]
        if df_model.empty or len(df_model) < 3:
            logging.warning(f"Modelo {mt} para {metric}: Insuficientes datos (n={len(df_model)}).")
            pvals[mt] = (np.nan, np.nan, np.nan)
            continue
        unique_forms = df_model["Forma_del_Pulso"].unique()
        if len(unique_forms) < 2:
            logging.warning(f"Modelo {mt} para {metric}: 'Forma_del_Pulso' tiene un solo nivel: {unique_forms}.")
        dur_var = df_model["Duracion_ms"].var()
        if dur_var == 0:
            logging.warning(f"Modelo {mt} para {metric}: Varianza cero en 'Duracion_ms'. Valores: {df_model['Duracion_ms'].tolist()}")
        try:
            mod = ols(f"{metric} ~ C(Forma_del_Pulso) * Duracion_ms", data=df_model).fit()
            anova_res = anova_lm(mod, typ=2)
            p_shape = anova_res.loc["C(Forma_del_Pulso)", "PR(>F)"] if "C(Forma_del_Pulso)" in anova_res.index else np.nan
            p_dur = anova_res.loc["Duracion_ms", "PR(>F)"] if "Duracion_ms" in anova_res.index else np.nan
            p_int = anova_res.loc["C(Forma_del_Pulso):Duracion_ms", "PR(>F)"] if "C(Forma_del_Pulso):Duracion_ms" in anova_res.index else np.nan
            if pd.isna(p_dur):
                logging.warning(
                    f"Modelo {mt} para {metric}: p-valor de 'Duracion_ms' es NA.\nResumen: mean={df_model['Duracion_ms'].mean():.3g}, var={dur_var:.3g}, min={df_model['Duracion_ms'].min()}, max={df_model['Duracion_ms'].max()}, unique={df_model['Duracion_ms'].unique()}.\nANOVA result:\n{anova_res}"
                )
            pvals[mt] = (p_shape, p_dur, p_int)
        except Exception as e:
            logging.error(f"Error en ANOVA para modelo {mt} y {metric}: {e}\nDatos: Forma_del_Pulso={df_model['Forma_del_Pulso'].unique()}, Duracion_ms={df_model['Duracion_ms'].tolist()}")
            pvals[mt] = (np.nan, np.nan, np.nan)
    return pvals


def format_stat(label, p, F):
    if pd.isna(p) or pd.isna(F):
        return f"{label}: NA"
    else:
        prefix = "*" if p < 0.05 else ""
        return f"{label}: {prefix}p={p:.3g} (F={F:.2f})"

def compute_model_stats(agg_df, metric):
    models = ['Threshold-based', 'Gaussian-based', 'MinimumJerk']
    stats = {}
    for mt in models:
        df_model = agg_df[agg_df['MovementType'] == mt]
        if df_model.empty or len(df_model) < 3:
            logging.warning(f"Modelo {mt} para {metric}: Insuficientes datos (n={len(df_model)}).")
            stats[mt] = {'forma_p': np.nan, 'duracion_p': np.nan, 'interaccion_p': np.nan,
                         'forma_F': np.nan, 'duracion_F': np.nan, 'interaccion_F': np.nan}
            continue
        try:
            mod = ols(f"{metric} ~ C(Forma_del_Pulso) * C(Duracion_ms)", data=df_model).fit()
            anova_res = anova_lm(mod, typ=2)
            p_shape = anova_res.loc["C(Forma_del_Pulso)", "PR(>F)"] if "C(Forma_del_Pulso)" in anova_res.index else np.nan
            p_dur = anova_res.loc["C(Duracion_ms)", "PR(>F)"] if "C(Duracion_ms)" in anova_res.index else np.nan
            p_int = anova_res.loc["C(Forma_del_Pulso):C(Duracion_ms)", "PR(>F)"] if "C(Forma_del_Pulso):C(Duracion_ms)" in anova_res.index else np.nan
            F_shape = anova_res.loc["C(Forma_del_Pulso)", "F"] if "C(Forma_del_Pulso)" in anova_res.index else np.nan
            F_dur = anova_res.loc["C(Duracion_ms)", "F"] if "C(Duracion_ms)" in anova_res.index else np.nan
            F_int = anova_res.loc["C(Forma_del_Pulso):C(Duracion_ms)", "F"] if "C(Forma_del_Pulso):C(Duracion_ms)" in anova_res.index else np.nan
            stats[mt] = {'forma_p': p_shape, 'duracion_p': p_dur, 'interaccion_p': p_int,
                         'forma_F': F_shape, 'duracion_F': F_dur, 'interaccion_F': F_int}
        except Exception as e:
            logging.error(f"Error en ANOVA para {mt} y {metric}: {e}")
            stats[mt] = {'forma_p': np.nan, 'duracion_p': np.nan, 'interaccion_p': np.nan,
                         'forma_F': np.nan, 'duracion_F': np.nan, 'interaccion_F': np.nan}
    return stats

def get_tukey_pvals_for_stimulus(agg_df, stim, metric):
    df_sub = agg_df[agg_df['Estímulo'] == stim]
    if df_sub['MovementType'].nunique() < 2:
        return None
    tukey_res = pairwise_tukeyhsd(endog=df_sub[metric].values,
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
    df = aggregated_df.copy()

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
    desired_forms = ["rectangular", "rombo", "triple rombo", "rampa ascendente"]
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
    global_latency_max = df[latency_metrics].max().max() if not df[latency_metrics].empty else None
    global_peak_max = df[peak_metrics].quantile(0.95).max() if not df[peak_metrics].empty else None

    if model_filter is not None and model_filter == ["Gaussian-based"]:
        box_width = 0.6 / 0.5  # cajas más anchas
        median_line_color = 'silver'  # mediana en plateado
        median_line_width = 3.0
    else:
        box_width = 0.6 / 2.2
        median_line_color = 'black'
        median_line_width = 4.0

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

        bp_obj = ax.boxplot(boxplot_data, positions=x_positions, widths=box_width,
                            patch_artist=True, showfliers=False,
                            medianprops={'color': median_line_color, 'linewidth': median_line_width})
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
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Estímulo")
        ax.set_ylabel(yaxis_units.get(metric, ""))
        ax.set_title(metric_labels.get(metric, metric))
        if metric in latency_metrics:
            ax.set_ylim(0, global_latency_max * 0.8)
        elif metric in peak_metrics:
            ax.set_ylim(0, global_peak_max * 0.8)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 0.8)

        # Se anota una línea con los p-valores del modelo Gaussiano (como ejemplo)
        # Se anota una caja en la esquina superior izquierda con los p-valores del modelo Gaussiano (dividido en tres líneas)
        model_stats = compute_model_stats(df, metric)
        gauss_stats = model_stats.get("Gaussian-based", {})
        if not gauss_stats:
            gauss_text = "Forma: NA\nDuración: NA\nInt: NA"
        else:
            forma_text = "Forma: " + (f"{'*' if not pd.isna(gauss_stats.get('forma_p')) and gauss_stats.get('forma_p') < 0.05 else ''}p={gauss_stats.get('forma_p'):.3g}" if not pd.isna(gauss_stats.get('forma_p')) else "NA")
            dur_text = "Duración: " + (f"{'*' if not pd.isna(gauss_stats.get('duracion_p')) and gauss_stats.get('duracion_p') < 0.05 else ''}p={gauss_stats.get('duracion_p'):.3g}" if not pd.isna(gauss_stats.get('duracion_p')) else "NA")
            int_text = "Int: " + (f"{'*' if not pd.isna(gauss_stats.get('interaccion_p')) and gauss_stats.get('interaccion_p') < 0.05 else ''}p={gauss_stats.get('interaccion_p'):.3g}" if not pd.isna(gauss_stats.get('interaccion_p')) else "NA")
            gauss_text = f"{forma_text}\n{dur_text}\n{int_text}"
        ax.text(0.01, 0.99, gauss_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))



        pval_matrix = get_tukey_pvals_for_stimulus(df, stim, metric)
        if pval_matrix is not None:
            box_positions = positions_by_stim[stim]
            pairs = list(itertools.combinations(sorted(box_positions.keys()), 2))
            add_significance_brackets(ax, pairs, pval_matrix, box_positions, y_offset=0.03, line_height=0.02, font_size=6)

    # Establecer el título final del gráfico usando final_title
    fig.suptitle(final_title, fontsize=16)
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
                    title=dict(text="Estímulo", font=dict(size=14)),
                    tickvals=list(stim_to_x.values()),
                    ticktext=list(stim_to_x.keys()),
                    range=[min(stim_to_x.values()) - 0.5, max(stim_to_x.values()) + 0.5]
                ),
                yaxis=dict(
                    title="Body Part",
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

def plot_global_summary(aggregated_df, output_dir):
    title = "Global_Summary"
    plot_summary_by_filters(aggregated_df, output_dir, title_prefix=title, model_filter=["Gaussian-based"])

def plot_summary_by_bodypart(aggregated_df, output_dir):
    for (dia, bp), df_sub in aggregated_df.groupby(['Dia experimental', 'body_part']):
        title = f"Summary_{dia}_{bp}"
        plot_summary_by_filters(df_sub, output_dir, title_prefix=title, model_filter=["Gaussian-based"])


def plot_model_validation_full_by_group(aggregated_df, output_dir, group_by, metrics=None):
    """
    Para cada grupo definido por 'group_by', ajusta un modelo ANOVA para cada una de las 6 métricas seleccionadas
    usando la fórmula:
         metric ~ C(Forma_del_Pulso) + C(Duracion_ms) [+ C(body_part) si existe] + C(MovementType)
    Extrae los p-values de cada factor: "Forma", "Duración", "Body Part" (si existe) y "Modelo" (C(MovementType)).
    Luego, genera un gráfico de barras agrupado para cada grupo en donde se muestra -log10(p) para cada factor y
    métrica, con una línea horizontal en -log10(p)=1.3 (p < 0.05). Las barras por encima de ese valor se marcan con asterisco.
    Se reordena la matriz resultante para que las filas sigan el orden de 'selected_metrics'.
    Las etiquetas del eje x se muestran en diagonal y los textos sobre las barras se reducen.
    """
    if metrics is None:
        metrics = selected_metrics

    results_list = []
    groups = aggregated_df.groupby(group_by)
    # Por cada grupo...
    for group_name, group_df in groups:
        include_body = 'body_part' in group_df.columns
        for metric in metrics:
            df_metric = group_df.dropna(subset=[metric, 'Forma_del_Pulso', 'Duracion_ms', 'MovementType'])
            if include_body:
                df_metric = df_metric.dropna(subset=['body_part'])
            if len(df_metric) < 3:
                continue
            # Construir la fórmula con body_part si existe
            if include_body:
                formula = f"{metric} ~ C(Forma_del_Pulso) + C(Duracion_ms) + C(body_part) + C(MovementType)"
            else:
                formula = f"{metric} ~ C(Forma_del_Pulso) + C(Duracion_ms) + C(MovementType)"
            try:
                model = ols(formula, data=df_metric).fit()
                anova_res = anova_lm(model, typ=2)
                factors = {}
                if "C(Forma_del_Pulso)" in anova_res.index:
                    factors["Forma"] = anova_res.loc["C(Forma_del_Pulso)", "PR(>F)"]
                if "C(Duracion_ms)" in anova_res.index:
                    factors["Duración"] = anova_res.loc["C(Duracion_ms)", "PR(>F)"]
                # Si se incluyó body_part, se omite su nombre para simplificar
                if "C(MovementType)" in anova_res.index:
                    factors["Modelo"] = anova_res.loc["C(MovementType)", "PR(>F)"]
                # La interacción se captura de la combinación
                if "C(Forma_del_Pulso):C(Duracion_ms)" in anova_res.index:
                    factors["Interacción"] = anova_res.loc["C(Forma_del_Pulso):C(Duracion_ms)", "PR(>F)"]
                for factor, p_val in factors.items():
                    results_list.append({
                        "Group": group_name,
                        "Metric": metric,
                        "Factor": factor,
                        "P Value": p_val
                    })
            except Exception as e:
                logging.warning(f"Fallo model validation en grupo {group_name}, métrica {metric}: {e}")
                continue

    results_df = pd.DataFrame(results_list)
    if results_df.empty:
        print("No hay datos suficientes para model validation.")
        return

    results_df['-log10(p)'] = -np.log10(results_df['P Value'])
    
    # Genera un gráfico para cada grupo
    unique_groups = results_df["Group"].unique()
    for group in unique_groups:
        group_df = results_df[results_df["Group"] == group]
        # Pivotear la tabla y reordenar las filas según selected_metrics
        pivot_table = group_df.pivot(index="Metric", columns="Factor", values="-log10(p)")
        pivot_table = pivot_table.reindex(metrics)  # asegura el mismo orden que en selected_metrics
        pivot_table.index = [metric_labels.get(m, m) for m in pivot_table.index]

        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_table.plot(kind="bar", ax=ax)
        ax.axhline(1.3, color="red", linestyle="--", label="p = 0.05 (-log10=1.3)")
        ax.set_ylabel("-log10(p-value)")
        # Rotar las etiquetas del eje x en diagonal
        plt.xticks(rotation=45, ha='right')
        # Reducir el tamaño de los números (bar labels)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8)
            for rect in container:
                if rect.get_height() > 1.3:
                    ax.text(rect.get_x() + rect.get_width()/2., rect.get_height(),
                            "*", ha='center', va='bottom', color="black", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        # Agregar un título completo para el grupo
        if isinstance(group, tuple):
            group_str = " / ".join(f"{col}={val}" for col, val in zip(group_by, group))
        else:
            group_str = f"{group_by[0]} = {group}"
        # Se reduce un poco la fuente del título para que no se corte
        fig.suptitle(f"Model Validation - {group_str}", fontsize=14)
        fname = sanitize_filename(f"model_validation_full_{group_str}.png")
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Gráfico Model Validation para grupo {group_str} guardado en: {out_path}")

def generate_anova_summary_table(aggregated_df, output_dir):
    """
    Genera una tabla resumen con los resultados de ANOVA (p-values, F-values, etc.),
    renombrando las columnas para que sean descriptivas y guardándola en CSV.
    """
    anova_df = do_significance_tests_aggregated_return(aggregated_df)
    # Renombramos columnas para mayor claridad
    anova_df = anova_df.rename(columns={
        'Dia experimental': 'Day',
        'Body Part': 'Body_Part',
        'Movement Model': 'Movement_Model',
        'Metric': 'Metric',
        'ANOVA Model': 'ANOVA_Model',
        'Factor': 'Factor',
        'Sum of Squares': 'SS',
        'Degrees of Freedom': 'df',
        'F Value': 'F',
        'P Value': 'p',
        'Partial Eta Squared': 'Partial_Eta_Sq',
        'N Observations': 'N'
    })
    out_csv = os.path.join(output_dir, 'anova_summary_table.csv')
    anova_df.to_csv(out_csv, index=False)
    print(f"Tabla ANOVA resumida guardada en: {out_csv}")
    return anova_df
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

def plot_anova_significance_heatmap_gaussian(aggregated_df, output_dir):
    """
    Genera un heatmap general que resume la significancia del modelo Gaussiano para las 6 métricas seleccionadas
    y para los 3 factores: Forma, Duración e Interacción. Cada celda muestra 1 (si -log10(p) > 1.3) o 0.
    Se reordenan las filas para que coincidan con el orden en selected_metrics.
    """
    anova_df = do_significance_tests_aggregated_return(aggregated_df)
    anova_df = anova_df[anova_df['Movement Model'] == "Gaussian-based"].copy()
    anova_df = anova_df[anova_df["Metric"].isin(selected_metrics)]
    # Remapear los factores para dejar solo Forma, Duración e Interacción (sin body_part)
    factor_mapping = {
        "C(Forma_del_Pulso)": "Forma",
        "C(Duracion_ms)": "Duración",
        "C(Forma_del_Pulso):C(Duracion_ms)": "Interacción"
    }
    anova_df['Factor'] = anova_df['Factor'].replace(factor_mapping)
    # Usar solo los factores deseados
    anova_df = anova_df[anova_df['Factor'].isin(["Forma", "Duración", "Interacción"])]
    # Eliminar duplicados por combinación de Métrica y Factor
    summary = anova_df.drop_duplicates(subset=["Metric", "Factor"])[["Metric", "Factor", "P Value"]]
    summary['neg_log_p'] = -np.log10(summary['P Value'])
    summary['significant'] = (summary['neg_log_p'] > 1.3).astype(int)
    pivot_table = summary.pivot(index='Metric', columns='Factor', values='significant')
    pivot_table = pivot_table.reindex(selected_metrics)
    pivot_table.index = [metric_labels.get(m, m) for m in pivot_table.index]

    cmap = LinearSegmentedColormap.from_list("binary_significance", ["lightblue", "salmon"], N=2)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot_table, annot=True, fmt="d", cmap=cmap,
                     cbar_kws={'ticks': [0, 1], 'label': 'Significancia'},
                     linewidths=0.5, linecolor='white')
    ax.set_xlabel("Factor")
    ax.set_ylabel("Métrica")
    ax.set_title("Heatmap de Significancia ANOVA (Modelo Gaussiano) - General", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(output_dir, sanitize_filename("anova_significance_heatmap_gaussian_general.png"))
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Heatmap de ANOVA (Modelo Gaussiano) general guardado en: {out_path}")


def plot_anova_significance_heatmap_gaussian_by_day(aggregated_df, output_dir):
    """
    Genera un heatmap para cada día que resume la significancia del modelo Gaussiano para las 6 métricas seleccionadas.
    Se consideran únicamente los factores: Forma, Duración e Interacción (sin concatenar body_part).
    El título sigue el formato: "Heatmap de Significancia ANOVA (Modelo Gaussiano) - Día: [day]"
    """
    anova_df = do_significance_tests_aggregated_return(aggregated_df)
    anova_df = anova_df[anova_df['Movement Model'] == "Gaussian-based"].copy()
    anova_df = anova_df[anova_df["Metric"].isin(selected_metrics)]
    factor_mapping = {
        "C(Forma_del_Pulso)": "Forma",
        "C(Duracion_ms)": "Duración",
        "C(Forma_del_Pulso):C(Duracion_ms)": "Interacción"
    }
    anova_df['Factor'] = anova_df['Factor'].replace(factor_mapping)
    anova_df = anova_df[anova_df['Factor'].isin(["Forma", "Duración", "Interacción"])]

    unique_days = aggregated_df["Dia experimental"].unique()
    for day in unique_days:
        day_str = str(day)
        day_sanitized = sanitize_filename(day_str)
        day_df = anova_df[anova_df['Dia experimental'] == day]
        summary = day_df.drop_duplicates(subset=["Metric", "Factor"])[["Metric", "Factor", "P Value"]]
        summary['neg_log_p'] = -np.log10(summary['P Value'])
        summary['significant'] = (summary['neg_log_p'] > 1.3).astype(int)
        pivot_table = summary.pivot(index='Metric', columns='Factor', values='significant')
        pivot_table = pivot_table.reindex(selected_metrics)
        pivot_table.index = [metric_labels.get(m, m) for m in pivot_table.index]

        plt.figure(figsize=(10, 8))
        cmap = LinearSegmentedColormap.from_list("binary_significance", ["lightblue", "salmon"], N=2)
        ax = sns.heatmap(pivot_table, annot=True, fmt="d", cmap=cmap,
                         cbar_kws={'ticks': [0, 1], 'label': 'Significancia'},
                         linewidths=0.5, linecolor='white')
        ax.set_xlabel("Factor")
        ax.set_ylabel("Métrica")
        ax.set_title(f"Heatmap de Significancia ANOVA (Modelo Gaussiano) - Día: {day_str}", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"anova_significance_heatmap_gaussian_day_{day_sanitized}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Heatmap de ANOVA (Modelo Gaussiano) para Día {day_str} guardado en: {out_path}")

def preprocess_estimulo(df):
    df['Forma_del_Pulso'] = df['Estímulo'].apply(
        lambda s: s.split(',')[0].strip().lower() if isinstance(s, str) and ',' in s else np.nan)
    df['Duracion_ms'] = df['Estímulo'].apply(extract_duration)
    df = df.loc[:, ~df.columns.duplicated()]
    return df



def preprocess_estimulo(df):
    df['Forma_del_Pulso'] = df['Estímulo'].apply(
        lambda s: s.split(',')[0].strip().lower() if isinstance(s, str) and ',' in s else np.nan)
    df['Duracion_ms'] = df['Estímulo'].apply(extract_duration)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# -------------------------------
# BLOQUE PRINCIPAL
# -------------------------------
if __name__ == "__main__":
    logging.info("Ejecutando el bloque principal del script.")
    submov_path = os.path.join(output_comparisons_dir, 'submovement_detailed_summary.csv')
    if os.path.exists(submov_path):
        submovements_df = pd.read_csv(submov_path)
        submovements_df = preprocess_estimulo(submovements_df)
        aggregated_df = aggregate_trial_metrics_extended(submovements_df)
        aggregated_df['Duracion_ms'] = aggregated_df['Estímulo'].apply(extract_duration)
        aggregated_df['Duracion_ms'] = aggregated_df['Duracion_ms'].apply(lambda x: str(int(x)) if not pd.isna(x) else x)
        aggregated_csv = os.path.join(output_comparisons_dir, 'aggregated_metrics.csv')
        aggregated_df.to_csv(aggregated_csv, index=False)
        print("Tabla agregada de métricas guardada en 'aggregated_metrics.csv'")
        
        # Llamadas a funciones de resumen:
        do_significance_tests_aggregated(aggregated_df, output_dir=output_comparisons_dir)
        aggregated_df["General"] = "General"

        
        # Resumen de boxplots generales (todos los modelos)
        plot_global_summary(aggregated_df, output_comparisons_dir)
        # Resumen por día + coordenada
        plot_summary_by_day_coord(aggregated_df, output_comparisons_dir)
        # Resumen por bodypart (por día)
        plot_summary_by_bodypart(aggregated_df, output_comparisons_dir)

       
        # Gráficos 3D de Gaussian-based
        for (dia, coord_x, coord_y), group in aggregated_df.groupby(['Dia experimental', 'Coordenada_x', 'Coordenada_y']):
            plot_3d_gaussian_boxplots_by_bodypart(group, output_comparisons_dir, day=dia, coord_x=coord_x, coord_y=coord_y)
        
        
        # Validación de modelos: por día y general
        # Llamada a la nueva validación con todos los factores
        plot_model_validation_full_by_group(aggregated_df, output_comparisons_dir, group_by=['Dia experimental'])
        plot_model_validation_full_by_group(aggregated_df, output_comparisons_dir, group_by=["General"])


        # Después de haber generado aggregated_df...
        anova_summary = generate_anova_summary_table(aggregated_df, output_comparisons_dir)
        descriptive_stats = generate_descriptive_stats_table(aggregated_df, output_comparisons_dir)
        plot_anova_significance_heatmap_gaussian(aggregated_df, output_comparisons_dir)
        plot_anova_significance_heatmap_gaussian_by_day(aggregated_df, output_comparisons_dir)


    else:
        print("No se encontró el archivo submovement_detailed_summary.csv para generar los análisis.")
    print("Proceso completo finalizado.")
