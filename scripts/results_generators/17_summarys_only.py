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

# --- IMPORTACIÓN DE PLOTLY PARA GRÁFICOS 3D INTERACTIVOS

import plotly.graph_objects as go

# --- CONFIGURACIÓN DEL LOGGING
log_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\filtered_processing_log.txt'
logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# --- DIRECTORIOS Y CONFIGURACIÓN INICIAL
stimuli_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information_expanded.csv'
csv_folder = r'C:\Users\samae\Documents\GitHub\stimulationb15\DeepLabCut\xv_lat-Br-2024-10-02\videos'
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\plot_trials'
if not os.path.exists(output_comparisons_dir):
    os.makedirs(output_comparisons_dir)

# Función para verificar archivos
def verificar_archivo(path, nombre_archivo):
    if not os.path.exists(path):
        logging.error(f"Archivo no encontrado: {path}")
        sys.exit(f"Archivo no encontrado: {path}")
    else:
        logging.info(f"Archivo encontrado: {path}")

verificar_archivo(stimuli_info_path, 'Stimuli_information.csv')

# Cargar stimuli_info
try:
    stimuli_info = pd.read_csv(stimuli_info_path)
    stimuli_info = stimuli_info[stimuli_info['Descartar'] == 'No']
    stimuli_info['Forma del Pulso'] = stimuli_info['Forma del Pulso'].str.lower()
except Exception as e:
    logging.error(f'Error al cargar Stimuli_information.csv: {e}')
    sys.exit(f'Error al cargar Stimuli_information.csv: {e}')

# Diccionarios de colores y paleta de pulse shapes
body_parts_specific_colors = {
    'Frente': 'blue',
    'Hombro': 'orange',
    'Codo': 'green',
    'Muneca': 'red',
    'Braquiradial': 'grey',
    'Bicep': 'brown'
}

body_parts = list(body_parts_specific_colors.keys())

# Asigna colores fijos para las formas (todo en minúsculas)
shape_colors = {
    "rectangular": "orange",
    "rombo": "blue",
    "rampa ascendente": "green",
    "rampa descendente": "purple",
    "triple rombo": "red"
}
def sanitize_filename(filename):
    """Elimina caracteres no permitidos en nombres de archivos."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

###############################################################################
# ETAPA 1: AGREGACIÓN DE MÉTRICAS POR ENSAYO (CON AJUSTE DE LATENCIAS)
###############################################################################
def aggregate_trial_metrics_extended(submovements_df):
    """
    Agrupa los submovimientos detallados por ensayo (identificados por Ensayo_Key, Ensayo, Estímulo,
    Dia experimental y body_part) y calcula métricas en milisegundos.
    
    Para convertir a tiempos relativos:
      • Threshold‑based: se asume que “Latencia al Inicio (s)” ya es (Frame-100)/100.
      • MinimumJerk: se calcula (t_start - 1) en s.
      • Gaussian‑based: se calcula (mu_gauss - 1) en s.
    """
    fs = 100.0  # frames/s

    def agg_threshold(group):
        return pd.Series({
            "lat_inicio_ms": group["Latencia al Inicio (s)"].min() * 1000,
            "lat_primer_pico_ms": group["Latencia al Pico (s)"].min() * 1000,
            "lat_pico_ultimo_ms": group["Latencia al Pico (s)"].max() * 1000,
            "dur_total_ms": ((group["Fin Movimiento (Frame)"].max() - group["Inicio Movimiento (Frame)"].min()) / fs) * 1000,
            "valor_pico_inicial": group["Valor Pico (velocidad)"].min(),
            "valor_pico_max": group["Valor Pico (velocidad)"].max(),
            "num_movs": group.shape[0],
            "lat_inicio_mayor_ms": group.loc[group["Valor Pico (velocidad)"].idxmax(), "Latencia al Inicio (s)"] * 1000,
            "lat_pico_mayor_ms": group.loc[group["Valor Pico (velocidad)"].idxmax(), "Latencia al Pico (s)"] * 1000,
            "delta_valor_pico": group["Valor Pico (velocidad)"].max() - group["Valor Pico (velocidad)"].min()
        })

    def agg_gaussian(group):
        # Convertir mu_gauss a latencia relativa restándole 1 s
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
            "delta_valor_pico": group["A_gauss"].max() - group["A_gauss"].min()
        })

    def agg_minjerk(group):
        # Restamos 1 s a t_start y t_peak
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
            "delta_valor_pico": group["valor_pico"].max() - group["valor_pico"].min()
        })

    # Agrega las columnas 'Coordenada_x', 'Coordenada_y' y 'Distancia Intracortical'
    grouping_cols = ['Ensayo_Key', 'Ensayo', 'Estímulo', 'MovementType', 'Dia experimental', 'body_part', 'Coordenada_x', 'Coordenada_y', 'Distancia Intracortical']

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
                "num_movs", "lat_inicio_mayor_ms", "lat_pico_mayor_ms", "delta_valor_pico"
            ]})
        for i, col in enumerate(grouping_cols):
            agg_metrics[col] = name[i]
        agg_list.append(agg_metrics)
    aggregated_df = pd.DataFrame(agg_list)
    return aggregated_df









###############################################################################
# ETAPA 2: TESTS DE HIPÓTESIS (ANOVA, Post-hoc, Friedman)
###############################################################################
def calc_partial_eta_sq(anova_table, factor_row='C(Q(\'Forma del Pulso\'))', resid_row='Residual'):
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
        tukey_res = pairwise_tukeyhsd(endog=df_clean[metric].values, groups=df_clean[factor_name].values, alpha=0.05)
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
    """
    Realiza ANOVA (con dos factores: extraídos de 'Estímulo') sobre la tabla de métricas agregadas.
    Guarda los resultados de ANOVA, partial eta² y Friedman en archivos CSV.
    """
    if output_dir is None:
        output_dir = output_comparisons_dir
    # Actualizamos los nombres: usamos "lat_pico_ultimo_ms" en lugar de "lat_pico_max_ms"
    aggregated_df['Forma del Pulso'] = aggregated_df['Estímulo'].apply(lambda s: s.split(', ')[0] if isinstance(s, str) and ', ' in s else np.nan)
    aggregated_df['Duración (ms)'] = aggregated_df['Estímulo'].apply(lambda s: float(s.split(', ')[1].replace(' ms','')) if isinstance(s, str) and ', ' in s else np.nan)
    metrics = ['lat_inicio_ms', 'lat_primer_pico_ms', 'lat_pico_ultimo_ms',
               'dur_total_ms', 'valor_pico_inicial', 'valor_pico_max',
               'num_movs', 'lat_inicio_mayor_ms', 'lat_pico_mayor_ms', 'delta_valor_pico']
    grouping = ['Dia experimental', 'body_part', 'MovementType']
    results_anova = []
    results_friedman = []
    for (dia, bp, movtype), df_sub in aggregated_df.groupby(grouping):
        if len(df_sub) < 3:
            continue
        for metric in metrics:
            data_metric = df_sub.dropna(subset=[metric])
            if len(data_metric) < 3:
                continue
            formula = f"Q('{metric}') ~ C(Q('Forma del Pulso')) * C(Q('Duración (ms)'))"
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
                        'Factor': row,
                        'sum_sq': anova_res.loc[row, 'sum_sq'],
                        'df': anova_res.loc[row, 'df'],
                        'F': anova_res.loc[row, 'F'],
                        'PR(>F)': anova_res.loc[row, 'PR(>F)'],
                        'Partial_Eta_Sq': calc_partial_eta_sq(anova_res, factor_row=row, resid_row='Residual'),
                        'Num_observations': len(data_metric)
                    })
            except Exception as e:
                logging.warning(f"Fallo ANOVA en {dia}, {bp}, {movtype}, {metric}: {e}")
                continue
        try:
            metric_friedman = 'lat_inicio_ms'
            df_sub['Condicion'] = df_sub['Forma del Pulso'] + "_" + df_sub['Duración (ms)'].astype(str)
            pivot_f = df_sub.pivot_table(index='Ensayo', columns='Condicion', values=metric_friedman)
            pivot_f = pivot_f.dropna(axis=1, how='all').dropna(axis=0, how='any')
            if pivot_f.shape[1] > 1 and pivot_f.shape[0] > 2:
                stats_result = friedmanchisquare(*[pivot_f[col] for col in pivot_f])
                results_friedman.append({
                    'Dia experimental': dia,
                    'body_part': bp,
                    'MovementType': movtype,
                    'Metric': metric_friedman,
                    'Friedman_statistic': stats_result.statistic,
                    'Friedman_pvalue': stats_result.pvalue,
                    'Num_conditions': pivot_f.shape[1],
                    'Num_subjects': pivot_f.shape[0]
                })
        except Exception as e:
            logging.warning(f"Fallo Friedman en {dia}, {bp}, {movtype}: {e}")
    anova_df = pd.DataFrame(results_anova)
    friedman_df = pd.DataFrame(results_friedman)
    anova_df.to_csv(os.path.join(output_dir, 'anova_twofactor_results_aggregated.csv'), index=False)
    friedman_df.to_csv(os.path.join(output_dir, 'friedman_results_aggregated.csv'), index=False)
    print("Resultados ANOVA y Friedman (agregados) guardados.")


def compute_model_pvals(agg_df, metric):
    """
    Para cada modelo (Threshold-based, Gaussian-based, MinimumJerk), se calcula la ANOVA 
    con dos factores (Forma del Pulso y Duración (ms)) sobre la métrica dada y se extraen:
      - p_shape: p-value para la forma
      - p_dur: p-value para la duración
      - p_int: p-value para la interacción
    Devuelve un diccionario con 3 valores por modelo.
    """
    models = ['Threshold-based', 'Gaussian-based', 'MinimumJerk']
    pvals = {}
    for mt in models:
        df_model = agg_df[agg_df['MovementType'] == mt]
        if len(df_model) < 3:
            pvals[mt] = (np.nan, np.nan, np.nan)
        else:
            try:
                mod = ols(f"Q('{metric}') ~ C(Q('Forma del Pulso')) * C(Q('Duración (ms)'))", data=df_model).fit()
                anova_res = anova_lm(mod, typ=2)
                p_shape = anova_res.loc["C(Q('Forma del Pulso'))", "PR(>F)"] if "C(Q('Forma del Pulso'))" in anova_res.index else np.nan
                p_dur = anova_res.loc["C(Q('Duración (ms)'))", "PR(>F)"] if "C(Q('Duración (ms)'))" in anova_res.index else np.nan
                p_int = anova_res.loc["C(Q('Forma del Pulso')):C(Q('Duración (ms)'))", "PR(>F)"] if "C(Q('Forma del Pulso')):C(Q('Duración (ms)'))" in anova_res.index else np.nan
                pvals[mt] = (p_shape, p_dur, p_int)
            except Exception as e:
                logging.warning(f"Error en ANOVA para modelo {mt} y métrica {metric}: {e}")
                pvals[mt] = (np.nan, np.nan, np.nan)
    return pvals

def get_tukey_pvals_for_stimulus(agg_df, stim, metric):
    """
    Para un estímulo (stim) y una métrica dada, calcula la matriz de p‐valores
    comparando los tres MovementType usando un test de Tukey.
    Devuelve una matriz pandas (DataFrame) o None si no hay suficientes grupos.
    """
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

def add_significance_brackets(ax, pairs, p_values, box_positions,
                              y_offset=0.02, line_height=0.03, font_size=10):
    """
    Dibuja brackets de significancia en un boxplot a partir del diccionario
    de p-values (p_values) y las posiciones de cada nivel (box_positions).
    """
    y_lim = ax.get_ylim()
    h_base = y_lim[1] + (y_lim[1] - y_lim[0]) * y_offset  # posición base un poco por encima
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
        # Si p < 0.05 se añade un asterisco
        if pval < 0.05:
            p_text = f"* p={pval:.3g}"
        else:
            p_text = f"p={pval:.3g}"
        ax.text((x1+x2)*0.5, h+0.001, p_text, ha='center', va='bottom', fontsize=font_size)
# ----------------------------
# FUNCIONES DE GRÁFICOS CON FILTROS
# ----------------------------

###############################################################################
# ETAPA 3: GRÁFICOS A PARTIR DE LAS MÉTRICAS AGREGADAS (SUMMARY)
###############################################################################
def plot_summary_by_filters(aggregated_df, output_dir, day=None, coord_x=None, coord_y=None, body_part=None, title_prefix="Global Summary"):
    """
    Genera gráficos (boxplots) para cada métrica usando la tabla agregada.
    Se puede filtrar por día, coordenadas y body_part.
    Se muestran los tres boxplots (por modelo) para cada estímulo, y se fuerza que el eje y inicie en 0.
    """
    df = aggregated_df.copy()
    if day is not None:
        df = df[df["Dia experimental"] == day]
    if coord_x is not None:
        df = df[df["Coordenada_x"] == coord_x]
    if coord_y is not None:
        df = df[df["Coordenada_y"] == coord_y]
    if body_part is not None:
        df = df[df["body_part"] == body_part]
    if df.empty:
        print("No hay datos tras aplicar los filtros.")
        return

    # Extraer forma y duración
    df['Forma'] = df['Estímulo'].apply(lambda s: s.split(',')[0].strip().lower() if isinstance(s, str) and ',' in s else np.nan)
    df['Duración (ms)'] = df['Estímulo'].apply(lambda s: float(s.split(',')[1].replace(' ms','')) if isinstance(s, str) and ',' in s else np.nan)
    shape_order = ["rectangular", "rombo", "rampa descendente", "triple rombo", "rampa ascendente"]
    df["Forma"] = pd.Categorical(df["Forma"], categories=shape_order, ordered=True)

    latency_metrics = ["lat_inicio_ms", "lat_primer_pico_ms", "lat_pico_ultimo_ms", "lat_inicio_mayor_ms", "lat_pico_mayor_ms"]
    peak_metrics = ["valor_pico_inicial", "valor_pico_max"]
    other_metrics = ["dur_total_ms", "delta_valor_pico", "num_movs"]
    metrics_order = latency_metrics + peak_metrics + other_metrics

    ordered_stimuli = df.sort_values(["Forma", "Duración (ms)"])["Estímulo"].unique().tolist()

    # Definir límites globales; se fuerza el mínimo a 0
    global_latency_max = df[latency_metrics].max().max() if not df[latency_metrics].empty else None
    global_peak_max = df[peak_metrics].quantile(0.95).max() if not df[peak_metrics].empty else None

    custom_cmaps = {
        "rectangular": LinearSegmentedColormap.from_list("custom_oranges", ["#FFDAB9", "#FF8C00"]),
        "rombo": LinearSegmentedColormap.from_list("custom_blues", ["#B0C4DE", "#00008B"]),
        "rampa descendente": LinearSegmentedColormap.from_list("custom_purples", ["#E6E6FA", "#800080"]),
        "triple rombo": LinearSegmentedColormap.from_list("custom_reds", ["#FFA07A", "#8B0000"]),
        "rampa ascendente": LinearSegmentedColormap.from_list("custom_greens", ["#98FB98", "#006400"])
    }
    shape_durations = {}
    for stim in ordered_stimuli:
        shape = stim.split(',')[0].strip().lower()
        try:
            dur = float(stim.split(',')[1].replace(' ms','').strip())
        except:
            continue
        shape_durations.setdefault(shape, []).append(dur)
    shape_duration_limits = {s: (min(durs), max(durs)) for s, durs in shape_durations.items()}

    movement_types = ['Threshold-based', 'Gaussian-based', 'MinimumJerk']
    offset_dict = {'Threshold-based': 0, 'Gaussian-based': 0.5, 'MinimumJerk': 0.75}
    gap_between = 1.5

    n_cols = math.ceil(len(metrics_order) / 2)
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 5, 2 * 5), squeeze=False)
    positions_by_stim = {}

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
            for mtype in movement_types:
                data = df_stim[df_stim['MovementType'] == mtype][metric].dropna().values
                if len(data) == 0:
                    continue
                boxplot_data.append(data)
                pos = current_pos + offset_dict[mtype]
                x_positions.append(pos)
                model_positions[mtype] = pos
            if model_positions:
                center = np.mean(list(model_positions.values()))
                group_centers.append(center)
                labels.append(stim)
                positions_by_stim[stim] = model_positions
                current_pos = max(x_positions) + gap_between
            else:
                current_pos += len(movement_types) * 0.6 + gap_between

        if not boxplot_data:
            ax.text(0.5, 0.5, "Sin datos", ha='center', va='center')
            continue

        bp_obj = ax.boxplot(boxplot_data, positions=x_positions, widths=0.6/2.2,
                            patch_artist=True, showfliers=False)
        for element in ['whiskers', 'caps']:
            for line in bp_obj[element]:
                line.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for pos, patch in zip(x_positions, bp_obj['boxes']):
            diffs = {stim: abs((positions_by_stim[stim]['Threshold-based'] if 'Threshold-based' in positions_by_stim[stim] else positions_by_stim[stim][movement_types[0]]) - pos)
                     for stim in positions_by_stim}
            stim_key = min(diffs, key=diffs.get)
            shape = stim_key.split(',')[0].strip().lower()
            try:
                dur = float(stim_key.split(',')[1].replace(' ms','').strip())
            except:
                dur = 0
            min_dur, max_dur = shape_duration_limits.get(shape, (0, 1))
            norm_dur = (dur - min_dur) / (max_dur - min_dur) if (max_dur - min_dur) > 0 else 0.5
            cmap = custom_cmaps.get(shape, plt.get_cmap("Greys"))
            patch.set_facecolor(cmap(norm_dur))
            patch.set_edgecolor('black')

        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Estímulo")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        # Fijar el mínimo del eje y a 0
        if metric in latency_metrics:
            ax.set_ylim(0, global_latency_max)
        elif metric in peak_metrics:
            ax.set_ylim(0, global_peak_max)
        else:
            ax.set_ylim(0, ax.get_ylim()[1])
    
        model_pvals = compute_model_pvals(df, metric)
        subtitle_text = (
            f"Threshold: p_shape={model_pvals['Threshold-based'][0]:.3f}, p_dur={model_pvals['Threshold-based'][1]:.3f}, p_int={model_pvals['Threshold-based'][2]:.3f}\n"
            f"Gaussian: p_shape={model_pvals['Gaussian-based'][0]:.3f}, p_dur={model_pvals['Gaussian-based'][1]:.3f}, p_int={model_pvals['Gaussian-based'][2]:.3f}\n"
            f"MinJerk: p_shape={model_pvals['MinimumJerk'][0]:.3f}, p_dur={model_pvals['MinimumJerk'][1]:.3f}, p_int={model_pvals['MinimumJerk'][2]:.3f}"
        )
        ax.text(0.95, 0.95, subtitle_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        for stim in positions_by_stim:
            pval_matrix = get_tukey_pvals_for_stimulus(df, stim, metric)
            if pval_matrix is not None:
                box_positions = positions_by_stim[stim]
                pairs = list(itertools.combinations(sorted(box_positions.keys()), 2))
                add_significance_brackets(ax, pairs, pval_matrix, box_positions,
                                          y_offset=0.1, line_height=0.05, font_size=10)
    fig.suptitle(f"{title_prefix}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = sanitize_filename(f"{title_prefix.replace(' ', '_')}.png")
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] {title_prefix} guardado en: {out_path}")

###############################################################################
# FUNCIONES PARA DIFERENTES AGRUPAMIENTOS
###############################################################################
def plot_summary_by_day_coord(aggregated_df, output_dir):
    """
    Genera un summary para cada combinación de Día experimental y Coordenadas (x,y) integrando
    todos los body_parts.
    """
    for (dia, coord_x, coord_y), df_sub in aggregated_df.groupby(['Dia experimental', 'Coordenada_x', 'Coordenada_y']):
        title = f"Summary_{dia}_Coord_{coord_x}_{coord_y}"
        # No filtramos por body_part (se integran todos)
        plot_summary_by_filters(df_sub, output_dir, title_prefix=title)

def plot_global_summary(aggregated_df, output_dir):
    """
    Genera un summary global integrando todos los datos (todos los días, coordenadas y body_parts).
    """
    title = "Global_Summary"
    plot_summary_by_filters(aggregated_df, output_dir, title_prefix=title)

def plot_summary_by_bodypart(aggregated_df, output_dir):
    """
    Genera un summary para cada día y cada body_part.
    """
    for (dia, bp), df_sub in aggregated_df.groupby(['Dia experimental', 'body_part']):
        title = f"Summary_{dia}_{bp}"
        plot_summary_by_filters(df_sub, output_dir, title_prefix=title)

def plot_3d_gaussian_boxplots_by_bodypart(aggregated_df, output_dir, day=None, coord_x=None, coord_y=None):
    """
    Genera gráficos 3D interactivos (usando Plotly) para el modelo Gaussian‑based,
    mostrando únicamente la "caja central" (de Q1 a Q3) y la línea vertical que la conecta,
    junto con la línea horizontal de la mediana para cada combinación de Estímulo y body_part.
    Además, se interpola una superficie (heatmap) a partir de las medianas utilizando una escala
    de color unificada (calculada entre el 5º y el 95º percentil de las medianas) para facilitar la comparación.
    El eje Z se ajusta para incluir el rango completo de cada caja (desde el mínimo hasta el máximo).
    El orden de los estímulos se basa en la "Forma" y "Duración (ms)".
    Los colores de las cajas se asignan según 'shape_colors'.
    Los gráficos se guardan en formato HTML para su interacción.
    """
    import plotly.graph_objects as go
    from scipy.interpolate import griddata

    # Filtrar para el modelo Gaussian-based y aplicar filtros opcionales
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

    # Definir las métricas a analizar
    metrics = ["lat_inicio_ms", "lat_primer_pico_ms", "lat_pico_ultimo_ms",
               "dur_total_ms", "valor_pico_inicial", "valor_pico_max",
               "num_movs", "lat_inicio_mayor_ms", "lat_pico_mayor_ms", "delta_valor_pico"]

    # Ordenar estímulos similar a los summary
    df['Forma'] = df['Estímulo'].apply(lambda s: s.split(',')[0].strip().lower() if isinstance(s, str) and ',' in s else 'unknown')
    df['Duración (ms)'] = df['Estímulo'].apply(lambda s: float(s.split(',')[1].replace(' ms','')) if isinstance(s, str) and ',' in s else np.nan)
    shape_order = ["rectangular", "rombo", "rampa descendente", "triple rombo", "rampa ascendente"]
    df["Forma"] = pd.Categorical(df["Forma"], categories=shape_order, ordered=True)
    ordered_stimuli = df.sort_values(["Forma", "Duración (ms)"])["Estímulo"].unique().tolist()
    ordered_bodyparts = body_parts  # Usar el orden definido en body_parts_specific_colors

    stim_to_x = {stim: i for i, stim in enumerate(ordered_stimuli)}
    bp_to_y = {bp: i for i, bp in enumerate(ordered_bodyparts)}

    # Calcular, para cada combinación de Estímulo y body_part, los estadísticos (mín, Q1, mediana, Q3, máx)
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

    # Para cada métrica, construir el gráfico 3D interactivo
    for metric in metrics:
        subdf = stats_df.dropna(subset=[f"{metric}_med", f"{metric}_vmin", f"{metric}_vmax"])
        if subdf.empty:
            continue

        # Extraer posiciones y valores
        x_vals = subdf['Estímulo'].map(stim_to_x).values
        y_vals = subdf['body_part'].map(bp_to_y).values
        med_vals = subdf[f"{metric}_med"].values
        q1_vals = subdf[f"{metric}_q1"].values
        q3_vals = subdf[f"{metric}_q3"].values
        vmin_vals = subdf[f"{metric}_vmin"].values
        vmax_vals = subdf[f"{metric}_vmax"].values

        # Definir la escala de color basada en la mediana (5º-95º percentil)
        cmin = np.percentile(med_vals, 5)
        cmax = np.percentile(med_vals, 95)

        # Interpolar la superficie de la mediana
        grid_x, grid_y = np.mgrid[min(x_vals):max(x_vals):100j, min(y_vals):max(y_vals):100j]
        grid_med = griddata((x_vals, y_vals), med_vals, (grid_x, grid_y), method='cubic')

        surface_med = go.Surface(
            x=grid_x, y=grid_y, z=grid_med,
            colorscale='Viridis', opacity=0.6, cmin=cmin, cmax=cmax,
            showscale=True, name='Mediana'
        )

        # Calcular el rango global de Z usando los extremos de las cajas (vmin y vmax)
        global_zmin = np.min(q1_vals)
        global_zmax = np.max(q3_vals) * 1.05  # o agregar un incremento fijo


        traces = []
        for idx, row in subdf.iterrows():
            x = stim_to_x[row['Estímulo']]
            y = bp_to_y.get(row['body_part'], 0)
            width = 0.4
            depth = 0.4

            # Dibujar la caja central (de Q1 a Q3)
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

            # Dibujar la línea vertical que conecta Q1 y Q3 (barra unificadora)
            center_line = go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[row[f"{metric}_q1"], row[f"{metric}_q3"]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            )
            traces.append(center_line)

            # Dibujar la línea de la mediana
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
        title=f"3D Gaussian-based Boxes for {metric} (Day: {day}, Coord: {coord_x}, {coord_y})",
        scene=dict(
            xaxis=dict(
                title="Estímulo",
                tickvals=list(stim_to_x.values()),
                ticktext=list(stim_to_x.keys())
            ),
            yaxis=dict(
                title="Body Part",
                tickvals=list(bp_to_y.values()),
                ticktext=list(bp_to_y.keys())
            ),
            # Ajuste clave aquí:
            zaxis=dict(
                title=metric,
                range=[
                    min(global_zmin - 0.05 * abs(global_zmax - global_zmin), -0.05 * global_zmax), 
                    global_zmax
                ]
            )
        )
    )


        fig = go.Figure(data=[surface_med] + traces, layout=layout)
        filename = sanitize_filename(f"3D_Gaussian_Boxplot_{metric}_Day_{day}_Coord_{coord_x}_{coord_y}.html")
        out_path = os.path.join(output_dir, filename)
        fig.write_html(out_path)
        print(f"3D Gaussian Boxplot for {metric} saved at: {out_path}")














###############################################################################
# BLOQUE PRINCIPAL
###############################################################################
if __name__ == "__main__":
    logging.info("Ejecutando el bloque principal del script.")
    submov_path = os.path.join(output_comparisons_dir, 'submovement_detailed_summary.csv')
    if os.path.exists(submov_path):
        submovements_df = pd.read_csv(submov_path)
        # ETAPA 1: Agregar métricas por ensayo y ajustar latencias
        aggregated_df = aggregate_trial_metrics_extended(submovements_df)
        aggregated_csv = os.path.join(output_comparisons_dir, 'aggregated_metrics.csv')
        aggregated_df.to_csv(aggregated_csv, index=False)
        print("Tabla agregada de métricas guardada en 'aggregated_metrics.csv'")
        
        # ETAPA 2: Pruebas de hipótesis
        #do_significance_tests_aggregated(aggregated_df, output_dir=output_comparisons_dir)
        
        # ETAPA 3: Generar gráficos summary con distintos niveles de agrupación
        # 1. Por día y por body_part (cada grupo separado)
        #plot_summary_by_bodypart(aggregated_df, output_comparisons_dir)
        
        # 2. Por día y coordenadas (integrando todos los body_parts)
        #plot_summary_by_day_coord(aggregated_df, output_comparisons_dir)
        
        # 3. Global (todos los datos integrados)
        #plot_global_summary(aggregated_df, output_comparisons_dir)

        # Generar gráficos 3D para el modelo Gaussian-based, por día y coordenadas
        for (dia, coord_x, coord_y), group in aggregated_df.groupby(['Dia experimental', 'Coordenada_x', 'Coordenada_y']):
            plot_3d_gaussian_boxplots_by_bodypart(group, output_comparisons_dir, day=dia, coord_x=coord_x, coord_y=coord_y)

    else:
        print("No se encontró el archivo submovement_detailed_summary.csv para generar los análisis.")
    print("Proceso completo finalizado.")