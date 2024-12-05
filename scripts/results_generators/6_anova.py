import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from itertools import combinations
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random

warnings.filterwarnings("ignore")

# Directorio de salida
output_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\filtered_variability_plots\extended_results'
os.makedirs(output_dir, exist_ok=True)

# Definir variables dependientes
dependent_vars = ['Latencia_al_Inicio_ms', 'Latencia_al_Pico_ms', 'Duracion_Total_ms', 'Valor_Pico_velocidad']

# Cargar y preprocesar los rangos de movimiento
movement_ranges_df_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\filtered_variability_plots\movement_ranges_summary.csv'
df = pd.read_csv(movement_ranges_df_path)

# Cargar datos experimentales
experimental_data = {
    "Día experimental": ["09/05", "15/05", "16/05", "18/05", "21/05", "22/05", "23/05", "24/05", "25/05", "28/05", "29/05"],
    "Coordenada_x": [8, 12, 4, 4, 10, 6, 14, 10, 6, 6, 12],
    "Coordenada_y": [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3],
    "Distancia al tejido (mm)": [5.5, 5.2, 4.4, 4.4, 5.4, 4.9, 5.3, 4.5, 0, 3.6, 3.8],
    "Peso (Kg)": [9, 8.7, 8.34, 8.5, 8.9, 8.9, 8.8, 8.8, 8.78, 9.04, 8.92]
}
experimental_df = pd.DataFrame(experimental_data)

# Asegurar que los nombres de columnas coincidan
if 'Día experimental' not in df.columns:
    df.rename(columns={'Dia experimental': 'Día experimental'}, inplace=True)

# Unir datos experimentales
df = df.merge(experimental_df, on="Día experimental", how="left")

# Preparar datos
df = df[df['Periodo'] == 'Durante Estímulo'].copy()
df['Latencia_al_Inicio_ms'] = df['Latencia al Inicio (s)'] * 1000
df['Latencia_al_Pico_ms'] = df['Latencia al Pico (s)'] * 1000
df['Duracion_Total_ms'] = df['Duración Total (s)'] * 1000
df.rename(columns={'Valor Pico (velocidad)': 'Valor_Pico_velocidad'}, inplace=True)
df['Forma_del_Pulso'] = df['Forma del Pulso'].astype('category')
df['Duracion_ms'] = df['Duración (ms)'].astype('category')

# Función para sanitizar nombres de archivos
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

# Función para realizar ANOVA de dos vías
def perform_two_way_anova(df, dep_var):
    model = smf.ols(f'{dep_var} ~ C(Forma_del_Pulso) * C(Duracion_ms)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

# Función para calcular estadísticas descriptivas
def calculate_descriptive_stats(df, group_by_cols, dependent_vars):
    descriptive_stats = []
    for dep_var in dependent_vars:
        grouped = df.groupby(group_by_cols)[dep_var].describe()
        grouped = grouped.reset_index()
        grouped['Variable'] = dep_var
        descriptive_stats.append(grouped)
    return pd.concat(descriptive_stats, axis=0)

# Función para generar gráficos combinados
def generate_combined_plots(day, bodypart, df_bodypart, dependent_vars, output_dir):
    # Crear subplots para las variables dependientes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, dep_var in enumerate(dependent_vars):
        ax = axes[i]
        sns.boxplot(
            x='Duracion_ms',
            y=dep_var,
            hue='Forma_del_Pulso',
            data=df_bodypart,
            ax=ax,
            palette='Set2'
        )
        # Eliminar bordes superiores y derechos
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_title(dep_var, pad=30)  # Aumentar el padding del título
        ax.set_xlabel('Duración (ms)')
        ax.set_ylabel(dep_var)
        ax.legend(title='Forma del Pulso', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)

        # Calcular el número de ensayos por categoría
        counts = df_bodypart.groupby(['Duracion_ms', 'Forma_del_Pulso'])[dep_var].count().reset_index()

        # Diccionario para rastrear alturas ocupadas por cada posición x
        height_tracker = {key: [] for key in df_bodypart['Duracion_ms'].cat.categories}

        for idx, row in counts.iterrows():
            ensayo_count = row[dep_var]
            if ensayo_count == 0:
                continue  # No mostrar si el conteo es cero

            duracion = row['Duracion_ms']
            forma = row['Forma_del_Pulso']
            x_position = (
                list(df_bodypart['Duracion_ms'].cat.categories).index(duracion) +
                0.15 * (list(df_bodypart['Forma_del_Pulso'].cat.categories).index(forma) - 0.5)
            )

            # Obtener altura base para colocar el texto
            y_base = df_bodypart[df_bodypart['Duracion_ms'] == duracion][dep_var].quantile(0.95)
            y_position = y_base + (df_bodypart[dep_var].max() * 0.02)

            # Ajustar dinámicamente para evitar solapamientos
            while y_position in height_tracker[duracion]:
                y_position += df_bodypart[dep_var].max() * 0.05

            height_tracker[duracion].append(y_position)  # Registrar altura utilizada

            ax.text(
                x=x_position,
                y=y_position,
                s=f'n={ensayo_count}',
                ha='center',
                fontsize=6,
                color='black'
            )

        # Realizar ANOVA de dos vías
        try:
            anova_table = perform_two_way_anova(df_bodypart, dep_var)
            pulse_pvalue = anova_table.loc['C(Forma_del_Pulso)', 'PR(>F)']
            duration_pvalue = anova_table.loc['C(Duracion_ms)', 'PR(>F)']
            interaction_pvalue = anova_table.loc['C(Forma_del_Pulso):C(Duracion_ms)', 'PR(>F)']
        except Exception as e:
            print(f'No se pudo realizar ANOVA para {dep_var} en {bodypart} el día {day}: {e}')
            pulse_pvalue = duration_pvalue = interaction_pvalue = np.nan

        # Agregar valores p al gráfico
        textstr = ''
        if not np.isnan(pulse_pvalue):
            textstr += f'Forma p-value: {pulse_pvalue:.4f}\n'
        else:
            textstr += 'Forma p-value: N/A\n'

        if not np.isnan(duration_pvalue):
            textstr += f'Duración p-value: {duration_pvalue:.4f}\n'
        else:
            textstr += 'Duración p-value: N/A\n'

        if not np.isnan(interaction_pvalue):
            textstr += f'Interacción p-value: {interaction_pvalue:.4f}'
        else:
            textstr += 'Interacción p-value: N/A'

        ax.text(0.02, 0.85, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Añadir título general y coordenadas
    coordenada_x = df_bodypart['Coordenada_x'].iloc[0]
    coordenada_y = df_bodypart['Coordenada_y'].iloc[0]
    plt.suptitle(f'Día: {day} - {bodypart}\nCoordenada: ({coordenada_x}, {coordenada_y})', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'Day_{day}_{bodypart}_combined_plots.png'
    plot_path = os.path.join(output_dir, sanitize_filename(plot_filename))
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico combinado guardado para el Día {day}, Bodypart: {bodypart}")




# Inicializar resultados de ANOVA
anova_results = []
descriptive_stats_all = []

for day in df['Día experimental'].unique():
    df_day = df[df['Día experimental'] == day]
    print(f'\nDía: {day}')
    for bodypart, df_bodypart in df_day.groupby('body_part'):
        print(f'\nBodypart: {bodypart}')
        print('Valores únicos de Forma_del_Pulso:', df_bodypart['Forma_del_Pulso'].unique())
        print('Valores únicos de Duracion_ms:', df_bodypart['Duracion_ms'].unique())

        # Generar gráficos combinados
        generate_combined_plots(day, bodypart, df_bodypart, dependent_vars, output_dir)

        # Realizar ANOVA y recolectar resultados
        for dep_var in dependent_vars:
            try:
                anova_table = perform_two_way_anova(df_bodypart, dep_var)
                pulse_pvalue = anova_table.loc['C(Forma_del_Pulso)', 'PR(>F)']
                duration_pvalue = anova_table.loc['C(Duracion_ms)', 'PR(>F)']
                interaction_pvalue = anova_table.loc['C(Forma_del_Pulso):C(Duracion_ms)', 'PR(>F)']
            except Exception as e:
                print(f'No se pudo realizar ANOVA para {dep_var} en {bodypart} el día {day}: {e}')
                pulse_pvalue = duration_pvalue = interaction_pvalue = np.nan

            anova_results.append({
                'Día experimental': day,
                'Bodypart': bodypart,
                'Variable': dep_var,
                'Forma p-value': pulse_pvalue,
                'Duración p-value': duration_pvalue,
                'Interacción p-value': interaction_pvalue
            })

        # Calcular estadísticas descriptivas
        descriptive_stats = calculate_descriptive_stats(df_bodypart, ['Forma_del_Pulso', 'Duracion_ms'], dependent_vars)
        descriptive_stats['Día experimental'] = day
        descriptive_stats['Bodypart'] = bodypart
        descriptive_stats_all.append(descriptive_stats)

# Guardar resultados
anova_results_df = pd.DataFrame(anova_results)
anova_results_df.to_csv(os.path.join(output_dir, 'anova_results_summary.csv'), index=False)

descriptive_stats_df = pd.concat(descriptive_stats_all, axis=0)
descriptive_stats_df.to_csv(os.path.join(output_dir, 'descriptive_stats_summary.csv'), index=False)

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# Función para realizar comparaciones de Wilcoxon con corrección de Bonferroni
def posthoc_wilcoxon(df, dependent_var, group_var, output_dir, title):
    groups = df[group_var].unique()
    pairs = list(combinations(groups, 2))
    p_values = []

    # Realizar pruebas Wilcoxon para cada par
    for g1, g2 in pairs:
        group1 = df[df[group_var] == g1][dependent_var]
        group2 = df[df[group_var] == g2][dependent_var]
        if len(group1) > 1 and len(group2) > 1:
            try:
                stat, p = wilcoxon(group1, group2)
                p_values.append((g1, g2, p))
            except ValueError:
                p_values.append((g1, g2, np.nan))
        else:
            p_values.append((g1, g2, np.nan))

    # Crear DataFrame de resultados
    p_values_df = pd.DataFrame(p_values, columns=["Group1", "Group2", "p-value"])

    # Verificar si hay al menos un valor válido antes de aplicar la corrección
    if p_values_df["p-value"].notnull().sum() > 0:
        corrected_p_values = multipletests(
            p_values_df["p-value"].dropna(), method="bonferroni")[1]
        # Asignar los valores corregidos respetando los índices originales
        p_values_df.loc[p_values_df["p-value"].notnull(), "Corrected p-value"] = corrected_p_values
    else:
        p_values_df["Corrected p-value"] = np.nan
        print(f"Advertencia: Sin valores p válidos para {title}")

    # Crear matriz para el heatmap
    heatmap_data = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for _, row in p_values_df.iterrows():
        heatmap_data.loc[row["Group1"], row["Group2"]] = row["Corrected p-value"]
        heatmap_data.loc[row["Group2"], row["Group1"]] = row["Corrected p-value"]

    # Máscara para valores NaN
    mask = heatmap_data.isnull()

    # Generar heatmap con colores diferenciados
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="coolwarm",
        cbar_kws={"label": "Corrected p-value"},
        fmt=".2g",
        vmin=0,
        vmax=1,
        square=True,
        mask=mask,
        linewidths=0.5,
        linecolor='gray'
    )

    # Añadir niveles de significancia al título
    plt.title(f"Heatmap of Corrected p-values\n{title}\n* p < 0.05, ** p < 0.01", fontsize=14)
    plt.tight_layout()

    # Guardar heatmap
    plot_path = os.path.join(output_dir, sanitize_filename(f"heatmap_{dependent_var}.png"))
    plt.savefig(plot_path)
    plt.close()
    print(f"Heatmap saved for {dependent_var} at {plot_path}")

    return p_values_df




# Añadir en el bucle de análisis para aplicar post hoc y generar heatmaps
for dep_var in dependent_vars:
    for day in df['Día experimental'].unique():
        df_day = df[df['Día experimental'] == day]
        for bodypart, df_bodypart in df_day.groupby('body_part'):
            print(f"\nPerforming Wilcoxon post hoc for {dep_var}, Day: {day}, Bodypart: {bodypart}")

            # Generar matriz combinada (duración x forma) como una sola variable
            df_bodypart["Combined_Group"] = df_bodypart["Duracion_ms"].astype(str) + "_" + df_bodypart["Forma_del_Pulso"].astype(str)

            # Aplicar pruebas post hoc y generar heatmap
            title = f"Day {day} - {bodypart} - {dep_var}"
            posthoc_results = posthoc_wilcoxon(df_bodypart, dep_var, "Combined_Group", output_dir, title)

            # Guardar los resultados en CSV
            csv_path = os.path.join(output_dir, sanitize_filename(f"posthoc_{dep_var}_{day}_{bodypart}.csv"))
            posthoc_results.to_csv(csv_path, index=False)
            print(f"Post hoc results saved at {csv_path}")

# Función para crear matriz de diferencias estadísticas y graficarla
def plot_statistical_matrix(df, dependent_var, group_var, day, bodypart, output_dir):
    # Grupos únicos
    groups = df[group_var].unique()
    pairs = list(combinations(groups, 2))
    p_values = []

    # Realizar pruebas Wilcoxon para cada par
    for g1, g2 in pairs:
        group1 = df[df[group_var] == g1][dependent_var]
        group2 = df[df[group_var] == g2][dependent_var]
        if len(group1) > 1 and len(group2) > 1:
            try:
                stat, p = wilcoxon(group1, group2)
                p_values.append((g1, g2, p))
            except ValueError:
                p_values.append((g1, g2, np.nan))
        else:
            p_values.append((g1, g2, np.nan))

    # Crear DataFrame con resultados
    p_values_df = pd.DataFrame(p_values, columns=["Group1", "Group2", "p-value"])
    if p_values_df["p-value"].notnull().sum() > 0:
        corrected_p_values = multipletests(p_values_df["p-value"].dropna(), method="bonferroni")[1]
        p_values_df.loc[p_values_df["p-value"].notnull(), "Corrected p-value"] = corrected_p_values
    else:
        p_values_df["Corrected p-value"] = np.nan
        print(f"Advertencia: Sin valores p válidos para {dependent_var} en {day} - {bodypart}")

    # Crear matriz para las diferencias estadísticas
    matrix = pd.DataFrame(index=groups, columns=groups, dtype=object)
    for _, row in p_values_df.iterrows():
        corrected_p = row["Corrected p-value"]
        if corrected_p < 0.01:
            matrix.loc[row["Group1"], row["Group2"]] = "**"
            matrix.loc[row["Group2"], row["Group1"]] = "**"
        elif corrected_p < 0.05:
            matrix.loc[row["Group1"], row["Group2"]] = "*"
            matrix.loc[row["Group2"], row["Group1"]] = "*"
        else:
            matrix.loc[row["Group1"], row["Group2"]] = "ns"
            matrix.loc[row["Group2"], row["Group1"]] = "ns"

    # Graficar la matriz
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        matrix.isin(["*", "**"]).astype(int),
        annot=matrix,
        fmt="",
        cmap="coolwarm",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        square=True,
        xticklabels=groups,
        yticklabels=groups,
    )
    plt.title(f"Statistical Differences Matrix\n{dependent_var} - {day} - {bodypart}", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    # Guardar la figura
    filename = sanitize_filename(f"{dependent_var}_{day}_{bodypart}_stat_matrix.png")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Statistical matrix plot saved: {filename}")

    return matrix


# Añadir en el bucle de análisis para generar las matrices de diferencias estadísticas
for dep_var in dependent_vars:
    for day in df['Día experimental'].unique():
        df_day = df[df['Día experimental'] == day]
        for bodypart, df_bodypart in df_day.groupby('body_part'):
            print(f"\nGenerating statistical matrix for {dep_var}, Day: {day}, Bodypart: {bodypart}")

            # Crear matriz combinada (duración x forma) como una sola variable
            df_bodypart["Combined_Group"] = df_bodypart["Duracion_ms"].astype(str) + "_" + df_bodypart["Forma_del_Pulso"].astype(str)

            # Generar y guardar la matriz
            matrix = plot_statistical_matrix(df_bodypart, dep_var, "Combined_Group", day, bodypart, output_dir)

            # Guardar matriz en CSV
            csv_path = os.path.join(output_dir, sanitize_filename(f"{dep_var}_{day}_{bodypart}_stat_matrix.csv"))
            matrix.to_csv(csv_path, index=True)
            print(f"Statistical matrix saved at: {csv_path}")
