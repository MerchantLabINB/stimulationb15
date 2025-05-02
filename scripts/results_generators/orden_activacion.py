import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al archivo CSV y carpeta de salida
ruta_csv = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\plot_trials4mad\aggregated_df_enriched.csv'
output_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\boxplots_latencia_por_sitio'
os.makedirs(output_dir, exist_ok=True)

# Cargar y preparar los datos
df = pd.read_csv(ruta_csv)
df = df[df["MovementType"] == "Gaussian-based"].copy()
df["body_part"] = df["body_part"].str.capitalize().replace({"Muneca": "Muñeca"})
df["lat_pico_mayor_ms"] = pd.to_numeric(df["lat_pico_mayor_ms"], errors='coerce')
df = df[df["body_part"].isin(["Hombro", "Codo", "Muñeca"])]

# --- NUEVO FILTRADO ---
df["Forma_del_Pulso"] = df["Estímulo"].str.split(",", n=1).str[0].str.strip().str.lower()
df["Duracion_ms"] = df["Estímulo"].str.extract(r'(\d+)').astype(float)
df = df[(df["Forma_del_Pulso"] == "rectangular") & (df["Duracion_ms"] == 1000)]

# Configuración visual
color_dict = {"Hombro": "gold", "Codo": "darkorange", "Muñeca": "purple"}
sns.set(style="whitegrid", font_scale=1.2)

# Escala Y común para todos los días (percentil 10–90)
y_min = df["lat_pico_mayor_ms"].quantile(0.10)
y_max = df["lat_pico_mayor_ms"].quantile(0.90)

# Agrupar por día
for dia, sub in df.groupby("Dia experimental"):
    plt.figure(figsize=(6, 6))

    # Orden de partes por mediana
    orden = (
        sub.groupby("body_part")["lat_pico_mayor_ms"]
        .median()
        .sort_values()
        .index.tolist()
    )

    ax = sns.boxplot(
        data=sub,
        x="body_part",
        y="lat_pico_mayor_ms",
        order=orden,
        palette=color_dict,
        showfliers=False,
        width=0.6,
        linewidth=0,
        boxprops=dict(linewidth=0),
        whiskerprops=dict(linewidth=0),
        capprops=dict(linewidth=0),
        medianprops=dict(color='silver', linewidth=3)
    )

    # Línea de medianas por encima
    medianas = sub.groupby("body_part")["lat_pico_mayor_ms"].median().reindex(orden)
    x_vals = list(range(len(orden)))
    y_vals = medianas.values
    ax.plot(x_vals, y_vals, color='black', linestyle='--', marker='o', zorder=10)

    # Agregar números de orden encima de cada boxplot
    # Agregar números grandes justo encima de cada boxplot
    for i, body in enumerate(orden):
        # Obtener valor máximo del boxplot de ese grupo
        grupo = sub[sub["body_part"] == body]["lat_pico_mayor_ms"].dropna()
        q3 = grupo.quantile(0.75)
        y_pos = q3 + (y_max - y_min) * 0.03  # apenas encima de la caja

        ax.text(i, y_pos, str(i + 1),
                ha='center', va='bottom',
                fontsize=22, fontweight='bold',
                color='black', zorder=11)


    # Coordenada para título
    coord_x = sub["Coordenada_x"].iloc[0]
    coord_y = sub["Coordenada_y"].iloc[0]
    titulo = f"Coordenada: ({coord_x}, {coord_y})"

    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel("Parte del cuerpo")
    ax.set_ylabel("Latencia al pico mayor (ms)")
    ax.set_ylim(y_min, y_max * 1.15)
    ax.set_xticklabels(orden)

    plt.tight_layout()

    # Guardar
    filename = f"latencia_pico_mayor_{dia.replace('/', '-')}_coord_{coord_x}_{coord_y}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, transparent=True)
    plt.close()
    print(f"[✓] Guardado: {filepath}")
