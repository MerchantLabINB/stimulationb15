"""
Script para generar imágenes PNG de patrones de estimulación para presentación
Genera 4 tipos de estímulos con colores específicos:
- Rectangular: amarillo
- Rombo: azul
- Triple rombo: rojo
- Rampa ascendente: verde
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def redondear_20(num):
    """Redondea un número al múltiplo de 20 más cercano hacia arriba"""
    while num % 20 != 0:
        num += 1
    return num

def estimulo(forma="rombo", amplitud=140000000, duracion=500000, frecuencia=120, duracion_pulso=200, compensar=True):
    """
    Genera un patrón de estimulación eléctrica
    
    Parámetros:
    - forma: Forma del estímulo (rectangular, rombo, rampa ascendente, triple rombo)
    - amplitud: En nanoamperios
    - duracion: En microsegundos
    - frecuencia: En Hz
    - duracion_pulso: Duración del pulso inicial en microsegundos
    - compensar: Si se debe compensar la carga
    """
    
    proporcion = duracion / 1000000
    num_pulsos = int(frecuencia * proporcion)
    mitad = int(duracion / 2)
    duracion_pulsos_bifasicos = duracion_pulso * 2
    duracion_total_en_no_cero = duracion_pulsos_bifasicos * num_pulsos
    duracion_espacios = duracion - duracion_total_en_no_cero
    num_espacios = num_pulsos - 1
    tiempo_cada_espacio = redondear_20(int(duracion_espacios / num_espacios))
    duracion_real = (tiempo_cada_espacio * num_espacios) + duracion_total_en_no_cero
    incremento_amplitud = amplitud / (num_pulsos/2)
    incremento_amplitud = round(incremento_amplitud, 2)
    
    value = 0
    tiempo_total = 0
    suma_negativos = 0
    time = duracion_pulso
    
    lista_amplitud = []
    lista_tiempo = []
    
    if forma in ["rampa ascendente", "rampa descendente"]:
        for i in range(num_pulsos):
            if forma == "rampa ascendente":
                value = (incremento_amplitud / 2) * (i + 1)
            elif forma == "rampa descendente":
                value = amplitud - ((incremento_amplitud / 2) * i)
            
            value = int(round(value, 0))
            
            if i == num_pulsos - 1:
                lista_amplitud.extend([-value, value])
                lista_tiempo.extend([time, time])
                tiempo_total += 2 * time
                break
            
            lista_amplitud.extend([-value, value, 0])
            lista_tiempo.extend([time, time, tiempo_cada_espacio])
            tiempo_total += (2 * time + tiempo_cada_espacio)
    
    elif forma == "rectangular":
        for i in range(num_pulsos):
            if compensar:
                if i % 2 == 0:
                    pass
                else:
                    pass
            
            value = amplitud
            value = round(value, 2)
            
            lista_amplitud.append(-value)
            lista_tiempo.append(time)
            tiempo_total += time
            
            lista_amplitud.append(value)
            lista_tiempo.append(time)
            tiempo_total += time
            
            lista_amplitud.append(0)
            lista_tiempo.append(tiempo_cada_espacio)
            tiempo_total += tiempo_cada_espacio
    
    elif forma == "rombo":
        value = 0
        for i in range(num_pulsos):
            if compensar:
                if i % 2 == 0:
                    pass
                else:
                    pass
            
            if i < int(num_pulsos/2):
                value += incremento_amplitud
            elif i == int(num_pulsos/2):
                lista_amplitud.append(int(-value))
                lista_tiempo.append(time)
                tiempo_total += time
                
                lista_amplitud.append(int(value))
                lista_tiempo.append(time)
                tiempo_total += time
                
                lista_amplitud.append(0)
                lista_tiempo.append(tiempo_cada_espacio)
                tiempo_total += tiempo_cada_espacio
                continue
            else:
                value -= incremento_amplitud
            
            if value == 0:
                continue
            
            value = round(value, 2)
            
            lista_amplitud.append(int(-value))
            lista_tiempo.append(time)
            tiempo_total += time
            
            lista_amplitud.append(int(value))
            lista_tiempo.append(time)
            tiempo_total += time
            
            lista_amplitud.append(0)
            lista_tiempo.append(tiempo_cada_espacio)
            tiempo_total += tiempo_cada_espacio
    
    elif forma == "triple rombo":
        value = 0
        for i in range(num_pulsos):
            if compensar:
                if i % 2 == 0:
                    pass
                else:
                    pass
            
            if i < int(num_pulsos / 2):
                value += incremento_amplitud
            elif i == int(num_pulsos / 2):
                lista_amplitud.append(int(-value))
                lista_tiempo.append(time)
                tiempo_total += time
                
                lista_amplitud.append(int(value))
                lista_tiempo.append(time)
                tiempo_total += time
                
                lista_amplitud.append(0)
                lista_tiempo.append(tiempo_cada_espacio)
                tiempo_total += tiempo_cada_espacio
                continue
            else:
                value -= incremento_amplitud
            
            if value == 0:
                continue
            
            value = round(value, 2)
            
            lista_amplitud.append(int(-value))
            lista_tiempo.append(time)
            tiempo_total += time
            
            lista_amplitud.append(int(value))
            lista_tiempo.append(time)
            tiempo_total += time
            
            lista_amplitud.append(0)
            lista_tiempo.append(tiempo_cada_espacio)
            tiempo_total += tiempo_cada_espacio
        
        # Triplicar para triple rombo
        lista_amplitud = lista_amplitud * 3
        lista_tiempo = lista_tiempo * 3
    
    return lista_amplitud, lista_tiempo

def plot_and_save_stimulus(amplitude_list, duration_list, color, output_filename):
    """
    Grafica y guarda un patrón de estimulación
    
    Parámetros:
    - amplitude_list: Lista de amplitudes
    - duration_list: Lista de duraciones
    - color: Color de la línea
    - output_filename: Nombre del archivo de salida
    """
    # Construir coordenadas para el gráfico
    x_vals = [0]
    y_vals = [0]
    current_time = 0
    
    for amp, dur in zip(amplitude_list, duration_list):
        next_time = current_time + dur
        x_vals.extend([current_time / 1000, next_time / 1000])
        y_vals.extend([amp / 1000, amp / 1000])
        current_time = next_time
    
    # Crear figura
    plt.figure(figsize=(10, 4))
    plt.step(x_vals, y_vals, where='post', color=color, linewidth=2)
    plt.xlabel('Tiempo (ms)', fontsize=12)
    plt.ylabel('Amplitud (µA)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {output_filename}")

def main():
    """Función principal que genera las 4 imágenes"""
    
    # Crear directorio de salida si no existe
    output_dir = "/Users/brunobustos/Documents/GitHub/stimulationb15/data/plot_trials"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parámetros comunes
    frecuencia = 120  # Hz
    duracion_pulso = 200  # µs
    amplitud = 140 * 1000  # 140 µA en nanoamperios
    
    # Configuraciones para cada estímulo
    estimulos = [
        {
            'forma': 'rectangular',
            'duracion': 500000,  # 500 ms en µs
            'color': '#FFA500',  # Amarillo/naranja
            'filename': 'estimulo_rectangular.png'
        },
        {
            'forma': 'rombo',
            'duracion': 500000,  # 500 ms en µs
            'color': '#0000FF',  # Azul
            'filename': 'estimulo_rombo.png'
        },
        {
            'forma': 'triple rombo',
            'duracion': 700000,  # 700 ms en µs
            'color': '#FF0000',  # Rojo
            'filename': 'estimulo_triple_rombo.png'
        },
        {
            'forma': 'rampa ascendente',
            'duracion': 1000000,  # 1000 ms en µs
            'color': '#008000',  # Verde
            'filename': 'estimulo_rampa_ascendente.png'
        }
    ]
    
    # Generar cada estímulo
    for config in estimulos:
        print(f"\nGenerando estímulo: {config['forma']}")
        
        # Generar patrón
        amplitude_list, duration_list = estimulo(
            forma=config['forma'],
            amplitud=amplitud,
            duracion=config['duracion'],
            frecuencia=frecuencia,
            duracion_pulso=duracion_pulso,
            compensar=True
        )
        
        # Graficar y guardar
        output_path = os.path.join(output_dir, config['filename'])
        plot_and_save_stimulus(
            amplitude_list,
            duration_list,
            config['color'],
            output_path
        )
    
    print("\n¡Todas las imágenes generadas exitosamente!")

if __name__ == "__main__":
    main()
