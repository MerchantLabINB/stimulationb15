# Stimulation.py - Script para controlar la estimulación eléctrica mediante un dispositivo STG200x.
import time
import datetime
import os
import clr

from System import Action
from System import *
from System import Array, UInt32, UInt64, Int32

clr.AddReference('C:\\Users\\samae\\Documents\\GitHub\\McsUsbNet_Examples\\McsUsbNet\\x64\\\McsUsbNet.dll')
from Mcs.Usb import CMcsUsbListNet, DeviceEnumNet, CStg200xDownloadNet, McsBusTypeEnumNet, STG_DestinationEnumNet

import tkinter as tk
from tkinter import ttk
from queue import Queue
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
from threading import Thread, Event
import threading

# Defining variables
device = None  # Define device at the top level of your script
device_disconnected = False
plot_queue = Queue()
pause_event = Event()

def update_plot():
    try:
        if not plot_queue.empty():
            amplitude_list, duration_list = plot_queue.get()
            plot_stimuli(amplitude_list, duration_list,plot_frame)
        root.after(100, update_plot)
    except Exception as e:
        print("Error updating plot:", e)
"""
def handle_plots():
    while True:
        plot_data = plot_queue.get()
        if plot_data is None:
            break
        amplitude_list, duration_list = plot_data
        plot_stimuli(amplitude_list, duration_list)
"""
def redondear_20 (num):
  while num % 20 != 0:
    num += 1
  return num

def pause_stimulation():
    pause_event.set()

def resume_stimulation():
    pause_event.clear()

def estimulo (forma = "rombo", amplitud = 140000000, duracion = 500000, frecuencia = 120,duracion_pulso = 200,compensar = True):
  
    # forma: Forma del estimulo (rectangular o rombo)
    # amplitud: En nanoamperios
    # duracion: En microsegundos
    # frecuencia: En Hz
    # duracion_pulso: Duración del pulso inicial en microsegundos **Se duplica en el bifasico
    print(f"Forma del estimulo: {forma}")
    print(f"Amplitud (en nanoamperios): {amplitud}")
    print(f"Duracion (en microsegundos): {duracion}")
    print(f"Frecuencia (en Hz): {frecuencia}")
    print(f"duracion_pulso: Duración del pulso inicial en microsegundos **Se duplica en el bifasico: {duracion_pulso}")

    #########
    # Ahora si hacemos los calculos necesarios para la construcción del patron:

    proporcion = duracion / 1000000 # Proprcion del tiempo para calcular el numero de pulsos Ej: Medio segundo es 0.5
    print('\n\nProporcion: '+str(proporcion))

    # Este valor es importante para la realización del for
    num_pulsos = int(frecuencia * proporcion) # Numero de pulsos bifasicos a graficar
    print ('Numero de pulsos: ' + str(num_pulsos))

    mitad = int (duracion / 2) # Mitad del tren  en microsegunods
    print('Mitad del tren en ms: ' + str(mitad)) # Esto para determinar la punta del estimulo

    duracion_pulsos_bifasicos = duracion_pulso * 2
    print('Duracion pulsos bifasicos:',duracion_pulsos_bifasicos)

    duracion_total_en_no_cero = duracion_pulsos_bifasicos * num_pulsos
    print('Tiempo total en no cero (en microsegundos):',duracion_total_en_no_cero)

    duracion_espacios = duracion - duracion_total_en_no_cero  # Duracion de los espacios entre pulsos
    print('Duracion total de espacios entre pulsos:',duracion_espacios)

    num_espacios = num_pulsos - 1
    print('Numero de espacios:',num_espacios)


    tiempo_cada_espacio = redondear_20(int(duracion_espacios / num_espacios)) # Redondeamos el valor a uno divisible por 20 microsegundos
    print('Tiempo de cada espacio entre pulsos sin redondear en microsegundos: ',tiempo_cada_espacio) # En microsegundos # Ahora si se lo está redondeando

    #tiempo_cada_espacio = int(round(tiempo_cada_espacio,0)) # Redondeado
    #print('Tiempo de cada espacio entre pulsos: ',tiempo_cada_espacio) # En microsegundos

    # Con esto podemos calcular el tiempo real del estimulo, ya que vamos a redondearlo:
    duracion_real = (tiempo_cada_espacio * num_espacios) + duracion_total_en_no_cero # Ej: (8068 * 59) + 24000 = 500012
    print('Duracion calculada del estimulo: ', duracion_real)

    ####### El incremento de la amplitud se utiliza en el elif de diamante
    incremento_amplitud = amplitud / (num_pulsos/2) # Cuanto incrementa el amperaje en cada pulso
    incremento_amplitud = round(incremento_amplitud,2)
    print('Incremento amplitud: ',incremento_amplitud)

    ######### Definiendo los valores a cambiar

    value = 0 # Valor de la amplitud cambiante
    tiempo_total = 0 # El tiempo sumado
    suma_negativos = 0 # Suma de espacios negativos intercaladas

    time = duracion_pulso
    
    ######### Estableciendo las listas a devolver
    lista_amplitud = []
    lista_tiempo = []

    if forma in ["rampa ascendente", "rampa descendente"]:
      for i in range(num_pulsos):
          
          if forma == "rampa ascendente":
              value = (incremento_amplitud/2) * (i+1)  # Linear increment
          elif forma == "rampa descendente":
              value = amplitud - ((incremento_amplitud/2) * (i))  # Linear decrement

          value = int(round(value, 0))  # Ensure the value is an integer

          if i == range(num_pulsos)[-1]:  # If it's the last pulse, don't add a zero period
              lista_amplitud.extend([-value, value])
              lista_tiempo.extend([time, time])
              tiempo_total += 2 * time
              break

          # Append negative and positive phases of the biphasic pulse
          lista_amplitud.extend([-value, value, 0])
          lista_tiempo.extend([time, time, tiempo_cada_espacio])

          tiempo_total += (2 * time + tiempo_cada_espacio)  # Update the total time to include both phases and the zero period



    if forma == "rectangular":
      for i in range(num_pulsos):
        #print (i)
        #pass

        if compensar:

          if i % 2 == 0: # si el valor es par se reducen 20
            tiempo_cada_espacio -= 20
            suma_negativos += 1
          else: # Si el valor es impar se suman 20
            tiempo_cada_espacio += 20

        value = amplitud
        #

        value = round(value,2)

        # f.write('-' +str(value)+ '\t' + str(time) + '\n')
        lista_amplitud.append(-value)
        lista_tiempo.append(time)

        tiempo_total += time

        # f.write(str(value)+ '\t' + str(time) + '\n')
        lista_amplitud.append(value)
        lista_tiempo.append(time)

        tiempo_total += time

        # f.write(str(0)+ '\t' + str(tiempo_cada_espacio) + '\n')
        lista_amplitud.append(0)
        lista_tiempo.append(tiempo_cada_espacio)

        tiempo_total += tiempo_cada_espacio
    
    elif forma == "rombo":
      # Deprecated(Era otra orma de abrir el archivo):
      # f = open('patron_' + forma + str(int(duracion/1000)) + '.dat','w')
      value = 0

      for i in range(num_pulsos):
        #print (i)
        #pass

        if compensar:

          if i % 2 == 0: # si el valor es par se reducen 20
            tiempo_cada_espacio -= 20
            suma_negativos += 1
          else: # Si el valor es impar se suman 20
            tiempo_cada_espacio += 20

        # Este valor nos ayuda a subir el valor en un patron de estimulación rombo
        if i < int(num_pulsos/2): #

          value += incremento_amplitud

        elif i == int(num_pulsos/2): #Esto es para que el estimulo central se repita, ej cuando la amplitud llega a 140
          #print(i, num_pulsos/2)
          #f.write('-' +str(value)+ '\t' + str(time) + '\n')
          lista_amplitud.append(int(-value))
          lista_tiempo.append(time)

          tiempo_total += time

          #f.write(str(value)+ '\t' + str(time) + '\n')
          lista_amplitud.append(int(value))
          lista_tiempo.append(time)

          tiempo_total += time

          #f.write(str(0)+ '\t' + str(tiempo_cada_espacio) + '\n')
          lista_amplitud.append(0)
          lista_tiempo.append(tiempo_cada_espacio)
          tiempo_total += tiempo_cada_espacio
          continue # Para evitar que el pulso central se  repita 3 veces (solo 2 esta bien por simetría)

        else:
          value -=incremento_amplitud

        if value == 0: # tambien sirve para iniciar en 0 el diamante
          continue # Para que al final del tren de estimulación no haya un espacio en cero extra

        value = round(value,2)

        # Deprecated
        time = duracion_pulso

        # f.write('-' +str(value)+ '\t' + str(time) + '\n')
        lista_amplitud.append(int(-value))
        lista_tiempo.append(time)

        tiempo_total += time

        # f.write(str(value)+ '\t' + str(time) + '\n')
        lista_amplitud.append(int(value))
        lista_tiempo.append(time)

        tiempo_total += time

        # Deprecated
        #if (i+1) == num_pulsos:
        #  continue # Esto es para evitar que se haga la linea de ammplitud 0 del final

        # f.write(str(0)+ '\t' + str(tiempo_cada_espacio) + '\n')
        lista_amplitud.append(0)
        lista_tiempo.append(tiempo_cada_espacio)
        tiempo_total += tiempo_cada_espacio

    elif forma == "rombo" or forma == "triple rombo":
        value = 0
        for i in range(num_pulsos):
            if compensar:
                if i % 2 == 0:
                    tiempo_cada_espacio -= 20
                    suma_negativos += 1
                else:
                    tiempo_cada_espacio += 20

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

        if forma == "triple rombo":
            lista_amplitud = lista_amplitud * 3
            lista_tiempo = lista_tiempo * 3

    print('Tiempo sumado', tiempo_total)
    print('Suma de espacios negativos intercalados:', suma_negativos, "\n")
    return lista_amplitud, lista_tiempo    

def plot_stimuli(amplitude_list, duration_list, frame):
    
    ax.clear()  # Clear previous plot

    # Inicializar el primer punto
    x_vals = [0]  # El tiempo se mostrará en milisegundos
    y_vals = [0]  # La amplitud se mostrará en microamperios
    current_time = 0  # Mantener el tiempo en microsegundos para un cálculo preciso

    # Construir x_vals y y_vals con conversión
    for amp, dur in zip(amplitude_list, duration_list):
        next_time = current_time + dur  # dur está en microsegundos
        # Añadir el tiempo de inicio y fin de cada amplitud, convertir el tiempo a milisegundos
        x_vals.extend([current_time / 1000, next_time / 1000])
        # Extender la amplitud actual (convertir a μA) y luego saltar a la siguiente amplitud
        y_vals.extend([amp / 1000, amp / 1000])  # Convertir nanoamperios a microamperios
        current_time = next_time
    
    ax.step(x_vals, y_vals, where='post')
    ax.set_title('Stimulation Pattern')
    ax.set_xlabel('Time (milliseconds)')
    ax.set_ylabel('Amplitude (microamperes)')

    # Embedding the plot into the Tkinter window
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    """""
    # Trazado
    plt.figure(figsize=(12, 6))
    plt.step(x_vals, y_vals, where='post')
    plt.title('Patrón de Estimulación')
    plt.xlabel('Tiempo (milisegundos)')  # Actualizar etiqueta a milisegundos
    plt.ylabel('Amplitud (microamperios)')  # Actualizar etiqueta a microamperios
    plt.grid(True)
    plt.show()
    """

def PollHandler(status, stgStatusNet, index_list):
    print('%x %s' % (status, str(stgStatusNet.TiggerStatus[0])))

def connect_device():
  global device,connect_button
  deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)

  print("found %d devices" % (deviceList.Count))

  for i in range(deviceList.Count):
      listEntry = deviceList.GetUsbListEntry(i)
      print("Device: %s   Serial: %s" % (listEntry.DeviceName,listEntry.SerialNumber))

  device = CStg200xDownloadNet();
  device.Stg200xPollStatusEvent += PollHandler;
  device.Connect(deviceList.GetUsbListEntry(0))

  voltageRange = device.GetVoltageRangeInMicroVolt(0);
  voltageResulution = device.GetVoltageResolutionInMicroVolt(0);
  currentRange = device.GetCurrentRangeInNanoAmp(0);
  currentResolution = device.GetCurrentResolutionInNanoAmp(0);

  print('Voltage Mode:  Range: %d mV  Resolution: %1.2f mV' % (voltageRange/1000, voltageResulution/1000.0))
  print('Current Mode:  Range: %d uA  Resolution: %1.2f uA' % (currentRange/1000, currentResolution/1000.0))
  
  connect_button.config(state='disabled')


def save_stimulation_data(forma, amplitud, duracion, frecuencia, duracion_pulso, amplitude_list, duration_list):
    """
    Saves stimulation data to a timestamped file.

    Parameters:
    - forma: The shape of the stimulus.
    - amplitud: The amplitude of the stimulus in nanoamperes.
    - duracion: The duration of the stimulus in microseconds.
    - frecuencia: The frequency of the stimulus in Hz.
    - duracion_pulso: The pulse duration in microseconds.
    - amplitude_list: List of amplitude values.
    - duration_list: List of duration values.
    """
    directory = "C:\\Users\\samae\\Documents\\GitHub\\GUI_pattern_generator\\data"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory}\\stimulus_data_{timestamp}.txt"
    
    with open(filename, "w") as file:
        file.write(f"Stimulus Preparation Time: {timestamp}\n")
        file.write(f"Forma: {forma}, Amplitud: {amplitud}, Duración: {duracion}, Frecuencia: {frecuencia}, Duración del Pulso: {duracion_pulso}\n")
        file.write("Amplitude List:\n")
        file.write(str(amplitude_list))
        file.write("\nDuration List:\n")
        file.write(str(duration_list))
    
    print(f"Data saved to {filename}")

def prepare_data():
  forma = forma_var.get()
  amplitud = 1000*int(amplitud_var.get()) # Se multiplica por 1000 para convertir a nanoamperios
  duracion = 1000*int(duracion_var.get()) # Se multiplica por 1000 para convertir a microsegundos
  frecuencia = int(frecuencia_var.get())
  duracion_pulso = int(duracion_pulso_var.get())
  #print ("amplitud: ",amplitud)
  compensation = True
  if duracion == 1000000 or duracion == 100000:
    compensation = False

  print ("Compensation: ",compensation)
  amplitude_list, duration_list = estimulo(forma=forma, amplitud=amplitud, duracion=duracion, frecuencia=frecuencia,duracion_pulso=200,compensar=True)
  # print(amplitude_list)

  global amplitude, duration
  amplitude = Array[Int32](amplitude_list);
  duration = Array[UInt64](duration_list); # in us (microseconds)

  save_stimulation_data(forma, amplitud, duracion, frecuencia, duracion_pulso, amplitude_list, duration_list)
  plot_queue.put((amplitude_list, duration_list))

  #plot_stimuli(amplitude_list, duration_list,plot_frame)



def prepare_stimulation():
  channelmap = Array[UInt32]([1,0,0,0])
  syncoutmap = Array[UInt32]([1,0,0,0])
  repeat = Array[UInt32]([1,0,0,0])

  device.SetupTrigger(0, channelmap, syncoutmap, repeat)
  device.SetCurrentMode();
  device.PrepareAndSendData(0, amplitude, duration, STG_DestinationEnumNet.channeldata_current)
  print("Data prepared")

def start_stimulation():
  device.SendStart(1)
  print("PULSO ENVIADO")



def close_device():
    global device  # Use the global keyword to refer to the global device
    if device is not None:
        device.Disconnect()

def on_closing():
    global device  # Use the global keyword to refer to the global device
    if device is not None:
        device.Disconnect()
    root.destroy()

def generate_patterns():
    # Define the patterns with respective durations
    patterns = [
        ("rectangular", 500), ("rectangular", 1000), ("rombo", 500),
        ("rombo", 750), ("rombo", 1000), ("rampa ascendente", 1000), ("rombo triple", 700)
    ]
    random.seed(13)  # Set the seed for reproducibility
    random.shuffle(patterns)  # Shuffle the patterns randomly
    print("Tamaño patterns",len(patterns) )
    print("Patterns: ",patterns)
    return patterns

def run_stimulation():
    for shape, duration in generate_patterns():
        print("Check pause state: paused =", not pause_event.is_set())
        pause_event.wait()  # Should pause here if event is set
        print("Starting stimulation")
        amplitude = 1000 * int(amplitud_var.get())  # Using GUI input
        frequency = int(frecuencia_var.get())
        prepare_and_start_stimulation(shape, amplitude, duration, frequency)
        start_stimulation()
        
        tiempo_interestimulo = random.randint(4, 7)  # ISI between 5 and 7 seconds
        print(f"Interstimulus interval: {tiempo_interestimulo} seconds")
        time.sleep(tiempo_interestimulo)
    
    start_stimulation_button.config(state='disabled')


def prepare_and_start_stimulation(shape, amplitude, duracion, frequency):
    amplitude_list, duration_list = estimulo(forma=shape, amplitud=amplitude, duracion=duracion*1000, frecuencia=frequency, duracion_pulso=200, compensar=True)
    amplitude_array = Array[Int32](amplitude_list)
    duration_array = Array[UInt64](duration_list)
    duracion_pulso = int(duracion_pulso_var.get())

    compensation = True
    
    if duracion == 1000 or duracion == 100:
      compensation = False

    print ("Compensation: ",compensation)
    
    save_stimulation_data(shape, amplitude, duracion, frequency, duracion_pulso, amplitude_list, duration_list)

    # Send plot data to the main thread via queue
    plot_queue.put((amplitude_list, duration_list))
    # Assuming device and other setup is correct
    device.PrepareAndSendData(0, amplitude_array, duration_array, STG_DestinationEnumNet.channeldata_current)
    print(f"Stimulation prepared for {shape} with {amplitude} nA, {duracion} ms, {frequency} Hz")

if __name__ == "__main__":
    # GUI and window initialization code
    root = tk.Tk()
    root.title("Stimulation Control")

    controls_frame = tk.Frame(root)
    controls_frame.pack(fill=tk.BOTH, expand=True)

    # Define a frame for plotting
    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True)
    pause_event.clear()

    # Initialize plot area
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    ax = fig.add_subplot(111)

    # Variables for GUI elements
    forma_var = tk.StringVar(value="rectangular")
    amplitud_var = tk.StringVar(value="15")
    duracion_var = tk.StringVar(value="50")
    frecuencia_var = tk.StringVar(value="120")
    duracion_pulso_var = tk.StringVar(value="200")

    # Define label and option lists
    labels = ["Forma:", "Amplitud (5-140 µA):", "Duracion (50-1000 ms):", "Frecuencia (Hz):", "Duracion del pulso (µs):"]
    variables = [forma_var, amplitud_var, duracion_var, frecuencia_var, duracion_pulso_var]
    options = [
        ["rectangular", "rombo", "rampa ascendente", "rampa descendente", "triple rombo"],
        [str(x) for x in range(5, 141, 5)],
        ["50", "100", "300", "500", "700", "750", "1000"],
        ["200", "120", "100", "60", "30", "15"],
        ["200"]
    ]

    # Create GUI elements dynamically
    for i, (label, var, opt) in enumerate(zip(labels, variables, options)):
        tk.Label(controls_frame, text=label).grid(row=i, column=0, sticky='w')
        ttk.Combobox(controls_frame, textvariable=var, values=opt).grid(row=i, column=1)

    # Buttons
    connect_button = tk.Button(controls_frame, text="Connect", command=connect_device)
    connect_button.grid(row=5, column=0)

    prepare_data_button = tk.Button(controls_frame, text="Prepare Data", command=prepare_data)
    prepare_data_button.grid(row=5, column=1)

    prepare_stimulation_button = tk.Button(controls_frame, text="Prepare Stimulation", command=prepare_stimulation)
    prepare_stimulation_button.grid(row=5, column=2)

    start_stimulation_button = tk.Button(controls_frame, text="Start Stimulation", command=start_stimulation)
    start_stimulation_button.grid(row=5, column=3)

    disconnect_button = tk.Button(controls_frame, text="Disconnect", command=close_device)
    disconnect_button.grid(row=5, column=4)

    start_sequence_button = tk.Button(controls_frame, text="Prepare Random Sequence", command=lambda: Thread(target=run_stimulation).start())
    start_sequence_button.grid(row=6, column=0)

    pause_button = tk.Button(controls_frame, text="Unpause Stimulation", command=pause_stimulation)
    pause_button.grid(row=6, column=1)

    resume_button = tk.Button(controls_frame, text="Pause Stimulation", command=resume_stimulation)
    resume_button.grid(row=6, column=2)
    resume_button.config(bg='red')

    plot_queue = Queue()
    root.after(100, update_plot)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
