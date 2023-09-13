import tkinter as tk
  
# Top level window
frame = tk.Tk()
frame.title("Stimulation Interface v1")
frame.geometry('1200x800')
# Function for getting Input
# from textbox and printing it 
# at label widget
  
#def printInput():
#    inp = inputtxt.get(1.0, "end-1c")
#    lbl.config(text = "Provided Input: "+inp)
  
# TextBox Creation
    
# Label Creation
lbl = tk.Label(frame, text = "frame rate")
#lbl.place(x = 10.0, y = 20.0)
lbl.place(relx=10,rely=10)

lbl.pack()
inputtxt = tk.Text(frame, height = 10, width = 10)
inputtxt.pack()

  
# Button Creation
#printButton = tk.Button(frame, text = "Print", command = printInput)
#printButton.pack()


frame.mainloop()