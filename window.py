import tkinter as tk
from tkinter.ttk import *
from tkinter import filedialog
from app import getBeggining,train, plot, predict





def print_prev():
    texto = getBeggining()
    print(texto)
    txt_edit.insert(tk.END,texto)


def Model_train():
    train()

def Model_plot():
    plot()

def Model_predict():
    predict()

def Open_dir():
    global folder_path
    folder_path = filedialog.askdirectory()
    print(folder_path)


window = tk.Tk()
window.title("Text Editor Application")
window.rowconfigure(0, minsize=50, weight=1)
window.columnconfigure(1, minsize=500, weight=1)

# Create style Object
style = Style()
 
#style.configure('TButton', font =
 #              ('calibri', 10, 'bold', 'underline'),
  #              foreground = 'red')
 
# Changes will be reflected

 

free_space = tk.Frame(window, relief=tk.RAISED, bd=2)


txt_edit = tk.Text(window)
fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)

btn_open = tk.Button(fr_buttons, text="Seleccionar directorio",command=Open_dir)
btn_datos = tk.Button(fr_buttons, text="Ver Datos iniciales",command=print_prev)
btn_train = tk.Button(fr_buttons, text="Entrenar",command=Model_train)
btn_predict = tk.Button(fr_buttons, text="Predecir resultado",command=Model_predict)
btn_plot = tk.Button(fr_buttons, text="Graficar modelo",command=Model_plot)
btn_exit = tk.Button(fr_buttons, text="Salir",command=Model_plot)

fr_buttons.grid(row=1, column=0, sticky="ns")
free_space.grid(row=0,column=1,sticky="ew")
txt_edit.grid(row=1, column=1, sticky="nsew")

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_datos.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
btn_train.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
btn_predict.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
btn_plot.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
btn_exit.grid(row=5, column=0, sticky="ew", padx=5, pady=5)

folder_path = tk.StringVar()


window.mainloop()
