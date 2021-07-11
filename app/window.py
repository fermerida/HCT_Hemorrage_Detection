import tkinter as tk
import tkinter.ttk as ttk
from ttkthemes import ThemedTk
from tkinter.constants import DISABLED, HORIZONTAL, RAISED
from tkinter.font import NORMAL
from tkinter import filedialog
from app import getBeggining, getEpochNumber,train, plot, predict
from compresion import comprise
import time
import threading



def print_prev():
    texto = getBeggining()
    print(texto)
    encabezado = '\n\n' + '---------------- Entorno de datos ----------------\n'
    txt_edit.insert(tk.END,encabezado)
    txt_edit.insert(tk.END,texto)

def recorrer(lista):
    resultado = ""
    contador = 1
    for x in lista:
        resultado += 'Epoch '+str(contador) +': ' + str(x) +'\n'
        contador+=1
    return resultado


def Model_train():
    
    data = train()
    stop_download()
    progress.pack_forget()
    train_loss = 'Perdida Entrenamiento:\n'+recorrer(data['loss'])
    val_loss   = 'Perdida Validación:\n'+recorrer(data['val_loss'])
    train_acc  = 'Certeza Entrenamiento:\n'+recorrer(data['accuracy'])
    val_acc    = 'Certeza Validación:\n'+recorrer(data['val_accuracy'])
    texto = train_acc +'\n'+val_acc +'\n' +train_loss +'\n'+val_loss +'\n'
    encabezado = '\n\n' + '---------------- Datos de entrenamiento ----------------\n'
    epochs = 'Numero de epochs: '+ str(getEpochNumber()) +'\n'
    txt_edit.insert(tk.END,encabezado)
    txt_edit.insert(tk.END,epochs)
    txt_edit.insert(tk.END,texto)
    btn_predict.config(state=NORMAL)
    btn_plot.config(state=NORMAL)

def Model_plot():
    plot()

def Model_predict():
    texto = predict()
    txt_edit.insert(tk.END,texto, 'warning')


def Open_dir():
    global file_path
    threading.Thread(target=start_download).start()
    #folder_path = filedialog.askdirectory()
    #print(folder_path)
    file_path =filedialog.askopenfilename()
    #txt_edit.insert(tk.END, file_path) # add this
    btn_train.config(state=NORMAL)

def step(inicio, final):
    for i in range(5):
        window.update_idletasks()
        progress['value'] += 20
        time.sleep(1)

def start_download():
    time.sleep(1)
    progress.start()   
    
def stop_download():
    time.sleep(5)
    progress.stop()
    
def compdata():
    comprise()

def quit():
    window.destroy()

def change_theme(self):
        self.style.theme_use(self.selected_theme.get())

        
window = ThemedTk(theme="elegance")
window.title("Predicción de diagnostico, Hemorragía cerebral")
window.rowconfigure(0, minsize=50, weight=1)
window.columnconfigure(1, minsize=900, weight=1)

# Add some style

 

free_space = ttk.Frame(window,relief=RAISED)
label2= ttk.Label(free_space,textvariable="Analisis",relief=RAISED)

progress = ttk.Progressbar(free_space,orient=HORIZONTAL,length=500,mode='indeterminate')
progress.pack(fill="none",expand=True)

txt_edit = tk.Text(window)
txt_edit.tag_config('warning', background="yellow", foreground="red")

fr_buttons = ttk.Frame(window,relief=RAISED)

btn_open = ttk.Button(fr_buttons, text="Seleccionar directorio",command=Open_dir)
btn_datos = ttk.Button(fr_buttons, text="Ver Datos iniciales",command=print_prev)
btn_train = ttk.Button(fr_buttons, text="Entrenar",command=Model_train,state=DISABLED)
btn_predict = ttk.Button(fr_buttons, text="Predecir resultado",command=Model_predict,state=DISABLED)
btn_plot = ttk.Button(fr_buttons, text="Graficar modelo",command=Model_plot,state=DISABLED)
btn_comp = ttk.Button(fr_buttons, text="Compresion de datos",command=compdata)
btn_exit = ttk.Button(fr_buttons, text="Salir",command=quit)

fr_buttons.grid(row=1, column=0, sticky="ns")
free_space.grid(row=0,column=0,sticky="nsew",columnspan=2)
txt_edit.grid(row=1, column=1, sticky="nsew")

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_datos.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
btn_train.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
btn_predict.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
btn_plot.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
btn_comp.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
btn_exit.grid(row=6, column=0, sticky="ew", padx=5, pady=5)

file_path = tk.StringVar()


window.mainloop()
