import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import interact, FloatSlider, RadioButtons

def create_example():
    amplitude_slider = FloatSlider(min=0.1, max=1.0, step=0.1, value=0.2)
    color_buttons = RadioButtons(options=['blue', 'green', 'red'])

    @interact(amplitude=amplitude_slider, color=color_buttons)
    def plot(amplitude, color):
        fig, ax = plt.subplots(figsize=(10, 10))

        x = np.linspace(0, 10, 1000)
        ax.plot(x, amplitude * np.sin(x), color=color, lw=5)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.1, 1.1)
        plt.show()
        
        
def first_task(A):
    if A == 0:
        print("Ты ничего не сделал.")
    elif A == 5:
        print("Все верно.")
    else:
        print("A не равно 5, а равно: {0}".format(A))