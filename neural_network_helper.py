from IPython.display import display, Image
from ipywidgets import interact, IntSlider, FloatSlider

def forward():
    k_slider = IntSlider(min=1, max=7, step=1, value=1, description="Номер слайда")
    @interact(k=k_slider)
    def interact_plot_knn(k):
        display(Image(filename='img\\slides\\for0{0}.png'.format(k)))
        
def backword():
    k_slider = IntSlider(min=1, max=6, step=1, value=1, description="Номер слайда")
    @interact(k=k_slider)
    def interact_plot_knn(k):
        display(Image(filename='img\\slides\\back{0}.png'.format(k)))        
        
def gradient():
    k_slider = IntSlider(min=1, max=6, step=1, value=1, description="Номер слайда")
    @interact(k=k_slider)
    def interact_plot_knn(k):
        display(Image(filename='img\\slides\\grad{0}.png'.format(k)))