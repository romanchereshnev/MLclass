from IPython.display import display, Image
from ipywidgets import interact, IntSlider, FloatSlider
import  classification_helper 
import numpy as np
from classification_helper import simple_plot, check_task

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc

from os import path


def forward():
    k_slider = IntSlider(min=1, max=7, step=1, value=1, description="Номер слайда")
    @interact(k=k_slider)
    def interact_plot_knn(k):
        display(Image(filename=path.join("img", "slides", 'for0{0}.png'.format(k))))
        
def backword():
    k_slider = IntSlider(min=1, max=6, step=1, value=1, description="Номер слайда")
    @interact(k=k_slider)
    def interact_plot_knn(k):
        display(Image(filename=path.join("img", "slides", 'back{0}.png'.format(k))))
     
        
def gradient():
    k_slider = IntSlider(min=1, max=6, step=1, value=1, description="Номер слайда")
    @interact(k=k_slider)
    def interact_plot_knn(k):
        display(Image(filename=path.join("img", "slides", 'grad{0}.png'.format(k))))


font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)

def get_data_for_task():
    return classification_helper.get_data_for_task()


class NNsigm(object):
    
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_)

    def sigm(self, X):
        return 1 / (1 + np.exp(-X))
    
    def activation(self, X):
        return self.sigm(self.net_input(X))
    
    def der_sigm(self, X):
        Z = self.net_input(X)
        return self.sigm(Z)*(1 + self.sigm(Z)) 
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.5, 1, 0)
        
    def fit(self, X, Y, w=None):
        if w is None:
            np.random.seed(7)
            self.w_ = np.random.randn(X.shape[1])
        else:
            self.w_ = w.copy()
        self.cost_ = []
        
        
        for i in range(self.n_iter):
            output = self.activation(X)
            error = (output - Y)            
            der = self.der_sigm(X)
            self.w_ -= self.eta * X.T.dot((error*der))
            cost = (error**2).sum() / 2.0
            self.cost_.append(cost)    
        return self

def check_sigmoid(sigmoid, X):
    model = NNsigm()
    model.w_ = np.ones(3)
    
    y = model.sigm(model.net_input(X))
    y_pred = sigmoid(X, model.w_)
    check_task(y.ravel(), y_pred.ravel(), stuff="сигмоиды", y_tp=y.ravel(), y_pred_tp=y_pred.ravel())    


def check_der_sigmoid(der_sigm, X):
    model = NNsigm()
    model.w_ = np.ones(3)
    
    y = model.der_sigm(X)
    y_pred = der_sigm(X, model.w_)
    check_task(y.ravel(), y_pred.ravel(), stuff="производной сигмоиды", y_tp=y.ravel(), y_pred_tp=y_pred.ravel())    
    
    
    
def check_grad(grad, X, y):
    model = NNsigm()
    model.w_ = np.ones(3)
    
    error = (model.activation(X) - y)            
    der = model.der_sigm(X)
    g = X.T.dot((error*der))
    
    g_pred = grad(X, y, model.w_)
    check_task(g.ravel(), g_pred.ravel(), stuff="градиента")    

def check_backpropagation(backpropagation, X, y):
    iters = 100
    a = 0.01
    model = NNsigm(eta=a, n_iter=iters)
    
    np.random.seed(7)
    w = np.random.randn(X.shape[1])
   
    
    W_pred, costs =  backpropagation(X, y, W_init=w, iters=iters, a=a)
    
    model.fit(X, y, w)
    W_my = model.w_
    
    check_task(W_my.ravel(), W_pred.ravel(), stuff="весов")    
    
    nn = NNsigm(n_iter=5000, eta=0.001)
    nn.w_ = W_pred

    plot_cost(costs)
    plot_reg(X, y, nn)


def plot_cost(costs):
    plt.plot(range(1, len(costs) + 1), costs, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()

def plot_reg(X, y, model):
    cmap = ListedColormap(['red', 'blue'])
    x1_min, x1_max = -1, 2
    x2_min, x2_max = -1, 2
    resolution = 0.01
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    a = np.array([np.ones_like(xx2.ravel()), xx1.ravel(), xx2.ravel()])
    Z = model.predict(a.T)

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle("Границы классов", fontsize=16)
    ax.set_xlabel("Нормализованный средний радиус", fontsize=14)
    ax.set_ylabel("Нормализованная средняя гладкость", fontsize=14)
    ax.contourf(xx1, xx2, Z.reshape(xx1.shape), alpha=0.4, cmap=cmap)
    ax.scatter(X[y==0, 1], X[y==0, 2], color='red')
    ax.scatter(X[y==1, 1], X[y==1, 2], color='blue')

    plt.show()          