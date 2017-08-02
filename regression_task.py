from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np # подгружаем дополнительную библиотеку     
        
font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)    

def plot_data():
    fig = plt.figure(figsize=(10, 8))
    k = 2
    np.random.seed(7)
    X = np.linspace(-5, 5, 10)
    y = k*X + np.random.randn(10)
    plt.scatter(X, y, color='k', marker='x', label="Реальные значения")
    plt.legend()
    plt.show()

def check_linear_function(linear_function):
    X = np.linspace(-5, 5, 10)
    k = 2
    
    y = k*X
    y_check = linear_function(k, X)
    
    if np.allclose(y, y_check):
        print("Все шикарно, молодец!")
    else:
        print("Что-то не так :(")
    
    plt.plot(X, y_check, color='k', label="Результат твоей линейной функции")
    plt.scatter(X, y, color='k', marker='x', label="Значения через которые\nфункция должна проходить")
    plt.legend()
    plt.show()


    
def check_loss_function(loss_function):
    k = 2
    np.random.seed(7)
    X = np.linspace(-5, 5, 10)
    y = k*X + np.random.randn(10)
    
    N = y.shape[0]

    real_J = np.sum( (k*X - y)**2) / (2*N)
    
    pred_J = loss_function(k=k, X=X, y=y)
    
    if np.allclose(real_J, pred_J):
        print("Все шикарно, молодец!")
    else:
        print("Что-то не так :(")
    
    print("Реальное значение ошибки: {0}".format(round(real_J, 4)))
    print("Твое значение ошибки: {0}".format(round(pred_J, 4)))
    
    

def check_gradient_function(gradient_function):
    k = 2
    np.random.seed(7)
    X = np.linspace(-5, 5, 10)
    y = k*X + np.random.randn(10)
    
    k_init = 1
    real_grad = np.mean((k_init*X - y) * X) 
    pred_grad = gradient_function(k=k_init, X=X, y=y)
    
    if np.allclose(real_grad, pred_grad):
        print("Все шикарно, молодец!")
    else:
        print("Что-то не так :(")
    
    print("Реальное значение градиента: {0}".format(round(real_grad, 4)))
    print("Твое значение градиента: {0}".format(round(pred_grad, 4)))
    
    
def check_gradient_descent(gradient_descent):
    k = 2
    np.random.seed(7)
    X = np.linspace(-5, 5, 10)
    y = k*X + np.random.randn(10)
    
    k_init = -2
    
    a = 0.03
    iters = 10
    
    k_real = k_init
    for i in range(iters):
         k_real = k_real - a*np.mean((k_real*X - y) * X) 
            
    k_pred, ks = gradient_descent(k_init=k_init, X=X, y=y, alpha=a, iters=iters)
    
    if np.allclose(k_real, k_pred):
        print("Все шикарно, молодец!")
    else:
        print("Что-то не так :(")
    
    print("Реальное значение коэффициента: {0}".format(round(k_real, 4)))
    print("Твое значение коэффициента: {0}".format(round(k_pred, 4)))
    
    fig = plt.figure(figsize=(10, 8))
    plt.plot(X, k_init * X, color='yellow', label="Результат линейной функции с начальным параметром")
    plt.plot(X, k_real * X, color='r', label="Результат хорошей линейной функции")
    plt.plot(X, k_pred * X, color='k', label="Результат твоей линейной функции")
    
    
    for k in ks:
        plt.plot(X, k * X, color='k', linestyle="dotted")
        
    plt.scatter(X, y, color='k', marker='x', label="Реальные значения")
    plt.legend()
    plt.show()    