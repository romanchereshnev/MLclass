from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
import warnings
from sklearn import linear_model
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.svm import SVC
        
font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)

def get_data():
    iris = datasets.load_iris()

    indexes = np.array(list(range(25)) + list(range(50, 75)) + list(range(100, 125)))

    data = iris.data[:, [2, 3]]
        
    X = data[indexes]
    y = iris.target[indexes]
    
    indexes_unseen = [26, 55, 130]

    X_unseen = data[indexes_unseen]
    y_unseen = iris.target[indexes_unseen]
    
    return X, y, X_unseen, y_unseen

def plot_iris(X, y, X_unseen=None, classifier=None, same_range=False):

    fig = plt.figure(figsize=(12, 8))
    
    if classifier:
        resolution=0.01
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min= X[:, 0].min() - 0.2*X[:, 0].max()
        x2_min= X[:, 1].min() - 0.2*X[:, 1].max()
        
        x1_max = X[:, 0].max() + 0.2*X[:, 0].max()
        x2_max = X[:, 1].max() + 0.2*X[:, 1].max()
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
       ##plt.xlim(x1_min, x1_max)
       ## №plt.ylim(x2_min, x2_max)
    
    for col, l, leg, in zip(['red', 'blue', 'green'], np.unique(y), ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']):
        plt.scatter(X[y==l, 0], X[y==l, 1], c=col, label=leg)
      
    if X_unseen is not None:
        plt.scatter(X_unseen[0, 0], X_unseen[0, 1], c='yellow', alpha=1.0, edgecolor='black', linewidths=2, marker='o', s=100)
        plt.scatter(X_unseen[1, 0], X_unseen[1, 1], c='yellow', alpha=1.0, edgecolor='black', linewidths=2, marker='^', s=100)
        plt.scatter(X_unseen[2, 0], X_unseen[2, 1], c='yellow', alpha=1.0, edgecolor='black', linewidths=2, marker='*', s=100)
        
    plt.legend(loc='best')
    plt.xlabel("Длина лепестка", fontsize=16)
    plt.ylabel('Ширина лепестка', fontsize=16)
    
    if same_range:
        ma = max([max(X[:, 0]), max(X[:, 1])]) + 1
        mi = min([min(X[:, 0]), min(X[:, 1])]) - 1
        
        plt.xlim(mi, ma)
        plt.ylim(mi, ma)
    
    plt.show()
    
def normilize_data(X, X_u):
    
    def normalize(d, mi, ma):
        return (d - mi) / (ma - mi)

    X_norm = np.zeros_like(X)
    X_unseen_norm = np.zeros_like(X_u)

    X_norm[:, 0] = normalize(X[:, 0], min(X[:, 0]), max(X[:, 0]))
    X_norm[:, 1] = normalize(X[:, 1], min(X[:, 1]), max(X[:, 1]))

    X_unseen_norm[:, 0] = normalize(X_u[:, 0], min(X[:, 0]), max(X[:, 0]))
    X_unseen_norm[:, 1] = normalize(X_u[:, 1], min(X[:, 1]), max(X[:, 1]))
    
    return X_norm, X_unseen_norm    

def choose_knn(X, y, X_u):
    k_slider = IntSlider(min=1, max=10, step=1, value=1, description="# соседей")

    @interact(k=k_slider)
    def interact_plot_knn(k):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X, y) 
        plot_iris(X, y, X_unseen=X_u, classifier=neigh)
        
def Tree(X, y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    plot_iris(X, y, classifier=clf)


def SVM(X, y):
    clf = SVC()
    clf.fit(X, y)
    plot_iris(X, y, classifier=clf)
        
def sigm(z):
    return 1.0 / (1.0 + np.exp(-z))

def interact_plot_sigmoid():
    theta0_sl = FloatSlider(min=-2, max=2, step=.1, value=0, description="$\\theta_0$ =")
    theta1_sl = FloatSlider(min=-2, max=2, step=.1, value=.1, description="$\\theta_1$ =")

    @interact(theta0=theta0_sl, theta1=theta1_sl)
    def interact_sigmoid(theta1, theta0):
        plot_sigmoid(theta1, theta0, plot_data=False)
        
def plot_sigmoid(theta1, theta0, plot_data=False):
    z = theta1*np.arange(-7, 7, 0.1) + theta0 
    phi_z = sigm(z)
    plt.plot(np.arange(-7, 7, 0.1), phi_z)
    
    plt.axvline(0.0, color='k')

    plt.ylim(-0.1, 1.1)
    plt.xlim(-6, 6)
    plt.xlabel('x')
    plt.ylabel('$\phi (\\theta_1x + \\theta_0)$')
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.yticks([0.0, 0.5, 1.0])

    if plot_data:
        np.random.seed(7)
        X = np.concatenate([np.linspace(-4, -1, 10) + np.random.rand(10), np.linspace(0, 4, 10) + np.random.rand(10)])
        y = np.concatenate([np.zeros(10,) , np.ones(10,)])
        plt.axhline(0.5, color='red')
        plt.scatter(X, y)
        plt.plot([X[9], X[9]], [0, sigm(X[9])], linestyle='dashed', color='k')
        plt.plot([X[11], X[11]], [1, sigm(X[11])], linestyle='dashed', color='k')
        plt.xlabel('z')
        plt.ylabel('$\phi (z)$')

    plt.tight_layout()
    plt.show()
  

def J_simple(theta0, theta1, X, y):
    return np.mean( (sigm(theta1*X + theta0) - y)**2 )

def J(theta0, theta1, X, y):
    pred = sigm(theta1*X + theta0)
    return np.mean(-y*np.log(pred) - (1 - y)*np.log(1 - pred))
    

def plot_simple_error(loss=J_simple):  
    np.random.seed(7)
    X = np.concatenate([np.linspace(-4, -1, 10) + np.random.rand(10), np.linspace(0, 4, 10) + np.random.rand(10)])
    y = np.concatenate([np.zeros(10,) , np.ones(10,)])
    
    angles1 = IntSlider(min=0, max=180, step=1, value=0, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=0, description='Горизонтальное')
        
    th1 = np.linspace(-15, 15, 30)
    th0 = np.linspace(-15, 15, 30)
    th1, th0 = np.meshgrid(th1, th0)
    
    Js = np.zeros_like(th0)
    for i in range(len(th0)):
        for j in range(len(th0)):
            Js[i, j] = loss(th0[i, j], th1[i, j], X, y)


    @interact(angle1=angles1, angle2=angles2)    
    def interact_simple_error(angle1, angle2):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('$\\theta_0$')
        ax.set_ylabel('$\\theta_1$')
        ax.set_zlabel('$J(\\theta)$')
        surf = ax.plot_surface(th0, th1, Js, cmap=cm.coolwarm)

        ax.view_init(angle1, angle2)
        plt.show()
        
def plot_simg_error():
    def cost_1(z):
        return - np.log(sigm(z))

    def cost_0(z):
        return - np.log(1 - sigm(z))

    z = np.arange(-10, 10, 0.1)
    phi_z = sigm(z)

    c1 = [cost_1(x) for x in z]
    plt.plot(phi_z, c1, label='$-y \log(\phi(z_i))$ if y=1')

    c0 = [cost_0(x) for x in z]
    plt.plot(phi_z, c0, linestyle='--', label='$-(1-y) \log(1-\phi(z_i))$ if y=0')

    plt.ylim(0.0, 5.1)
    plt.xlim([0, 1])
    plt.xlabel('$\phi$(z)')
    plt.ylabel('J($\\theta$)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
 

def multy_log(X, y):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, y)
    plot_iris(X, y, X_unseen=None, classifier=logreg, same_range=False)

# ********************************************************************    
    
def get_data_for_task():
    ind = [0, 4]
    data = load_breast_cancer()
    X = data.data[:100:2][:, ind]
    S = StandardScaler().fit(X)
    X = S.transform(X) / 3
    y = data.target[:100:2]
    
    X[y==0, 0] #+= 0.4
    X[y==0, 1] #-= 0.2 

    X = np.hstack( (np.ones( (X.shape[0], 1)), X))
  
    return X, y

def simple_plot(X, y):
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle("Границы классов", fontsize=16)
    ax.set_xlabel("Нормализованный средний радиус", fontsize=14)
    ax.set_ylabel("Нормализованная средняя плавность", fontsize=14)
    ax.scatter(X[y==0, 1], X[y==0, 2], color='red')
    ax.scatter(X[y==1, 1], X[y==1, 2], color='blue')
    plt.show() 

def check_task(y, y_pred, stuff, y_tp=None, y_pred_tp=None):
    if np.allclose(y, y_pred):
        print("Все шикарно, молодец!")
    else:
        print("Что-то не так :(")
    if y_tp is None:
        print("Реальное значение " + stuff + ": {0}".format(np.round(y, 4)))
        print("Твое значение " + stuff + ": {0}".format(np.round(y_pred, 4)))
    else:
        print("Реальное значение " + stuff + ": {0}".format(np.round(y_tp, 4)))
        print("Твое значение " + stuff + ": {0}".format(np.round(y_pred_tp, 4)))
        
        
class Lg():
    def __init__(self):
        self.theta = None
        
    def sigmoid(self, theta, X):
        return 1.0 / (1.0 + np.exp(-X.dot(theta)))
    
    def error(self, theta, X, y):
        pred = self.sigmoid(theta, X)
        return np.mean(-y*np.log(pred) - (1 - y)*np.log(1 - pred))/2
    
    def gradient(self, theta, X, y):
        pred = self.sigmoid(theta, X)
        return np.dot(X.T, (pred - y.reshape(-1, 1))) / X.shape[0]
    
    def gradient_descent(self, init_theta, X, y, alpha, iters):
        theta = init_theta
        self.errors = [self.error(theta, X, y)]
        for i in range(iters):
            theta = theta - alpha*self.gradient(theta, X, y)
            self.errors.append(self.error(theta, X, y))
            
        return theta

    def fit(self, X, y, iters=10, alpha=0.01):
        init_theta = np.random.rand(X.shape[1], 1)
        self.theta = self.gradient_descent(init_theta, X, y, alpha, iters)
    
    def predict(self, X):
        return np.around(self.sigmoid(self.theta, X)) 
      

def check_sigmoid(sigmoid, X):
    lg = Lg()
    theta = np.array([[1],[1],[1]])
    
    y = lg.sigmoid(theta=theta, X=X[:5])
    y_pred = sigmoid(theta=theta, X=X[:5])
    check_task(y.ravel(), y_pred.ravel(), stuff="сигмоиды", y_tp=y.ravel(), y_pred_tp=y_pred.ravel())    
    
def check_loss_func(loss_func, X, y):
    lg = Lg()
    theta = np.array([[1],[1],[1]])
    
    J = lg.error(theta=theta, X=X[:5], y=y[:5])
    J_pred = loss_func(theta=theta, X=X[:5], y=y[:5])
    check_task(J, J_pred, stuff="функции потерь", y_tp=None, y_pred_tp=None) 
    
def check_gradient_function(gradient_function, X, y):
    lg = Lg()
    theta = np.array([[1],[1],[1]])
    grad = lg.gradient(theta=theta, X=X, y=y)
    grad_pred = gradient_function(theta=theta, X=X, y=y)
    check_task(grad, grad_pred, stuff="градиента", y_tp=grad.ravel(), y_pred_tp=grad_pred.ravel()) 
    
    
def check_gradient_descent(gradient_descent, X, y):
    lg = Lg()
    theta = np.array([[1.0],[1.0],[1.0]])
    
    alpha = 5.0
    iters = 50
    theta_real = lg.gradient_descent(init_theta=theta, X=X, y=y, alpha=alpha, iters=iters)
    theta_pred, errors = gradient_descent(init_theta=theta, X=X, y=y, alpha=alpha, iters=iters)
    check_task(theta_real, theta_pred, stuff="значений тета", y_tp=theta_real.ravel(), y_pred_tp=theta_pred.ravel()) 
    
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle("Значения функции потерь", fontsize=16)
    ax.set_xlabel("Номер итерации", fontsize=14)
    ax.set_ylabel("Функция потерь", fontsize=14)
    ax.plot(range(1, len(errors)+1), errors)
    plt.show()
    
    lg.theta = theta_pred
    cmap = ListedColormap(['red', 'blue'])
    x1_min, x1_max = -1, 2
    x2_min, x2_max = -1, 2
    resolution = 0.01
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    a = np.array([np.ones_like(xx2.ravel()), xx1.ravel(), xx2.ravel()])
    Z = lg.predict(a.T)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle("Границы классов", fontsize=16)
    ax.set_xlabel("Нормализованный средний радиус", fontsize=14)
    ax.set_ylabel("Нормализованная средняя гладкость", fontsize=14)
    ax.contourf(xx1, xx2, Z.reshape(xx1.shape), alpha=0.4, cmap=cmap)
    ax.scatter(X[y==0, 1], X[y==0, 2], color='red')
    ax.scatter(X[y==1, 1], X[y==1, 2], color='blue')
    
    plt.show()    