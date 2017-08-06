import numpy as np

def check(name, glob):
    if name not in glob:
        print(name + " несозданно.")
        return False
    return True


def check_var(glob):
    if val_check("val1", 1, glob) and val_check("val2", 20, glob):
        print("Молодец!")
        
        
def val_check(name, real, glob):
    if check(name, glob):
        v = glob[name]
        if v != real:
            print(name + " не равно " + str(real))
            return False
        return True
    else:
        return False
    
def check_formula(glob):
    if val_check('c', (10*10-3*3) / 2, glob):
        print("Молодец!")
        
def check_array(X, res, myres=None):
    if myres is None:
        myres = 1
        for i in X:
            myres = myres*i
    if np.allclose(res, myres):
        print("Молодец!")
    else:
        print("Что не так :(")
        print("Твой результат {0}, а должно быть {1}.".format(res, myres))   

def task_1(X, a, res):
    check_array(X, res, myres=a*X)
    
def task_2(A, B, a, res):
    check_array(A, res, myres=A - a*B)
    
def task_3(A, B, a, res):
    check_array(A, res, myres=(A - a*B)**2)
    
def task_4(A, B, a, res):
    check_array(A, res, myres=sum((A - a*B)**2))     