

import random

def func(x):
    return (3*x) + 2

def loss_func(x, y, w, b):
    return ( y - ( (w*x) + b) )**2


def gradient_w_func(x, y, w, b):
    return 2*( y - ( (w*x) + b) )*-x

def gradient_b_func(x, y, w, b):
    return 2*( y - ( (w*x) + b) )*-1



def opt_func_non_batch(Xs, Ys, learning_rate, steps=100):

    w = 0.0
    b = 0.0
    for s in range(steps):
        for i in range (0, len(Xs)):
            grad_w = gradient_w_func(Xs[i], Ys[i], w, b)
            grad_b = gradient_b_func(Xs[i], Ys[i], w, b)
            w = w - (learning_rate*grad_w) 
            b = b - (learning_rate*grad_b)
        
        print(f"loss {s}: {loss_func(Xs[i], Ys[i], w, b)}")
    return w, b

def opt_func_batch(Xs, Ys, learning_rate, batch_size = 5, steps=100):

    w = 0.0
    b = 0.0
    for s in range(steps):
        for b in range(1, int(len(Xs)/batch_size)+1):
            w_acc = 0
            b_acc = 0
            batch_begin = (b-1)*batch_size
            batch_end = b*batch_size
            for i in range (batch_begin, batch_end):
                grad_w = gradient_w_func(Xs[i], Ys[i], w, b)
                grad_b = gradient_b_func(Xs[i], Ys[i], w, b)
                new_w = w - (learning_rate*grad_w) 
                new_b = b - (learning_rate*grad_b)
                w_acc += new_w
                b_acc += new_b
            w = w_acc / batch_size
            b = b_acc/ batch_size
        print(f"loss {s}: {loss_func(Xs[i], Ys[i], w, b)}")
    return w, b




x = [random.random()*10 for _ in range (0, 10)]
y = [func(xi) for xi in x]
print(x)
print(y)
print("............ non batch.............")
res = opt_func_non_batch(x, y, 0.01)
print(f"w: {res[0]}, b: {res[1]}")
print("............. batch ............")
res = opt_func_batch(x, y, 0.01)
print(f"w: {res[0]}, b: {res[1]}")