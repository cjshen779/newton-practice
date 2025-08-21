import math 
import numpy as np 

## maybe need set maximum iterations

epsilon = 0.001 
delta = 0.0001 

def first_deriv(f, x, delta): 
    result = (f(x+delta) - f(x))/delta  
    return result 
    
def second_deriv(f, x, delta): 
    result = (first_deriv(f, x + delta, delta) - first_deriv(f, x, delta))/delta 
    return result 

def optimize(x_0, f): 
    # initial vals 
    x_vals = [x_0] 
    x_t = x_0  
    
    dist = math.inf   
    
    while dist > epsilon: 
        x_t = x_t - (first_deriv(f, x_t, delta)/second_deriv(f,x_t, delta)) 
        dist = abs(x_t - x_vals[-1]) 
        x_vals.append(x_t)

    return f"(x, f(x)) = {(x_t, f(x_t))}" 
