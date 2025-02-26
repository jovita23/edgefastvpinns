# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)

eps = 0.5

Z = np.tanh(15 * X * Y) * np.cos(2*np.pi*Y*X) 

plt.contourf(X, Y, Z, cmap='jet', levels=100)
plt.colorbar()
plt.show()

# plt.contourf(X, Y, eps, cmap='jet', levels=100)
# plt.colorbar()
# plt.show()
# %%
from sympy import symbols, sin, cos, exp, tanh, diff, pretty
import sympy as sp
#import lambdify to convert symbolic expression to a function
from sympy.utilities.lambdify import lambdify

# Define the symbols
X, Y, eps, b1, b2 = symbols('X Y eps b1 b2')

# Original expression
expr = tanh(15 * X * Y) * cos(2*np.pi*Y*X) 

eps = 0.5
b1 = 1
b2 = 0
# Compute the negative Laplacian of the expression
# Laplacian in 2D: d^2/dx^2 + d^2/dy^2
laplacian_expr = -eps * (diff(  expr, X, X) + diff( expr, Y, Y)) + b1 * diff(expr, X) + b2 * diff(expr, Y)
# simplified_laplacian_expr = laplacian_expr.simplify()
import inspect

laplacian_expr = laplacian_expr.simplify()

fn = lambdify((X, Y), laplacian_expr, "numpy")

fn_text = inspect.getsource(fn)

# in the function_text, replace the following items with following values
changes = {"X": "x", "Y":"y", "sin": "np.sin", "cos": "np.cos", "tanh": "np.tanh", "exp": "np.exp", "log": "np.log", "sqrt": "np.sqrt", "pi": "np.pi"}

#replace the _lambdifygenerated with f
fn_text = fn_text.replace("_lambdifygenerated", "f")

#replace the items in the changes dictionary
for key, value in changes.items():
    fn_text = fn_text.replace(key, value)

print(fn_text)

# %%

# generated by above code

def f(x, y):
    return x**2*(188.495559215388*np.sin(6.28318530717959*x*y) + 19.7392088021787*np.cos(6.28318530717959*x*y)*np.sinh(30*x*y) + 450.0*np.cos(6.28318530717959*x*y)*np.tanh(15*x*y))/(np.cosh(30*x*y) + 1) + y**2*(188.495559215388*np.sin(6.28318530717959*x*y) + 19.7392088021787*np.cos(6.28318530717959*x*y)*np.sinh(30*x*y) + 450.0*np.cos(6.28318530717959*x*y)*np.tanh(15*x*y))/(np.cosh(30*x*y) + 1) - 6.28318530717959*y*np.sin(6.28318530717959*x*y)*np.tanh(15*x*y) + 15*y*np.cos(6.28318530717959*x*y)/np.cosh(15*x*y)**2


# Check if the f(x,y) and the laplacian_expr are the same

#lamdaify the laplacian expression
f_sym = lambdify((X, Y), laplacian_expr,"numpy")


# get 100 random points
x = np.random.uniform(-1, 1, 1000)
y = np.random.uniform(-1, 1, 1000)

# evaluate the expression at these points
f_eval_sp = f_sym(x, y)
f_eval_np = f(x, y)

# check if the values are the same
print(np.allclose(f_eval_sp, f_eval_np))


# %% 
# %%
