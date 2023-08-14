#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def r1(B):
    return (1 - (np.sqrt(1 + (4 / B)))) / 2

def r2(B):
    return (1 + (np.sqrt(1 + (4 / B)))) / 2

def k(B):
    return np.log(((1 - r1(B)) / (-r1(B))) * (r2(B) / (r2(B) - 1)))

def g(y, B):
    return np.log(((1 - r1(B)) / (y - r1(B))) * ((r2(B) - y) / (r2(B) - 1)))

def f1(y, B):
    return (2 / (k(B) * y * (1 - y))) * g(y, B)


# if we increase the variance we will cover both boundries
v = 1000
s = np.random.normal(0, v)
t = np.random.normal(0, v ) 

B = 2 * v
y_values = np.linspace( 0 , 1 , 100)
f1_values = f1(y_values, B)

plt.plot(y_values, f1_values, label = "v")
plt.xlabel('y')
plt.ylabel('f1(y)')
plt.title('Analytical Solution')
plt.legend()
plt.show()
