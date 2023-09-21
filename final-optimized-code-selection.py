#%%
import numpy as np
import matplotlib.pyplot as plt
import random as random
import time

#%%
start_time = time.time()

def Wright_Fisher_model(N, p0, generations, mu, v):
    
    # Initialize 'p' as an array with a desired length and set all elements to 'p0':
    p = np.full(100 * N, p0)
    
    for jj in range(generations):
        
        # Describing mutations all over the simulation and fluctuating selection:
        s = np.random.normal(0, v) #s = sigma
        t = np.random.normal(0, v) #t = tau

        # Update 'p' using the given equations, based on the previous value of 'p':
        p = p + mu * (1 - p) +  (p * (1 - p) * (s - t)) / (1 + (p * s) + (t * (1 - p)))
        
        # Describing drift:
        allele_counts = np.random.binomial(2 * N, p)
        p = allele_counts / (2. * N)
        
        p[(p==1)]=0
        
    return p 

N = 1000
p0 = 0.01
generations = 10 * N
mu = 1 / (10 * N)
v_values = [1e-2, 5e-2, 2e-2, 1e-1]  


#%%
#saving data:
# Collect the final allele frequencies for different 'v' values
all_final_frequencies = [Wright_Fisher_model(N, p0, generations, mu, v) for v in v_values]

#%%
#defining the analytical solution function:
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

#%%
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plotting the SFS for each 'v' value in one figure
for i, v in enumerate(v_values):

    # Create a histogram with the final frequency data and specified bin width
    bin_width = np.linspace(((1 / N) + (1 /(2 * N))), 1, 100)
    
    counts, bins = np.histogram(all_final_frequencies[i], bins=bin_width)

    # Calculate the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate the Riemann sum to estimate the area under the curve
    riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))
    
    # Normalizing the curve
    normalized_counts = counts / riemann_sum
    
    color = color_cycle[i % len(color_cycle)]
    
    print(f"Area under simulation curve for v={v}: {np.sum(normalized_counts) * (bin_centers[1] - bin_centers[0])}")
    
    plt.plot(bin_centers, normalized_counts, label=f'Simulated v={v}')
       
    #same process for the second function:
    B = 2 * N * 2 * v
    
    f1_values = f1(bin_centers, B)  
    
    riemann_sum_analytical = np.sum(f1_values * (bin_centers[1] - bin_centers[0]))
    
    normalized_curve = f1_values / riemann_sum_analytical
    
    print(f"Area under analytical solution curve for v={v}:{np.sum(normalized_curve) * (bin_centers[1] - bin_centers[0])}")
    
    plt.plot(bin_centers, normalized_curve, linestyle='--', label=f'Analytical v={v}', color=color)
    
plt.xlabel('Frequency')
plt.ylabel('Normalized Counts / Normalized Analytical Values')
plt.title('SFS Fluctuating Selection & Normalized Analytical Solution')
plt.legend()
plt.show()

#%%
# Record the end time
end_time = time.time()

# Calculate the total running time
running_time = end_time - start_time
print("Total running time:", running_time, "seconds")










