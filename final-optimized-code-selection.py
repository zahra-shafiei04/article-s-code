#%%
import numpy as np
import matplotlib.pyplot as plt
import random as random
import time

#%%
start_time = time.time()

def Wright_Fisher_model(N, p0, generations, mu, v):
    
    allele_frequencies = []
    f = []
    
    # Initialize 'p' as an array with any desired length and set all elements to 'p0':
        
    p = np.full(100000, p0)  

    for jj in range(generations):
        
        # Describing mutations all over the simulation and fluctuating selection:
                
        s = np.random.normal(0, v)
        t = np.random.normal(0, v)
        
        # Update 'p' using the given equations, based on the previous value of 'p':
        p = p + mu * (1 - (2 * p)) + (p * (1 - p) * (s - t)) / (1 + (p * s) + (t * (1 - p)))
        
        #I have to clip the values becuase I'm changing variance insanely(otherwise it can be commented):
        p = np.clip(p, 0, 1)
        
        # Describing drift:
        allele_counts = np.random.binomial(2 * N, p)
        p = allele_counts / (2 * N)
        
        # Appending the final frequency after these two steps:
        allele_frequencies.append(p)
        f1 = allele_frequencies[-1]
        
        allele_frequencies.append(allele_frequencies)
        f.append(f1)
        
    return allele_frequencies, f

N = 100
p0 = 0.01
generations = 1000
mu = 1e-3

v_values = [1e-10, 1e-5, 1, 10]  # List of 'v' values to iterate over

# Collect the results for different 'v' values
all_results = []
for v in v_values:
    allele_frequencies, f = Wright_Fisher_model(N, p0, generations, mu, v)
    all_results.append(f)


# Plotting the SFS for each 'v' value in one figure
for i, v in enumerate(v_values):
    
    #plt.hist(all_results[i], bins=100, label=f'v={v}')
    
    # Calculate the histogram
    counts, bins = np.histogram(all_results[i], bins=100)

    # Get the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Create the line plot
    plt.plot(bin_centers, counts, label=f'v={v}')
    
plt.xlabel('Frequency')
plt.ylabel('Counts')
plt.title('SFS Fluctuating Selection')
plt.legend()
plt.show()

# Record the end time
end_time = time.time()

# Calculate the total running time
running_time = end_time - start_time
print("Total running time:", running_time, "seconds")

#%%
#This is the best version of my code up to now which is working 
#I deleted for loop for simulation and I'm keeping only final value to plot SFS

start_time = time.time()

def Wright_Fisher_model(N, p0, generations, mu, v):
    
    f_final = None  # Initialize variable to store the final allele frequency

    # Initialize 'p' as an array with a desired length and set all elements to 'p0':
    p = np.full(100000, p0)

    for jj in range(generations):
        # Describing mutations all over the simulation and fluctuating selection:
        s = np.random.normal(0, v)
        t = np.random.normal(0, v)
        
        # Update 'p' using the given equations, based on the previous value of 'p':
        p = p + mu * (1 - (2 * p)) + (p * (1 - p) * (s - t)) / (1 + (p * s) + (t * (1 - p)))
        
        # Clip the values to ensure they stay within [0, 1]:
        p = np.clip(p, 0, 1)
        
        # Describing drift:
        allele_counts = np.random.binomial(2 * N, p)
        p = allele_counts / (2 * N)
        
        # Store the final frequency after these two steps:
        f_final = p
        
    return f_final

N = 100
p0 = 0.01
generations = 1000
mu = 1e-3
v_values = [1e-10,1e-5, 1, 10]  


# Collect the final allele frequencies for different 'v' values
all_final_frequencies = []
for v in v_values:
    final_frequency = Wright_Fisher_model(N, p0, generations, mu, v)
    all_final_frequencies.append(final_frequency)


# Plotting the SFS for each 'v' value in one figure
for i, v in enumerate(v_values):
    # Create a histogram with only the final frequency for each 'v' value
    counts, bins = np.histogram(all_final_frequencies[i], bins=100)

    # Get the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate the Riemann sum to estimate the area under the curve
    riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))

    print(f"Estimated area under curve for v={v}: {riemann_sum}")
    
    # Create the line plot
    plt.plot(bin_centers, counts, label=f'v={v}')
    
plt.xlabel('Frequency')
plt.ylabel('Counts')
plt.title('SFS Fluctuating Selection')
plt.legend()
plt.show()

# Record the end time
end_time = time.time()

# Calculate the total running time
running_time = end_time - start_time
print("Total running time:", running_time, "seconds")








