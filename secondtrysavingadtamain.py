#%%
import numpy as np
import matplotlib.pyplot as plt
import random as random
import time

#%%
#defining the function: 
start_time = time.time()

def Wright_Fisher_model(N, p0, generations, mu, v):
    
    p = np.full(a , p0)
    
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
v_values = [1e-2]  

#%%
#saving proccess:
a = 1000000
batch_size = 100000
num_batches = a // batch_size
 
output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

for batch in range(num_batches):
    
    batch_a = [Wright_Fisher_model(batch_size, p0, generations, mu, v) for v in v_values]
    
    output_filename = f"{output_directory}\\final_frequencies_batch{batch}.txt"

    np.savetxt(output_filename, batch_a , delimiter=',', fmt='%f')


#%%
#plotting process:
    
plt.figure()
# Loop through each batch
for batch in range(num_batches):
    
    # Load data from the batch file
    loaded_data = np.loadtxt(f'{output_directory}\\final_frequencies_batch{batch}.txt', delimiter=',')
    
    # Define bin edges and compute the histogram
    bin_width = np.linspace(((1 / N) + (1 /(2 * N))), 1, 100)
    
    counts, bins = np.histogram(loaded_data, bins=bin_width)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))
    
    normalized_counts = counts / riemann_sum
    
    plt.plot(bin_centers, normalized_counts , label=f'Batch { batch + 1}')


# Initialize an empty array to collect allele frequency data from all batches
all_data = []

# Loop through each batch
for batch in range(num_batches):
    
    # Load data from the batch file 
    loaded_data = np.loadtxt(f'{output_directory}\\final_frequencies_batch{batch}.txt', delimiter=',')
    
    # Append the data to the all_data array
    all_data.append(loaded_data)

# Concatenate all data from different batches into a single array
all_data = np.concatenate(all_data)

# Create a histogram of the combined data

bin_width = np.linspace(((1 / N) + (1 /(2 * N))), 1, 100)

counts, bins = np.histogram(all_data, bins=bin_width)

bin_centers = (bins[:-1] + bins[1:]) / 2

riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))

normalized_counts = counts / riemann_sum

# Plot the combined histogram as a single curve 
plt.plot(bin_centers, normalized_counts, marker='o',  color='green' , label='all Data')

# Show the plot with both curves
plt.legend()
plt.xlabel("Frequency")
plt.ylabel("Normalized Counts")
plt.title("Normalized Frequency Distribution")
plt.show()

