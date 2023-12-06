#%%
import numpy as np
import matplotlib.pyplot as plt
import random as random
import time
import scipy

#%%
#defining the function: 
start_time = time.time()

def Wright_Fisher_model(N, p0, generations, mu, v, a, ms, mt, x):
   
    p = np.full(a , p0)
    
    for jj in range(generations):
        
        # describing fluctuating selection:
        s = np.random.normal(ms , np.sqrt(v), a) #s = sigma
        t = np.random.normal(mt , np.sqrt(v), a) #t = tau
        
        # store the value of selection coefficient:
        st = (s - t) / (1 + (p * s) + (t * (1 - p)))
        
        # main equation describing mutation and fluctuating selection:
        p = p + mu * (1 - p) +  (p * (1 - p) * (s - t)) / (1 + (p * s) + (t * (1 - p)))
        
        # check if selection coefficient is pushing too much, adjust p:    
        p[(p < 0)] = 0
        p[(p > 1)] = 1
        
        # describing drift:
        allele_counts = np.random.binomial(2 * N, p)
        p = allele_counts / (2. * N)
        
        p[(p == 1)] = 0

    return st 

# initial value for describing phenomenons:
N = 1000
p0 = 0.01
generations = 10 * N
mu = 1 / (10 * N)

# initial value to decribe flactuating selection:
v_values = [0 , 1e-5]  
x = 0.01
ms_values = [-x/2]
mt_values = [ x/2]

#%%
#saving proccess:
a = 10**5
batch_size = 10**4
num_batches = a // batch_size

output_directory = r"C:\Users\Zahra\research codes-st value"

for i, v in enumerate(v_values):
    
    for j in range(len(ms_values)):
       
        for batch in range(num_batches):
             
            ms_val = ms_values[j]
            mt_val = mt_values[j]
                        
            s = np.random.normal(ms_val, np.sqrt(v), a)  # s = sigma
            t = np.random.normal(mt_val, np.sqrt(v), a)  # t = tau
            
            st_values = Wright_Fisher_model(N, p0, generations, mu, v, a, s, t, x)
            
            output_filename = f"{output_directory}\\st_b{batch}_v={v}_ms={ms_val}_mt={mt_val}.txt"
        
            np.savetxt(output_filename, st_values , delimiter=',', fmt='%f')


#%%
# Plot histograms for each value of v and corresponding (ms, mt):
output_directory = r"C:\Users\Zahra\research codes-st value"

a = 10**5
batch_size = 10**4
num_batches = a // batch_size

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, v in enumerate(v_values):
    
    color = color_cycle[i % len(color_cycle)]
    
    for j in range(len(ms_values)):
        
        ms_val = ms_values[j]
        mt_val = mt_values[j]
        
        all_st_values = []
    
        for batch in range(num_batches):
            
            # Load data from the batch file
            loaded_data = np.loadtxt(f"{output_directory}\\st_b{batch}_v={v}_ms={ms_val}_mt={mt_val}.txt", delimiter=',')
        
            all_st_values.append(loaded_data)
        
        all_st_values = np.concatenate(all_st_values)
    
        plt.figure(figsize=(8, 6))
        plt.hist(all_st_values, bins=100)
        plt.xlabel('Selection Coefficient')
        plt.ylabel('Counts')
        plt.title(f'Histogram of Selection Coefficient (v = {v}_ms={ms_val}_mt={mt_val})')
    
        # Calculate S.D. for each selection coefficient
        std_deviation = np.std(all_st_values)
        mean = np.mean(all_st_values)
    
        plt.axvline(mean, color='red', linestyle='--', label='Mean')
    
        # Add a text label for the mean value
        plt.text(mean, plt.ylim()[1]*0.9, f'Mean = {mean:.4f}', color='red', ha='center')
        
        # Add a text label for the standard deviation value
        plt.text(mean, plt.ylim()[1]*0.85, f'Standard Deviation = {std_deviation:.4f}', color='green', ha='center')

        print(f"Standard Deviation of Selection Coefficients for v={v}_ms={ms_val}_mt={mt_val}: {std_deviation}")
        print(f"Mean of Selection Coefficients for v={v}_ms={ms_val}_mt={mt_val}: {mean}")
        
        plt.show()  
        

#%%
# Record the end time
end_time = time.time()

# Calculate the total running time
running_time = end_time - start_time
print("Total running time:", running_time, "seconds")


