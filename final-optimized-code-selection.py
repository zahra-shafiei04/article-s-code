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

        # main equation describing mutation and fluctuating selection:
        p = p + mu * (1 - p) +  (p * (1 - p) * (s - t)) / (1 + (p * s) + (t * (1 - p)))
        
        #check if selection coefficient is pushing too much, adjust p:    
        p[(p < 0)] = 0
        p[(p > 1)] = 1
        
        # Describing drift:
        allele_counts = np.random.binomial(2 * N, p)
        p = allele_counts / (2. * N)
        
        p[(p == 1)] = 0

    return p 

#initial value for describing phenomenon:
N = 1000
p0 = 0.01
generations = 10 * N
mu = 1 / (10 * N)

#initial value to decribe flactuating selection:
v_values = [0, 1e-5]  
x = 0.01
ms_values = [-x/2]
mt_values = [x/2]


#%%
#saving proccess:
a = 10**5
batch_size = 10**4
num_batches = a // batch_size
 
output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

for i, v in enumerate(v_values):
    
    for j in range(len(ms_values)):
       
        for batch in range(num_batches):
             
            ms_val = ms_values[j]
            mt_val = mt_values[j]
                        
            s = np.random.normal(ms_val, np.sqrt(v), a)  # s = sigma
            t = np.random.normal(mt_val, np.sqrt(v), a)  # t = tau
    
            batch_a = Wright_Fisher_model(N, p0, generations, mu, v, a, s, t, x)
        
            output_filename = f"{output_directory}\\p_b{batch}_v={v}_ms={ms_val}_mt={mt_val}.txt"
        
            np.savetxt(output_filename, batch_a, delimiter=',', fmt='%f')
     
#%%
#defining the analytical solution function:
    
def r1(B):
    return (1 - (np.sqrt(1 + (4 / B))))/ 2

def r2(B):
    return (1 + (np.sqrt(1 + (4 / B)))) / 2

def k(B):
    return np.log(((1 - r1(B)) / (-r1(B))) * (r2(B) / (r2(B) - 1)))

def g(y, B):
    return np.log(((1 - r1(B)) / (y - r1(B))) * ((r2(B) - y) / (r2(B) - 1)))

def f1(y, B):
    return (2 / (k(B) * y * (1 - y))) * g(y, B)

#%%
#plotting process:
#back up directory for huge data set, number of trajectories) 1e6 : 
# output_directory = r"C:\Users\Zahra\research codes_max trajectories"

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

a = 10**5
batch_size = 10**4
num_batches = a // batch_size

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, v in enumerate(v_values):
    
    color = color_cycle[i % len(color_cycle)]
        
    for j in range(len(ms_values)):
               
        ms_val = ms_values[j]
        mt_val = mt_values[j]
        
        for batch in range(num_batches):
        
            loaded_data = np.loadtxt(f"{output_directory}\\p_b{batch}_v={v}_ms={ms_val}_mt={mt_val}.txt", delimiter=',')

            # Define bin edges and compute the histogram:
                
            bin_width = np.linspace(((1 / N) + (1 /(2 * N))), 1, 101)
    
            counts, bins = np.histogram(loaded_data, bins=bin_width)
    
            bin_centers = (bins[:-1] + bins[1:]) / 2
    
            riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))
    
            normalized_counts = counts / riemann_sum
    
            plt.plot(bin_centers, normalized_counts, color=color)


        # Initialize an empty array to collect allele frequency data from all batches
        all_data = []

        for batch in range(num_batches):
        
            loaded_data = np.loadtxt(f"{output_directory}\\p_b{batch}_v={v}_ms={ms_val}_mt={mt_val}.txt", delimiter=',')
    
            all_data.append(loaded_data)

        all_data = np.concatenate(all_data)

        # Create a histogram of the combined data:
            
        bin_width = np.linspace(((1 / N) + (1 /(2 * N))), 1, 101)

        counts, bins = np.histogram(all_data, bins=bin_width)

        bin_centers = (bins[:-1] + bins[1:]) / 2

        riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))

        all_normalized_counts = counts / riemann_sum

        print(f"Area under simulation curve {np.sum( all_normalized_counts * (bin_centers[1] - bin_centers[0]))}")

        plt.plot(bin_centers, all_normalized_counts, marker='o' , label=f'all Data_v={v}_ms={ms_val}_mt={mt_val}', color=color)

plt.legend()
plt.xlabel("Frequency")
plt.ylabel("Normalized Counts")
plt.title("Normalized Frequency Distribution")
plt.show()

#%%
#plotting analytical answer:
for i, v in enumerate(v_values):

    for j in range(len(ms_values)):
        
        color = color_cycle[j % len(color_cycle)]
        
        ms_val = ms_values[j]
        mt_val = mt_values[j]         
      
        B = 2 * N * 2 * v
    
        f1_values = f1(bin_centers, B)  
    
        riemann_sum_analytical = np.sum(f1_values * (bin_centers[1] - bin_centers[0]))
    
        normalized_curve = f1_values / riemann_sum_analytical
    
        print(f"Area under analytical solution curve for v={v}_ms={ms_val}_mt={mt_val}:{np.sum(normalized_curve) * (bin_centers[1] - bin_centers[0])}")
    
        plt.plot(bin_centers, normalized_curve, linestyle='--', label=f'Analytical v={v}_ms={ms_val}_mt={mt_val}',color=color)

plt.xlabel('Frequency')
plt.ylabel('Normalized Counts / Normalized Analytical Values')
plt.title('SFS Fluctuating Selection & Normalized Analytical Solution')
plt.legend()
plt.show()

#%%
#evaluating genetic variation:
a = 10**5
batch_size = 10**4
num_batches = a // batch_size
 
output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

GV_values = []

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, v in enumerate(v_values):
    
    color = color_cycle[i % len(color_cycle)]

    # Loop through each batch
    for batch in range(num_batches):
        # Load data from the batch file
        V = loaded_data = np.loadtxt(f"{output_directory}\\p_b{batch}_v={v}_ms={ms_val}_mt={mt_val}.txt", delimiter=',')

    GV = (1 / len(V)) * 2 * np.sum(V * (1 - V))
    
    print(f"GV for v = {v}: {GV}")
    
    GV_values.append(GV)
    
plt.plot(v_values, GV_values,marker ='o')
plt.xlabel('v')
plt.ylabel('GV')
plt.title('Genetic Variation (GV) vs. Fluctuating Selection (v)')
plt.grid(True)
plt.show()

#%%
# Record the end time
end_time = time.time()

# Calculate the total running time
running_time = end_time - start_time
print("Total running time:", running_time, "seconds")

