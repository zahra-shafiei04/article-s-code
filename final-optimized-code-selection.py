#%%
import numpy as np
import matplotlib.pyplot as plt
import random as random
import time
import scipy

#%%
#defining the function: 
start_time = time.time()

def Wright_Fisher_model(N, p0, generations, mu, v, a):
    
    p = np.full(a , p0)
    
    for jj in range(generations):
        
        # Describing mutations all over the simulation and fluctuating selection:
        s = np.random.normal(ms , np.sqrt(v), a) #s = sigma
        t = np.random.normal(mt , np.sqrt(v), a) #t = tau

        # Update 'p' using the given equations, based on the previous value of 'p':
        p = p + mu * (1 - p) +  (p * (1 - p) * (s - t)) / (1 + (p * s) + (t * (1 - p)))
        
        # Describing drift:
        allele_counts = np.random.binomial(2 * N, p)
        p = allele_counts / (2. * N)
        
        #checking if frequency hits the boundry (0) and if a mutation is happening with rate mu:
        # if (p == 0) and (np.random.rand() <= mu * 2 * N):
        #     p = 1 / N
        # elif    
        p[(p==1)]=0
        
    return p 


N = 1000
p0 = 0.01
generations = 10 * N
mu = 1 / (10 * N)
v_values = [1e-5, 1e-2]  
ms = -0.15
mt = -0.15

#%%
#saving proccess:
a = 10**5
batch_size = 10**4
num_batches = a // batch_size
 
output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

for batch in range(num_batches):
    
    batch_a = [Wright_Fisher_model(N, p0, generations, mu, v, a) for v in v_values]

    for i, v in enumerate(v_values):
        
        output_filename = f"{output_directory}\\final_frequencies_batch{batch}_v{v}.txt"
        
        np.savetxt(output_filename, batch_a[i], delimiter=',', fmt='%f')

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

plt.figure()

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, v in enumerate(v_values):
    
    color = color_cycle[i % len(color_cycle)]
    
# Loop through each batch
    for batch in range(num_batches):
        
        # Load data from the batch file
        loaded_data = np.loadtxt(f"{output_directory}\\final_frequencies_batch{batch}_v{v}.txt", delimiter=',')

        # Define bin edges and compute the histogram
        bin_width = np.linspace(((1 / N) + (1 /(2 * N))), 1, 101)
    
        counts, bins = np.histogram(loaded_data, bins=bin_width)
    
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
        riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))
    
        normalized_counts = counts / riemann_sum
    
        plt.plot(bin_centers, normalized_counts , label=f'Batch { batch + 1}', color=color)


    # Initialize an empty array to collect allele frequency data from all batches
    all_data = []

    # Loop through each batch
    for batch in range(num_batches):
        
        # Load data from the batch file 
        loaded_data = np.loadtxt(f"{output_directory}\\final_frequencies_batch{batch}_v{v}.txt", delimiter=',')
    
        # Append the data to the all_data array
        all_data.append(loaded_data)

    # Concatenate all data from different batches into a single array
    all_data = np.concatenate(all_data)

    # Create a histogram of the combined data

    bin_width = np.linspace(((1 / N) + (1 /(2 * N))), 1, 101)

    counts, bins = np.histogram(all_data, bins=bin_width)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))

    all_normalized_counts = counts / riemann_sum

    print(f"Area under simulation curve {np.sum( all_normalized_counts * (bin_centers[1] - bin_centers[0]))}")

    # Plot the combined histogram as a single curve 
    plt.plot(bin_centers, all_normalized_counts, marker='o' , label='all Data', color=color)

# Show the plot with both curves
plt.legend()
plt.xlabel("Frequency")
plt.ylabel("Normalized Counts")
plt.title("Normalized Frequency Distribution")
plt.show()


#plotting analytical answer:
for i, v in enumerate(v_values):
    
    color = color_cycle[i % len(color_cycle)]
    
    B = 2 * N * 2 * v
    
    f1_values = f1(bin_centers, B)  
    
    riemann_sum_analytical = np.sum(f1_values * (bin_centers[1] - bin_centers[0]))
    
    normalized_curve = f1_values / riemann_sum_analytical
    
    print(f"Area under analytical solution curve for v={v}:{np.sum(normalized_curve) * (bin_centers[1] - bin_centers[0])}")
    
    plt.plot(bin_centers, normalized_curve, linestyle='--', label=f'Analytical v={v}',color=color)
    
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
        V = loaded_data = np.loadtxt(f"{output_directory}\\final_frequencies_batch{batch}_v{v}.txt", delimiter=',')

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

