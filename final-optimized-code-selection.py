#%%
import numpy as np
import matplotlib.pyplot as plt
import random as random
import time

#%%
#defining the function: 
start_time = time.time()

def Wright_Fisher_model(N, x0, generations, mu, fluctuation, length_x , mean_σ, mean_τ, bias):
    
    x = np.full(length_x , x0)
    
    for i in range(generations):
        
        # describing parameters for fluctuating selection:
        σ = np.random.normal(mean_σ , np.sqrt(fluctuation), length_x) 
        τ = np.random.normal(mean_τ , np.sqrt(fluctuation), length_x) 

        # main equation describing fluctuating selection:
        x = x +  (x * (1 - x) * (σ - τ)) / (1 + (x * σ) + (τ * (1 - x)))
        
        #check if selection coefficient is pushing too much, adjust frequency(x):    
        x[(x < 0)] = 0
        x[(x > 1)] = 1
        
        # Describing drift:
        allele_counts = np.random.binomial(N, x)
        x = allele_counts / N
                       
        #checking if frequency hits the boundry (0) and if a mutation is happening with rate mu:
        mutation_condition = (x == 0) & (np.random.rand(length_x) <= mu * N)
        x[mutation_condition] = 1 / N
         
        x[(x == 1)] = 0

    return x 

#initial value for describing phenomenon:
N = 1000
x0 = 0.01
generations = 10 * N
mu = 1 / (10 * N)

#initial value to decribe flactuating selection:
fluctuation_values = np.linspace(0, 1e-1, 20)
bias_values = np.linspace(0, 3e-3, 20)

#In case of simulating article's claim:
#bias_values = [0]

#%%
#Matrix to store results for vectorized bias and selective fluctuation:
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size
 
output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

for i, fluctuation in enumerate(fluctuation_values):
    
    for j, bias in enumerate(bias_values):
        
        mean_σ = [-bias/2]
        mean_τ = [bias/2]
        
        for batch in range(num_batches):
                
            batch_length_x = Wright_Fisher_model(N, x0, generations, mu, fluctuation, length_x, mean_σ, mean_τ, bias)
            
            output_filename = f"{output_directory}\\x_batch{batch}_fluctuation={fluctuation}_bias={bias}.txt"
            
            np.savetxt(output_filename, batch_length_x, delimiter=',', fmt='%f')

#%%
#defining the analytical solution function: 
def r1(β):
    return (1 - (np.sqrt(1 + (4 / β))))/ 2

def r2(β):
    return (1 + (np.sqrt(1 + (4 / β)))) / 2

def k(β):
    return np.log(((1 - r1(β)) / (-r1(β))) * (r2(β) / (r2(β) - 1)))

def g(y, β):
    return np.log(((1 - r1(β)) / (y - r1(β))) * ((r2(β) - y) / (r2(β) - 1)))

def f1(y, β):
    return (2 / (k(β) * y * (1 - y))) * g(y, β)

#%%
#plotting process:
output_directory =  r"C:\Users\Zahra\research codes -  fluctuating selection"
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, fluctuation in enumerate(fluctuation_values):
    
    color = color_cycle[i % len(color_cycle)]
        
    for batch in range(num_batches):
        
        loaded_data = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={fluctuation}_bias={bias}.txt", delimiter=',')

        # Define bin edges and compute the histogram:
        bin_width = np.linspace((2 / N) , 1, 101)
    
        counts, bins = np.histogram(loaded_data, bins=bin_width)
    
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
        riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))
    
        normalized_counts = counts / riemann_sum
    
        plt.plot(bin_centers, normalized_counts, color=color)
        
#%%
output_directory =  r"C:\Users\Zahra\research codes -  fluctuating selection"
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, fluctuation in enumerate(fluctuation_values):
    
    color = color_cycle[i % len(color_cycle)]
    
    # Initialize an empty array to collect allele frequency data from all batches for each fluctuation:
    all_data = []

    for batch in range(num_batches):
        
        loaded_data = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={fluctuation}_bias={bias}.txt", delimiter=',')
    
        all_data.append(loaded_data)

    all_data = np.concatenate(all_data)

    # Create a histogram of the combined data:      
    bin_width = np.linspace((2 / N) , 1, 101)

    counts, bins = np.histogram(all_data, bins=bin_width)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))

    all_normalized_counts = counts / riemann_sum

    print(f"Area under simulation curve {np.sum( all_normalized_counts * (bin_centers[1] - bin_centers[0]))}")

    plt.plot(bin_centers, all_normalized_counts, marker='o' , label=f'all Data_fluctuation={fluctuation}_bias={bias}', color=color)

plt.legend()
plt.xlabel("Frequency")
plt.ylabel("Normalized Counts")
plt.title("Normalized Frequency Distribution")
plt.show()


#plotting analytical answer:
for i, fluctuation in enumerate(fluctuation_values):
 
    color = color_cycle[i % len(color_cycle)]
      
    β = N * 2 * fluctuation
    
    f1_values = f1(bin_centers, β)  
    
    riemann_sum_analytical = np.sum(f1_values * (bin_centers[1] - bin_centers[0]))
    
    normalized_curve = f1_values / riemann_sum_analytical
    
    print(f"Area under analytical solution curve for fluctuation={fluctuation}_bias={bias}:{np.sum(normalized_curve) * (bin_centers[1] - bin_centers[0])}")
    
    plt.plot(bin_centers, normalized_curve, linestyle='--', label=f'Analytical fluctuation={fluctuation}_bias={bias}',color=color)

plt.xlabel('Frequency')
plt.ylabel('Normalized Counts / Normalized Analytical Values')
plt.title('SFS Fluctuating Selection & Normalized Analytical Solution')
plt.legend()
plt.show()

#%%
#evaluating genetic variation(GV) for aggregated data with vectorized bias and selective fluctuation:
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

GV_values = np.zeros((len(fluctuation_values), len(bias_values)))

# Loop over fluctuation values:
for i, fluctuation in enumerate(fluctuation_values):
    
    for j, bias in enumerate(bias_values):
        
        all_data = []

        for batch in range(num_batches):
        
            V = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={fluctuation}_bias={bias}.txt", delimiter=',')
    
            all_data.append(V)

        all_data = np.concatenate(all_data)
 
        GV = (1 / len(all_data)) * 2 * np.sum(all_data * (1 - all_data))
       
        print(f"GV for fluctuation = {fluctuation}- bias = {bias} : {GV}")
 
        GV_values[i, j] = GV

# Plotting the heatmap:
plt.imshow(GV_values, extent=[min(bias_values), max(bias_values), min(fluctuation_values), max(fluctuation_values)], aspect='auto', origin='lower')
plt.colorbar(label='Genetic Variation (GV)')
plt.xlabel('bias values')
plt.ylabel('fluctuation values')
plt.title('Genetic Variation')
plt.show()


#%%
#evaluating genetic variation in one dimension to check bias = 0 case:
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

GV_values = []

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()

for i, fluctuation in enumerate(fluctuation_values):    
    GV_values_batch = []  
        
    for batch in range(num_batches):
  
        V = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={fluctuation}_bias={bias}.txt", delimiter=',')

        GV_batch = (1 / len(V)) * 2 * np.sum(V * (1 - V))
        print(f"GV for fluctuation = {fluctuation} _ batch = {batch}: {GV_batch}")
        GV_values_batch.append(GV_batch)
        plt.text(fluctuation, GV_batch, f'{batch}', fontsize=8, ha='right', va='bottom')

   
    color = color_cycle[i % len(color_cycle)]
    plt.scatter([fluctuation] * len(GV_values_batch), GV_values_batch, marker='o', color=color)

    GV = (1 / len(V)) * 2 * np.sum(V * (1 - V))
    print(f"GV for fluctuation = {fluctuation}: {GV}")
    GV_values.append(GV)
    plt.text(fluctuation, GV, f'{fluctuation:.2e}', fontsize=8, ha='right', va='bottom')
plt.scatter(fluctuation_values, GV_values, marker='o', color='black')

plt.xlabel('fluctuation')
plt.ylabel('GV')
plt.title('Genetic Variation (GV) vs. Fluctuating Selection (fluctuation)')
plt.grid(True)
plt.show()

#%%
# Record the end time
end_time = time.time()

# Calculate the total running time
running_time = end_time - start_time
print("Total running time:", running_time, "seconds")

