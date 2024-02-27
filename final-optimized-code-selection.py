#%%
import numpy as np
import matplotlib.pyplot as plt
import random as random
import time

#%%
#defining the function: 
start_time = time.time()

def Wright_Fisher_model(N, x0, generations, mu, v, length_x , mean_σ, mean_τ, δ):
    
    x = np.full(length_x , x0)
    
    for i in range(generations):
        
        # describing parameters for fluctuating selection:
        σ = np.random.normal(mean_σ , np.sqrt(v), length_x) 
        τ = np.random.normal(mean_τ , np.sqrt(v), length_x) 

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
v_values = np.linspace(0, 1e-1, 20)
δ_values = np.linspace(0, 3e-3, 20)


for j, δ in enumerate(δ_values):
    
    mean_σ = [-δ/2]
    mean_τ = [δ/2]

#In case of simulating article's claim:
#δ_values = [0]

#%%
#Matrix to store results for vectorized bias = δ and selective fluctuation = v:
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size
 
output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        mean_σ = [-δ/2]
        mean_τ = [δ/2]
        
        for batch in range(num_batches):
                
            batch_length_x = Wright_Fisher_model(N, x0, generations, mu, v, length_x, mean_σ, mean_τ, δ)
            
            output_filename = f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt"
            
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

for i, v in enumerate(v_values):
    
    color = color_cycle[i % len(color_cycle)]
        
    for batch in range(num_batches):
        
        loaded_data = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt", delimiter=',')

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
plt.figure()

for i, v in enumerate(v_values):
    
    color = color_cycle[i % len(color_cycle)]
    
    # Initialize an empty array to collect allele frequency data from all batches for each v:
    all_data = []

    for batch in range(num_batches):
        
        loaded_data = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt", delimiter=',')
    
        all_data.append(loaded_data)

    all_data = np.concatenate(all_data)

    # Create a histogram of the combined data:      
    bin_width = np.linspace((2 / N) , 1, 101)

    counts, bins = np.histogram(all_data, bins=bin_width)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    riemann_sum = np.sum(counts * (bin_centers[1] - bin_centers[0]))

    all_normalized_counts = counts / riemann_sum

    print(f"Area under simulation curve {np.sum( all_normalized_counts * (bin_centers[1] - bin_centers[0]))}")

    plt.plot(bin_centers, all_normalized_counts, marker='o' , label=f'all Data_fluctuation={v}_bias={δ}', color=color)

plt.legend()
plt.xlabel("Frequency")
plt.ylabel("Normalized Counts")
plt.title("Normalized Frequency Distribution")
plt.show()


#plotting analytical answer:
for i, v in enumerate(v_values):
 
    color = color_cycle[i % len(color_cycle)]
      
    β = N * 2 * v
    
    f1_values = f1(bin_centers, β)  
    
    riemann_sum_analytical = np.sum(f1_values * (bin_centers[1] - bin_centers[0]))
    
    normalized_curve = f1_values / riemann_sum_analytical
    
    print(f"Area under analytical solution curve for fluctuation={v}_bias={δ}:{np.sum(normalized_curve) * (bin_centers[1] - bin_centers[0])}")
    
    plt.plot(bin_centers, normalized_curve, linestyle='--', label=f'Analytical fluctuation={v}_bias={δ}',color=color)

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

GV_values = np.zeros((len(v_values), len(δ_values)))

# Loop over fluctuation values:
for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        all_data = []

        for batch in range(num_batches):
        
            V = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt", delimiter=',')
    
            all_data.append(V)

        all_data = np.concatenate(all_data)
 
        GV = (1 / len(all_data)) * 2 * np.sum(all_data * (1 - all_data))
       
        print(f"GV for fluctuation = {v}- bias = {δ} : {GV}")
 
        GV_values[i, j] = GV

# Plotting the heatmap:
plt.imshow(GV_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower')
plt.colorbar(label='Genetic Variation (GV)')
plt.xlabel('bias values')
plt.ylabel('fluctuation values')
plt.title('Genetic Variation')
plt.show()

#between bias and fluctuation:
v_values = δ_values

plt.plot(δ_values, v_values , color='white')

plt.show()

#between bias and drift = bias and fluctuation - all together:
v_values = 1 / ( N * ((1 - (1 / (N * δ_values)))**2))

plt.plot(δ_values, v_values , color='white')

plt.ylim(0, 0.1)

plt.show()

#%%
# Record the end time
end_time = time.time()

# Calculate the total running time
running_time = end_time - start_time
print("Total running time:", running_time, "seconds")


#%%
#equation between bias and drift = drift and fluctuation - all together : 
    
#first way:   
#difference function = a two variable function of v and δ
#initial value to decribe flactuating selection:
v_values = np.linspace(0+1e-100, 1e-1, 20)
δ_values = np.linspace(0+1e-100, 3e-3, 20)
N = 1000

values = []

for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        d = (1 / (2 * N * δ)) - ((1 - np.sqrt( 1 / (N * v)))/ 2 )
        
        if d == 0 :
            
           print("Found d = 0 for (v, δ):", (v, δ))
           
           values.append((v, δ))
           
        else:
           print(False)
           
#%%
#second way:
#difference function if we write v based on δ : (not my favorite)
v_values = 1 / ((N * (1 - (1 / (N * δ_values)))**2))

plt.plot(δ_values, v_values, color = 'red')

plt.xlabel('δ_values')
plt.ylabel('v_values')
plt.show()

#%%
#3D plot of GV and difference for all values
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

v_values = np.linspace(0, 1e-1, 20)
δ_values = np.linspace(0, 3e-3, 20)

GV_values = np.zeros((len(v_values), len(δ_values)))
d_values = np.zeros((len(v_values), len(δ_values)))

# Loop over fluctuation values:
for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        all_data = []
        
        for batch in range(num_batches):
            
            V = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt", delimiter=',')
            
            all_data.append(V)
            
        all_data = np.concatenate(all_data)
        
        GV = (1 / len(all_data)) * 2 * np.sum(all_data * (1 - all_data))
        
        d = (1 / (2 * N * δ)) - ((1 - np.sqrt( 1 / (N * v)))/ 2 )
        
        GV_values[i, j] = GV
        d_values[i, j] = d


# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for v_values and δ_values
V, δ = np.meshgrid(v_values, δ_values)

# Plot the 3D surface for GV_values
surf1 = ax.plot_surface(V, δ, GV_values, cmap='viridis', label='Genetic Variation (GV)')

# Plot the 3D surface for d_values
surf2 = ax.plot_surface(V, δ, d_values, cmap='plasma', label='d function')

# Add labels and title
ax.set_xlabel('Fluctuation Values')
ax.set_ylabel('Bias Values')
ax.set_zlabel('Values')
ax.set_title('Genetic Variation and d Function')

# Add color bars
fig.colorbar(surf1, shrink=0.5, aspect=5, label='Genetic Variation (GV)')
fig.colorbar(surf2, shrink=0.5, aspect=5, label='d function')

plt.show()

#%%
#3D plot of GV and difference when 0 < d < 0.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

GV_values = np.zeros((len(v_values), len(δ_values)))
d_values = np.zeros((len(v_values), len(δ_values)))

# Loop over fluctuation values:
for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        all_data = []
        
        for batch in range(num_batches):
            
            V = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt", delimiter=',')
            
            all_data.append(V)
            
        all_data = np.concatenate(all_data)
        
        GV = (1 / len(all_data)) * 2 * np.sum(all_data * (1 - all_data))
        
        d = (1 / (2 * N * δ)) - ((1 - np.sqrt( 1 / (N * v)))/ 2 )
        
        print(f"GV for fluctuation = {v}- bias = {δ} : {GV}")
        
        GV_values[i, j] = GV
        d_values[i, j] = d
        
# Masking d values outside the range [0, 0.1]
masked_d_values = np.ma.masked_where((d_values < 0) | (d_values > 0.1), d_values)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for v_values and δ_values
V, δ = np.meshgrid(v_values, δ_values)

# Plot the 3D surface for GV_values
surf1 = ax.plot_surface(V, δ, GV_values, cmap='viridis', label='Genetic Variation (GV)')

# Plot the 3D surface for masked_d_values
surf2 = ax.plot_surface(V, δ, masked_d_values, cmap='plasma', label='d function')

# Add labels and title
ax.set_xlabel('Fluctuation Values')
ax.set_ylabel('Bias Values')
ax.set_zlabel('Values')
ax.set_title('Genetic Variation and d Function')

# Add color bars
fig.colorbar(surf1, shrink=0.5, aspect=5, label='Genetic Variation (GV)')
fig.colorbar(surf2, shrink=0.5, aspect=5, label='d function')

plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

v_values = np.linspace(0, 1e-1, 20)
δ_values = np.linspace(0, 3e-3, 20)

GV_values = np.zeros((len(v_values), len(δ_values)))
d_values = np.zeros((len(v_values), len(δ_values)))

# Loop over fluctuation values:
for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        all_data = []
        
        for batch in range(num_batches):
            
            V = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt", delimiter=',')
            
            all_data.append(V)
            
        all_data = np.concatenate(all_data)
        
        GV = (1 / len(all_data)) * 2 * np.sum(all_data * (1 - all_data))
        
        d = (1 / (2 * N * δ)) - ((1 - np.sqrt( 1 / (N * v)))/ 2 )
        
        GV_values[i, j] = GV
        d_values[i, j] = d

# Plotting both heatmaps in one figure
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot heatmap for GV_values
im1 = axs[0].imshow(GV_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower')
axs[0].set_xlabel('Bias Values')
axs[0].set_ylabel('Fluctuation Values')
axs[0].set_title('Genetic Variation (GV)')
plt.colorbar(im1, ax=axs[0], label='Genetic Variation (GV)')

# Plot heatmap for d_values
im2 = axs[1].imshow(d_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower')
axs[1].set_xlabel('Bias Values')
axs[1].set_ylabel('Fluctuation Values')
axs[1].set_title('d Function')
plt.colorbar(im2, ax=axs[1], label='d function')

plt.tight_layout()
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

v_values = np.linspace(0, 1e-1, 20)
δ_values = np.linspace(0, 3e-3, 20)

GV_values = np.zeros((len(v_values), len(δ_values)))
d_values = np.zeros((len(v_values), len(δ_values)))

# Loop over fluctuation values:
for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        all_data = []
        
        for batch in range(num_batches):
            
            V = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt", delimiter=',')
            
            all_data.append(V)
            
        all_data = np.concatenate(all_data)
        
        GV = (1 / len(all_data)) * 2 * np.sum(all_data * (1 - all_data))
        
        d = (1 / (2 * N * δ)) - ((1 - np.sqrt( 1 / (N * v)))/ 2 )
        
        print(f"GV for fluctuation = {v}- bias = {δ} : {GV}")
        
        GV_values[i, j] = GV
        d_values[i, j] = d
        
# Masking d values outside the range [0, 0.1]
masked_d_values = np.ma.masked_where((d_values < 0) | (d_values > 0.1), d_values)

# Plotting the heatmaps
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot heatmap for GV_values
im1 = axs[0].imshow(GV_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower', cmap='viridis')
axs[0].set_xlabel('Bias Values')
axs[0].set_ylabel('Fluctuation Values')
axs[0].set_title('Genetic Variation (GV)')
plt.colorbar(im1, ax=axs[0], label='Genetic Variation (GV)')

# Plot heatmap for masked_d_values
im2 = axs[1].imshow(masked_d_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower', cmap='plasma')
axs[1].set_xlabel('Bias Values')
axs[1].set_ylabel('Fluctuation Values')
axs[1].set_title('d Function (0 < d < 0.1)')
plt.colorbar(im2, ax=axs[1], label='d function')

plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

v_values = np.linspace(0, 1e-1, 20)
δ_values = np.linspace(0, 3e-3, 20)

GV_values = np.zeros((len(v_values), len(δ_values)))
d_values = np.zeros((len(v_values), len(δ_values)))

# Loop over fluctuation values:
for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        all_data = []
        
        for batch in range(num_batches):
            
            V = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt", delimiter=',')
            
            all_data.append(V)
            
        all_data = np.concatenate(all_data)
        
        GV = (1 / len(all_data)) * 2 * np.sum(all_data * (1 - all_data))
        
        d = (1 / (2 * N * δ)) - ((1 - np.sqrt( 1 / (N * v)))/ 2 )
        
        print(f"GV for fluctuation = {v}- bias = {δ} : {GV}")
        
        GV_values[i, j] = GV
        d_values[i, j] = d
        
# Masking d values outside the range [0, 0.1]
masked_d_values = np.ma.masked_where((d_values < 0) | (d_values > 0.1), d_values)

# Plotting the heatmaps on the same plot
plt.figure(figsize=(10, 6))

# Plot heatmap for GV_values
plt.imshow(GV_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Genetic Variation (GV)')

# Plot heatmap for masked_d_values with alpha value to make it transparent
plt.imshow(masked_d_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower', cmap='plasma', alpha=0.5)
plt.colorbar(label='d function')

plt.xlabel('Bias Values')
plt.ylabel('Fluctuation Values')
plt.title('Genetic Variation and d Function')
plt.show()

