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
#plotting SFS:
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
       
        #print(f"GV for fluctuation = {v}- bias = {δ} : {GV}")
 
        GV_values[i, j] = GV

# Plotting the heatmap:
plt.imshow(GV_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower')
plt.colorbar(label='Genetic Variation (GV)')
plt.xlabel('bias values')
plt.ylabel('fluctuation values')
plt.title('Genetic Variation')
plt.show()

#%%
# Record the end time
end_time = time.time()

# Calculate the total running time
running_time = end_time - start_time
print("Total running time:", running_time, "seconds")
      

#%%
#difference function heatmap:
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

v_values = np.linspace(0, 1e-1, 20)
δ_values = np.linspace(0, 3e-3, 20)

GV_values = np.zeros((len(v_values), len(δ_values)))
d_values = np.zeros((len(v_values), len(δ_values)))


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
        
# Masking d values outside the range [0, 0.1]
masked_d_values = np.ma.masked_where((d_values < 0) | (d_values > 0.1), d_values)

# Plotting all heatmaps in one figure
fig, axs = plt.subplots(1, 3, figsize=(10 , 5))

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


# Plot heatmap for masked_d_values
im3 = axs[2].imshow(masked_d_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower', cmap='plasma')
axs[2].set_xlabel('Bias Values')
axs[2].set_ylabel('Fluctuation Values')
axs[2].set_title('d Function (0 < d < 0.1)')
plt.colorbar(im2, ax=axs[2], label='d function')

plt.tight_layout()
plt.show()

#%%
#ratio function heatmap:
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size

output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"

v_values = np.linspace(0, 1e-1, 20)
δ_values = np.linspace(0, 3e-3, 20)

GV_values = np.zeros((len(v_values), len(δ_values)))
r_values = np.zeros((len(v_values), len(δ_values)))


for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        all_data = []
        
        for batch in range(num_batches):
            
            V = np.loadtxt(f"{output_directory}\\x_batch{batch}_fluctuation={v}_bias={δ}.txt", delimiter=',')
            
            all_data.append(V)
            
        all_data = np.concatenate(all_data)
        
        GV = (1 / len(all_data)) * 2 * np.sum(all_data * (1 - all_data))
        
        r = (1 / ((N * δ) * (1 - np.sqrt( 1 / (N * v)))))
        
        GV_values[i, j] = GV       
        r_values[i, j] = r
        
# Masking r values outside the range [0, 2]
masked_r_values = np.ma.masked_where((r_values < 0) | (r_values > 2), r_values)

# Plotting all heatmaps in one figure
fig, axs = plt.subplots(1, 3, figsize=(10 , 5))

# Plot heatmap for GV_values
im1 = axs[0].imshow(GV_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower')
axs[0].set_xlabel('Bias Values')
axs[0].set_ylabel('Fluctuation Values')
axs[0].set_title('Genetic Variation (GV)')
plt.colorbar(im1, ax=axs[0], label='Genetic Variation (GV)')

# Plot heatmap for r_values
im2 = axs[1].imshow(r_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower')
axs[1].set_xlabel('Bias Values')
axs[1].set_ylabel('Fluctuation Values')
axs[1].set_title('r Function')
plt.colorbar(im2, ax=axs[1], label='ratio function')


# Plot heatmap for masked_r_values
im3 = axs[2].imshow(masked_r_values, extent=[min(δ_values), max(δ_values), min(v_values), max(v_values)], aspect='auto', origin='lower', cmap='plasma')
axs[2].set_xlabel('Bias Values')
axs[2].set_ylabel('Fluctuation Values')
axs[2].set_title('r Function (0 < r < 2)')
plt.colorbar(im2, ax=axs[2], label='ratio function')

plt.tight_layout()
plt.show()

#%%
#trying to plot contours for difference 
v_values = np.linspace(0+1e-10, 1e-1, 20)
δ_values = np.linspace(0+1e-10, 3e-3, 20)

Δ, V = np.meshgrid(δ_values, v_values)

d_values = np.zeros((len(v_values), len(δ_values)))

for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        d = (1 / (2 * N * δ)) - ((1 - np.sqrt(1 / (N * v))) / 2)
        
        d_values[i, j] = d

# Contour levels
# levels = np.linspace(np.min(d_values), np.max(d_values), 50)
levels = [0]

# Plot
fig, ax = plt.subplots()
contour = ax.contour(Δ, V, d_values,levels = levels)
ax.clabel(contour, inline=True, fontsize=8)  
ax.set_xlabel('δ values')
ax.set_ylabel('v values')
plt.title('Contours for Difference')

# Set logarithmic scale for axes
# plt.xscale('log')
# plt.yscale('log')

plt.show()

#%%
#trying to plot contours for ratio function
v_values = np.linspace(0+1e-10, 1e-1, 20)
δ_values = np.linspace(0+1e-10, 3e-3, 20)

Δ , V = np.meshgrid(δ_values, v_values)

r_values = np.zeros((len(v_values), len(δ_values)))

for i, v in enumerate(v_values):
    
    for j, δ in enumerate(δ_values):
        
        r = (1 / ((N * δ) * (1 - np.sqrt( 1 / (N * v)))))
        
        r_values[i, j] = r

levels = np.linspace(np.min(r_values), np.max(r_values), 100)
# levels = [1]

# Plot
fig, ax = plt.subplots()
contour = ax.contour(Δ, V, r_values,levels = levels)
ax.clabel(contour, inline=True, fontsize = 8)  
ax.set_xlabel('δ values')
ax.set_ylabel('v values')
plt.title('Contours for ratio')

# Set logarithmic scale for axes
plt.xscale('log')
plt.yscale('log')

plt.show()

#%%
# Defining functions
# def Φ1(x, δ, v):
#     return (1/(4 * x * (1 - x))) * ((x/(1 - x))**(-δ/v))
# Define Φ1 and Φ2 functions
def Φ1(x, δ, v):
    if x == 0 or x == 1 or v == 0:
        return 0  
    else:
        return np.log((1/(4 * x * (1 - x))) * ((x/(1 - x))**(-δ/v)))

def Φ2(x, δ, N):
    return np.log((1/(4 * x * (1 - x))) * (np.exp(N * δ * (-2 * x + 1))))

#%%
# Initialize lists
Φ1_values = []
Φ2_values = []

# Define parameters
N = 1000
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size
output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"
v_values = np.linspace(0, 1e-1, 20)
δ_values = np.linspace(0, 3e-3, 20)
X = np.linspace(0,0.5, 100)

#reading the function for given parameters
Φ1_batch = [Φ1(x, δ, v) for x in X]
Φ1_values.extend(Φ1_batch)
        
Φ2_batch = [Φ2(x, δ, N) for x in X]
Φ2_values.extend(Φ2_batch)
       
# Plot ln(Φ1) and ln(Φ2) with respect to Data
plt.figure()
plt.scatter(X, Φ1_values, label='ln(Φ1)-bias and fluctuation - polynomial')
plt.scatter(X, Φ2_values, label='ln(Φ2)-bias and drift - exponential')
plt.xlabel('x')
plt.ylabel('ln(Φ(x))')
plt.legend()
plt.title('ln(Φ1) and ln(Φ2) vs frequency')
plt.show()

#%%
# Define only the differences between Φ1 and Φ2 functions
def Φ1d(x, δ, v):
    if x == 0 or x == 1 or v == 0:
        return 0  
    else:
        return np.log(((x/(1 - x))**(-δ/v)))

def Φ2d(x, δ, N):
    return np.log((np.exp(N * δ * (-2 * x + 1))))

#%%
# Initialize lists
Φ1d_values = []
Φ2d_values = []
all_data = []

# Define parameters
N = 1000
length_x = 10**4
batch_size = 10**4
num_batches = length_x // batch_size
output_directory = r"C:\Users\Zahra\research codes -  fluctuating selection"
v_values = np.linspace(0, 1e-1, 20)
δ_values = np.linspace(0, 3e-3, 20)
X = np.linspace(0, 0.5, 100)

# Calculate Φ1 and Φ2 for the filtered data
Φ1d_batch = [Φ1d(x, δ, v) for x in X]
Φ1d_values.extend(Φ1d_batch)
        
Φ2d_batch = [Φ2d(x, δ, N) for x in X]
Φ2d_values.extend(Φ2d_batch)
       
# Plot ln(Φ1) and ln(Φ2) with respect to Data
plt.figure()
plt.scatter(X, Φ1d_values, label='ln(Φ1d)-bias and fluctuation - polynomial')
plt.scatter(X, Φ2d_values, label='ln(Φ2d)-bias and drift - exponential')
plt.xlabel('x')
plt.ylabel('ln(Φ(x))')
plt.legend()
plt.title('different part of ln(Φ1) and ln(Φ2) vs frequency')
plt.show()












