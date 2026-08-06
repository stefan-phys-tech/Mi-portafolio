import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# configuration
particle_radius_0 = 2.385e-6
radius_factors = [0.4, 1.0, 1.6]
passive_densities = [round(d * 0.01, 2) for d in range(51)]

sims_per_condition = 15 
data_directory = "/mnt/c/Users/stefa/OneDrive/Desktop/CFTC/Negative_torque_56/"

n_active = 50 
record_intervals = 1
alpha_opt = 1.8

# time automation 
time_to_simulate = 3000  
expected_frames = int(time_to_simulate / record_intervals)

steady_state_time = 2000 
cutoff_frame = int(steady_state_time / record_intervals)

fig, ax = plt.subplots(figsize=(10, 7)) 

# palette (magenta, purple, blue)
plot_styles = {
    0.4: {'color': '#DC267F', 'label': '$R = 0.4 R_0$'},
    1.0: {'color': '#785EF0', 'label': '$R = 1.0 R_0$'},
    1.6: {'color': '#648FFF', 'label': '$R = 1.6 R_0$'}
}

# sweep and data extraction ---
for factor in radius_factors:
    current_radius = factor * particle_radius_0
    box_size = 120 * current_radius
    cluster_cutoff_radius = 2.2 * current_radius
    
    c_max_per_density = []
    errors_per_density = []
    
    for density in passive_densities:
        c_max_replicas = []
        print(f"Processing: R={factor}R0 | Density={density*100:.0f}%")
        
        for i in range(1, sims_per_condition + 1):
            filename = f"sim_pas_{density:.2f}_R_{factor:.1f}R0_run_{i}.bin"
            file_path = os.path.join(data_directory, filename)
            
            if not os.path.exists(file_path):
                continue
                
            raw_data = np.fromfile(file_path, dtype=np.float32)
            
            # automatic discard for incomplete simulations
            if len(raw_data) % expected_frames == 0:
                num_frames = expected_frames
            elif len(raw_data) % (expected_frames + 1) == 0:
                num_frames = expected_frames + 1
            else:
                continue
            
            floats_per_frame = len(raw_data) // num_frames
            n_passive = (floats_per_frame - (3 * n_active)) // 2
            cutoff_index = 2 * n_passive
            
            data_2d = raw_data.reshape(num_frames, floats_per_frame)
            active_particles = data_2d[:, cutoff_index:].reshape(num_frames, n_active, 3)
            
            # apply the global steady-state cutoff
            steady_active_particles = active_particles[cutoff_frame:, :, :]
            
            c_max_temp_replica = []
            frame_step = 25 
            
            for j in range(0, len(steady_active_particles), frame_step):
                positions = steady_active_particles[j, :, 0:2]
                tree = cKDTree(positions, boxsize=box_size)
                pairs = tree.query_pairs(cluster_cutoff_radius)
                
                if not pairs:
                    c_max_temp_replica.append(0)
                    continue
                    
                rows, cols = zip(*pairs)
                matrix_data = np.ones(len(rows))
                adjacency_matrix = coo_matrix((matrix_data, (rows, cols)), shape=(n_active, n_active))
                _, labels = connected_components(csgraph=adjacency_matrix, directed=False)
                counts = np.bincount(labels)
                valid_groups = counts[counts >= 3]
                
                c_max = valid_groups.max() if len(valid_groups) > 0 else 0
                c_max_temp_replica.append(c_max)
                
            if c_max_temp_replica:
                c_max_replicas.append(np.mean(c_max_temp_replica))
            
        if c_max_replicas:
            c_max_per_density.append(np.mean(c_max_replicas))
            # standard error of the mean
            errors_per_density.append(np.std(c_max_replicas) / np.sqrt(len(c_max_replicas))) 
        else:
            c_max_per_density.append(np.nan)
            errors_per_density.append(np.nan)
            
    # convert to numpy arrays to add/subtract the shading
    x_axis_percentage = np.array([d * 100 for d in passive_densities])
    c_max_array = np.array(c_max_per_density)
    errors_array = np.array(errors_per_density)
    
    line_color = plot_styles[factor]['color']
    line_label = plot_styles[factor]['label']
    
    # 1. draw the main solid line
    ax.plot(x_axis_percentage, c_max_array, color=line_color, linestyle='-', linewidth=2.5, label=line_label)
    
    # 2. fill the standard deviation area (shading)
    ax.fill_between(x_axis_percentage, 
                    c_max_array - errors_array, 
                    c_max_array + errors_array, 
                    color=line_color, alpha=0.3, edgecolor='none')

ax.set_xlabel(r'Passive particle density, $\rho_p$ (%)', fontsize=14)
ax.set_ylabel(r'$C_{max}$', fontsize=14)
ax.set_ylim(0, 30) 


ax.set_title(r'Effect of Radius on Cluster Formation', fontsize=15)

ax.legend(fontsize=12, frameon=True)
ax.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
output_path = os.path.join(data_directory, "plot_cmax_vs_density_by_radius_shaded.png")
plt.savefig(output_path, dpi=300)

print(f"\nPlot successfully generated! Saved at:\n{output_path}")
plt.show()