import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# configuration 
# all variables are strictly lowercase
particle_radius_0 = 2.385e-6
radius_factors = [0.4, 1.0, 1.6]
passive_densities = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
sims_per_condition = 15

data_directory = "/mnt/c/Users/stefa/OneDrive/Desktop/CFTC/Negative_torque_56/"
n_active = 50
record_intervals = 1

# frame_step = 1 means we measure c_max every 1 second of the simulation
frame_step = 1
time_to_simulate = 3000  
expected_frames = int(time_to_simulate / record_intervals)

# colormap 
colors = plt.cm.viridis(np.linspace(0, 0.9, len(passive_densities)))

# temporal sweep 
for idx_factor, factor in enumerate(radius_factors):
    
    # create a new figure for each radius factor
    fig, ax = plt.subplots(figsize=(8, 6))
    
    current_radius = factor * particle_radius_0
    box_size = 120 * current_radius
    cluster_cutoff_radius = 2.2 * current_radius
    
    for idx_dens, dens in enumerate(passive_densities):
        print(f"Calculating evolution: R={factor}R0 | Density={dens*100}%")
        
        c_max_all_replicas = []
        
        for i in range(1, sims_per_condition + 1):

            if factor == 1.6 and dens == 0.50 and i > 10:
                continue

            filename = f"sim_pas_{dens:.2f}_R_{factor:.1f}R0_run_{i}.bin"
            file_path = os.path.join(data_directory, filename)
            
            if not os.path.exists(file_path):
                continue
                
            raw_data = np.fromfile(file_path, dtype=np.float32)
            
            # discard corrupted files 
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
            
            c_max_series = []
            
            # sampling from t=0 to the end 
            for j in range(0, expected_frames, frame_step):
                positions = active_particles[j, :, 0:2]
                tree = cKDTree(positions, boxsize=box_size)
                pairs = tree.query_pairs(cluster_cutoff_radius)
                
                if not pairs:
                    c_max_series.append(0)
                    continue
                    
                rows, cols = zip(*pairs)
                matrix_data = np.ones(len(rows))
                adjacency_matrix = coo_matrix((matrix_data, (rows, cols)), shape=(n_active, n_active))
                _, labels = connected_components(csgraph=adjacency_matrix, directed=False)
                counts = np.bincount(labels)
                valid_groups = counts[counts >= 3]
                
                c_max = valid_groups.max() if len(valid_groups) > 0 else 0
                c_max_series.append(c_max)
                
            c_max_all_replicas.append(c_max_series)
            
        if c_max_all_replicas:
            n_valid = len(c_max_all_replicas)
            # calculate the average c_max over time across available replicas
            c_max_avg = np.mean(c_max_all_replicas, axis=0)
            time_axis = np.arange(0, expected_frames, frame_step) * record_intervals
            
            ax.plot(time_axis, c_max_avg, color=colors[idx_dens], linewidth=2.5, 
                    label=f'{dens*100:.0f}% ', alpha=0.85)
            
    # individual subplot formatting 
    ax.set_title(r'Radius $R = {}$ $R_0$'.format(factor), fontsize=15, pad=10)
    ax.set_xlabel(r'$\tau$ (s)', fontsize=13)
    ax.set_ylabel(r'$\langle C_{max} \rangle$', fontsize=14) 
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # place the legend centered below the plot for every generated figure
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

    plt.suptitle(r'Temporal evolution of cluster size for steady-state identification', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # save dynamically per radius factor
    output_name = f"evolucion_temporal_Cmax_R_{factor}.png"
    output_path = os.path.join(data_directory, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"-> Individual plot successfully generated! Saved at: {output_path}")
    
    plt.show()
    plt.close(fig)