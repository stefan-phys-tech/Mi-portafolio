import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# configuration 
particle_radius_0 = 2.385e-6
radius_factors = [0.4, 1.0, 1.6]
alphas_to_test = [1.8,1.9, 2]
sims_per_condition = 5
data_directory = "/mnt/c/Users/stefa/OneDrive/Desktop/CFTC/"

n_passive = 0
n_active = 50 
record_intervals = 0.1
frame_step = 1000 
floats_per_frame = (2 * n_passive) + (3 * n_active)
cutoff_index = 2 * n_passive 

# --- 2. extraction and individual averaged plotting ---
for alpha in alphas_to_test:
    print(f"\nProcessing and generating plot for alpha = {alpha:.1f}...")
    
    # create a new EXCLUSIVE figure for this alpha
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for factor in radius_factors:
        current_radius = factor * particle_radius_0
        box_size = 120 * current_radius
        cluster_cutoff_radius = 2.2 * current_radius
        
        curves_this_factor = []
        plot_times = None
        
        # read the 5 replicas for each combination of alpha and radius
        for i in range(1, sims_per_condition + 1):
            filename = f"sim_alpha_{alpha:.1f}_R_{factor:.1f}R0_run_{i}.bin"
            file_path = os.path.join(data_directory, filename)
            
            if not os.path.exists(file_path):
                continue
                
            raw_data = np.fromfile(file_path, dtype=np.float32)
            num_frames = len(raw_data) // floats_per_frame
            data_2d = raw_data.reshape(num_frames, floats_per_frame)
            active_particles = data_2d[:, cutoff_index:].reshape(num_frames, n_active, 3)

            if plot_times is None:
                plot_times = (np.arange(num_frames) * record_intervals)[::frame_step]

            c_max_local = []
            for j in range(0, num_frames, frame_step):
                positions = active_particles[j, :, 0:2]
                tree = cKDTree(positions, boxsize=box_size)
                pairs = tree.query_pairs(cluster_cutoff_radius)
                
                if not pairs:
                    c_max_local.append(0)
                    continue
                    
                rows, cols = zip(*pairs)
                matrix_data = np.ones(len(rows))
                adjacency_matrix = coo_matrix((matrix_data, (rows, cols)), shape=(n_active, n_active))
                n_components, labels = connected_components(csgraph=adjacency_matrix, directed=False)
                counts = np.bincount(labels)
                valid_groups = counts[counts >= 3]
                
                c_max = valid_groups.max() if len(valid_groups) > 0 else 0
                c_max_local.append(c_max)
                
            curves_this_factor.append(c_max_local)

        # if we successfully read curves, calculate the statistical mean
        if curves_this_factor:
            min_length = min([len(curve) for curve in curves_this_factor])
            trimmed_curves = [curve[:min_length] for curve in curves_this_factor]
            trimmed_plot_times = plot_times[:min_length]
            
            matrix = np.array(trimmed_curves)
            mean_val = np.mean(matrix, axis=0)
            std_err = np.std(matrix, axis=0)
            
            # plot the clean mean with scatter bars
            ax.errorbar(trimmed_plot_times, mean_val, yerr=std_err, marker='o', linestyle='-', 
                        capsize=3, label=f'{factor} $R_0$')

    # --- 3. individual plot configuration and saving ---
    # titles and labels translated to english
    ax.set_title(r'Phenomenological Collapse with $\alpha =$ ' + f'{alpha:.1f}', fontsize=16)
    ax.set_xlabel(r'$\tau$ (s)', fontsize=14)
    ax.set_ylabel(r'Average $C_{max}$', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    # save the individual plot
    output_filename = f"fine_mesh_collapse_alpha_{alpha:.1f}.png"
    plot_path = os.path.join(data_directory, output_filename)
    fig.savefig(plot_path, dpi=300)
    
    # crucial: close the figure to free RAM and avoid opening multiple windows
    plt.close(fig) 
    
    print(f"  -> Plot saved at: {output_filename}")

print("\nAnalysis completed! The independent images have been saved in your folder.")