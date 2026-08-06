import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


sims_per_condition = 15 
data_directory = "/mnt/c/Users/stefa/OneDrive/Desktop/CFTC/Negative_torque_56/"

plots_folder = os.path.join(data_directory, "final_plots", "phi_distributions")
os.makedirs(plots_folder, exist_ok=True)

# configuration 
particle_radius_0 = 2.385e-6
radius_factors = [0.4, 1.0, 1.6]

# we keep 5% steps so histograms don't overlap and remain readable
passive_densities = [round(d * 0.05, 2) for d in range(11)]
n_active = 50 
record_intervals = 1

# we keep frame_step = 1 for maximum resolution in the histograms
frame_step = 1 
time_to_simulate = 3000  
expected_frames = int(time_to_simulate / record_intervals)

# 
steady_state_time = 0 
cutoff_frame = int(steady_state_time / record_intervals)

# palette (magenta, purple, blue) 
plot_styles = {
    0.4: {'color': '#DC267F', 'label': '$R = 0.4 R_0$'}, 
    1.0: {'color': '#785EF0', 'label': '$R = 1.0 R_0$'}, 
    1.6: {'color': '#648FFF', 'label': '$R = 1.6 R_0$'}  
}

# main loop per density 
for density in passive_densities:
    print(f"\n--- Generating distribution for Density = {density*100:.0f}% ---")
    
    phi_distributions = {factor: [] for factor in radius_factors}
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for factor in radius_factors:
        current_radius = factor * particle_radius_0
        box_size = 120 * current_radius
        cluster_cutoff_radius = 2.2 * current_radius
        
        for i in range(1, sims_per_condition + 1):
            filename = f"sim_pas_{density:.2f}_R_{factor:.1f}R0_run_{i}.bin"
            file_path = os.path.join(data_directory, filename)
            
            if not os.path.exists(file_path): 
                continue
            
            raw_data = np.fromfile(file_path, dtype=np.float32)
            
            if len(raw_data) % expected_frames == 0:
                num_frames = expected_frames
            elif len(raw_data) % (expected_frames + 1) == 0:
                num_frames = expected_frames + 1
            else:
                print(f"  -> Warning: Incomplete or corrupted file skipped ({file_path})")
                continue
            
            floats_per_frame = len(raw_data) // num_frames
            n_passive = (floats_per_frame - (3 * n_active)) // 2
            cutoff_index = 2 * n_passive
            
            data_2d = raw_data.reshape(num_frames, floats_per_frame)
            active_particles = data_2d[:, cutoff_index:].reshape(num_frames, n_active, 3)
            
            steady_active_particles = active_particles[cutoff_frame:, :, :]
            
            for j in range(0, len(steady_active_particles), frame_step):
                pos = steady_active_particles[j, :, 0:2]
                angles = steady_active_particles[j, :, 2]
                
                if np.max(pos) >= box_size or np.min(pos) < 0:
                    break 
                
                pairs = cKDTree(pos, boxsize=box_size).query_pairs(cluster_cutoff_radius)
                if not pairs: 
                    continue
                
                rows, cols = zip(*pairs)
                adjacency_matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_active, n_active))
                _, labels = connected_components(csgraph=adjacency_matrix, directed=False)
                counts = np.bincount(labels)
                
                valid_groups = np.where(counts >= 7)[0]
                if len(valid_groups) == 0: 
                    continue
                
                main_label = valid_groups[np.argmax(counts[valid_groups])]
                cluster_idx = np.where(labels == main_label)[0]
                
                cluster_pos = pos[cluster_idx]
                cluster_angles = angles[cluster_idx]
                cluster_size = len(cluster_idx)
                
                p0 = cluster_pos[0]
                displacement = cluster_pos - p0
                displacement -= box_size * np.round(displacement / box_size)
                pos_unwrap = p0 + displacement
                
                com = np.mean(pos_unwrap, axis=0)
                r_vec = pos_unwrap - com
                distances = np.linalg.norm(r_vec, axis=1)
                distances[distances == 0] = 1e-10 
                
                r_hat_x = r_vec[:, 0] / distances
                r_hat_y = r_vec[:, 1] / distances
                
                n_hat_x = np.cos(cluster_angles)
                n_hat_y = np.sin(cluster_angles)
                
                phi_i = (r_hat_x * n_hat_y) - (r_hat_y * n_hat_x)
                phi_cluster = np.sum(phi_i) / cluster_size
                
                phi_distributions[factor].append(phi_cluster)

    # step histograms
    bins = np.linspace(-1, 1, 100)
    
    for factor in radius_factors:
        data = phi_distributions[factor]
        if not data:
            continue
            
        weights = np.ones_like(data) / len(data)
        ax.hist(data, bins=bins, weights=weights, color=plot_styles[factor]['color'], 
                label=plot_styles[factor]['label'], histtype='step', linewidth=2.5)

    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(r'$\Phi$', fontsize=18)
    ax.set_ylim(0, 0.15) 
    ax.set_ylabel(r'$P(\Phi)$', fontsize=18)
    
    
    ax.set_title(r'Rotational Order at $\rho_p = {:.0f}\%$'.format(density*100), fontsize=15, pad=15)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle=':', alpha=0.6) 

    ax.legend(fontsize=13, frameon=True, loc='upper right')
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    output_filename = f"figure_3g_step_density_{density*100:.0f}.png"
    output_path = os.path.join(plots_folder, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

print("\nDynamics plots successfully redesigned! Check the folder.")