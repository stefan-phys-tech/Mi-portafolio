import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# configuration
particle_radius_0 = 2.385e-6
radius_factors = [0.4, 1.0, 1.6]
passive_densities = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50] 

sims_per_condition = 15 
data_directory = "/mnt/c/Users/stefa/OneDrive/Desktop/CFTC/Negative_torque_56/"

# folder for the output videos
videos_folder = os.path.join(data_directory, "final_plots", "representative_videos")
if not os.path.exists(videos_folder):
    os.makedirs(videos_folder)

n_active = 50 
video_fps = 30
frame_step = 1 # half speed (2x smoothness) to clearly see the dynamics

record_intervals = 1
time_to_simulate = 3000
expected_frames = int(time_to_simulate / record_intervals)
steady_state_time = 2000 # point where c_max has stabilized
cutoff_frame = int(steady_state_time / record_intervals)

print(f"\nStarting massive rendering of representative videos (up to {time_to_simulate}s)...")

for factor in radius_factors:
    current_radius = factor * particle_radius_0
    box_size = 120 * current_radius
    cluster_cutoff_radius = 2.2 * current_radius
    
    for density in passive_densities:
        video_path = os.path.join(videos_folder, f"representative_video_pas_{density:.2f}_r_{factor:.1f}r0.mp4")
        gif_path = video_path.replace('.mp4', '.gif')
        
        if os.path.exists(video_path) or os.path.exists(gif_path):
            print(f"Video already exists for R={factor}R0, Density={density*100:.0f}%. Skipping...")
            continue
            
        print(f"\n--- Analyzing R={factor}R0 | Density={density*100:.0f}% ---")
        
        # find the most representative replica 
        replica_means = []
        valid_files = []
        
        for i in range(1, sims_per_condition + 1):
            file_path = os.path.join(data_directory, f"sim_pas_{density:.2f}_R_{factor:.1f}R0_run_{i}.bin")
            if not os.path.exists(file_path): 
                continue
            
            raw_data = np.fromfile(file_path, dtype=np.float32)
            
            # shielding against corrupted/incomplete files 
            # use expected_frames (3000) to deduce the exact size of 1 frame
            approx_floats = len(raw_data) / expected_frames
            n_passive = int(round((approx_floats - (3 * n_active)) / 2))
            floats_per_frame = int((2 * n_passive) + (3 * n_active))
            
            # calculate how many full frames actually exist
            num_frames = len(raw_data) // floats_per_frame
            
            # safety margin: require at least 95% of the simulation
            if num_frames < expected_frames * 0.95: 
                continue
            
            # trim the final "garbage" (incomplete frames)
            clean_data = raw_data[:num_frames * floats_per_frame]
            cutoff_index = 2 * n_passive
            
            # now the reshape will not fail
            data_2d = clean_data.reshape(num_frames, floats_per_frame)
            
            # dynamic trimming from steady state to the end
            active_particles = data_2d[:, cutoff_index:].reshape(num_frames, n_active, 3)
            steady_active_particles = active_particles[cutoff_frame:, :, :]
            
            c_max_temp = []
            corrupted_replica = False
            
            # evaluate representativeness every 100 frames
            for j in range(0, len(steady_active_particles), 100):
                pos = steady_active_particles[j, :, 0:2]
                
                max_pos = np.max(pos)
                min_pos = np.min(pos)
                
                if max_pos >= box_size or min_pos < 0:
                    print(f"\n      [!] PHYSICS ALERT: Particles out of bounds in {os.path.basename(file_path)}")
                    print(f"          Box size:         {box_size:.6e}")
                    print(f"          Max pos detected: {max_pos:.6e}")
                    print(f"          Min pos detected: {min_pos:.6e}")
                    print("          -> Discarding this replica and moving to the next...\n")
                    corrupted_replica = True
                    break # exit the frame loop
                
                # if everything is fine, calculate the cluster
                pairs = cKDTree(pos, boxsize=box_size).query_pairs(cluster_cutoff_radius)
                if not pairs:
                    c_max_temp.append(0)
                    continue
                    
                rows, cols = zip(*pairs)
                adjacency_matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_active, n_active))
                _, labels = connected_components(csgraph=adjacency_matrix, directed=False)
                counts = np.bincount(labels)
                groups = counts[counts >= 3]
                c_max_temp.append(groups.max() if len(groups) > 0 else 0)
                
            # if the radar detected a failure, skip to the next file
            if corrupted_replica:
                continue
                
            replica_means.append(np.mean(c_max_temp))
            valid_files.append(file_path)

        if not replica_means:
            print("No valid data found. Skipping.")
            continue
            
        global_mean = np.mean(replica_means)
        representative_idx = np.argmin(np.abs(np.array(replica_means) - global_mean))
        chosen_file = valid_files[representative_idx]
        
        print(f"Global mean c_max: {global_mean:.2f} | Chosen replica: {os.path.basename(chosen_file)}")
        
        # generate the video 
        raw_data = np.fromfile(chosen_file, dtype=np.float32)
        
        # apply the same shielding to the chosen file
        approx_floats = len(raw_data) / expected_frames
        n_passive = int(round((approx_floats - (3 * n_active)) / 2))
        floats_per_frame = int((2 * n_passive) + (3 * n_active))
        num_frames = len(raw_data) // floats_per_frame
        clean_data = raw_data[:num_frames * floats_per_frame]
        
        cutoff_index = 2 * n_passive
        data_2d = clean_data.reshape(num_frames, floats_per_frame)
        all_passive_particles = data_2d[:, :cutoff_index].reshape(num_frames, n_passive, 2)
        all_active_particles = data_2d[:, cutoff_index:].reshape(num_frames, n_active, 3)
        
        frames_to_render = np.arange(0, num_frames, frame_step)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_aspect('equal')
        
        passive_patches = [patches.Circle((0,0), radius=current_radius, fc='lightgray', ec='none', alpha=0.6) for _ in range(n_passive)]
        for p in passive_patches: 
            ax.add_patch(p)
            
        active_patches = [patches.Circle((0,0), radius=current_radius, fc='red', ec='darkred', lw=1.5, zorder=5) for _ in range(n_active)]
        for p in active_patches: 
            ax.add_patch(p)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # fully translated title and labels in LaTeX formatting
        ax.set_title(rf'Radius = {factor}$R_0$ | $\rho_p$ = {density*100:.0f}% | File: {os.path.basename(chosen_file)}', fontsize=11)
        
        def animate(frame_idx):
            real_idx = frames_to_render[frame_idx]
            
            for i, (x, y) in enumerate(all_passive_particles[real_idx]):
                passive_patches[i].center = (x, y)
                
            for i, (x, y) in enumerate(all_active_particles[real_idx, :, 0:2]):
                active_patches[i].center = (x, y)
            
            # updated time text inside the video
            time_text.set_text(f'Time: {real_idx} s')
            
            return passive_patches + active_patches + [time_text]

        ani = animation.FuncAnimation(fig, animate, frames=len(frames_to_render), interval=1000/video_fps, blit=True)
        
        try:
            print("Rendering MP4...")
            ani.save(video_path, writer='ffmpeg', fps=video_fps, dpi=120)
        except:
            print("Saving as GIF (FFmpeg not detected)...")
            ani.save(gif_path, writer='pillow', fps=video_fps, dpi=100)
            
        plt.close(fig)
        print("Completed!")

print("\nThe entire massive rendering process has finished!")