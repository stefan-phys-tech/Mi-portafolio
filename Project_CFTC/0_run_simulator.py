import subprocess
import os
from concurrent.futures import ProcessPoolExecutor

executable_name = "./cpp_sim_exec" 
data_directory = "/mnt/c/Users/stefa/OneDrive/Desktop/CFTC/Negative_torque_56/" 

# Ensure the target directory exists
os.makedirs(data_directory, exist_ok=True) 

# Fixed parameters
particle_radius_0 = 2.385e-6    #given in meters
density_active_particles = 0.011  #as a proportion of 1
torque_strength = -56    #in units of kBT (converted in script)
active_velocity = 1.9e-6  #given in meters per second
time_to_simulate = 3000  #given in seconds

# Resolution and boundary settings
delta_t = 0.004  #the timestep of the simulation in seconds
record_intervals = 1 #how often you want positions to be recorded in seconds
boundary_size = 120 #the length of the side of the simulation box (in particle radii)

# Scaled physics
alpha_op = 1.8 # Lennard-Jones scaling energy parametery
E_LJ_base = 150.0 #in units of kBT 
radius_factors = [0.4, 1.0, 1.6]
passive_densities = [round(d * 0.01, 2) for d in range(51)]

sims_per_condition = 15
max_workers = 5 

def run_simulation(args):
    """Helper function to execute a single simulation process."""
    filename, sim_arguments = args
    print(f"-> Iniciando: {filename}")
    
    # Execute the C++ binary
    subprocess.run([executable_name] + sim_arguments, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    
    print(f"-> ¡Completado!: {filename}")
    return filename


def prepare_tasks():
    """Generates and formats all the arguments required for the pending simulations."""
    for passive_density in passive_densities:
        tasks = []
        for factor in radius_factors:
            current_radius = factor * particle_radius_0
            
            # Apply the verified phenomenological collapse
            modified_lj = E_LJ_base * (factor ** alpha_op)
            
            for i in range(1, sims_per_condition + 1):
                # Filename now tracks passive density, radius factor, and run number
                filename = f"sim_pas_{passive_density:.2f}_R_{factor:.1f}R0_run_{i}.bin"
                full_path = os.path.join(data_directory, filename)
                
                argument_list = [
                    current_radius, 
                    density_active_particles, 
                    passive_density, 
                    modified_lj, 
                    torque_strength, 
                    active_velocity, 
                    time_to_simulate, 
                    delta_t, 
                    record_intervals, 
                    boundary_size, 
                    full_path
                ]
                str_arguments = [str(arg) for arg in argument_list]
                tareas.append((filename, str_arguments))
    return tasks


if __name__ == '__main__':
    pending_tasks = prepare_tasks()
    total_sims = len(pending_tasks)

    print("=" * 60)
    print(" STARTING GAP-FILLING SIMULATION CAMPAIGN (1% Resolution) ")
    print(f" Total remaining simulations to compute: {total_sims}")
    print(f" Safety limit: {MAX_WORKERS} concurrent cores")
    print("=" * 60 + "\n")

    # Parallel execution ensuring we never exceed the MAX_WORKERS limit
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # map() evaluates and distributes all tasks efficiently across the CPU cores
        list(executor.map(run_simulation, pending_tasks)) 

    print("\nSimulation campaign completed successfully!")