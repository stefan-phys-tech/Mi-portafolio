#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

class Simulator{

private:

    struct Constants {
        //constants
        static constexpr double PI = 3.14159265;
        static constexpr int T = 303; //K
        static constexpr double visc = 0.0021; //N s m^-2
        static constexpr double kB = 1.38e-23; //J K-1
        static constexpr double scaling = 0.25; //0.25

        //number of particles
        static int num_particles(double density, double radius, double box_prop) {
            return static_cast<int>(density/((PI*radius*radius)/(pow(box_prop*radius,2))));
        }
        //cutoff calculations
        static double cutoff(double radius) {
            return 10*radius;
        }
        static double tor_cutoff(double radius) {
            return 8*radius;
        }
        static double cutoff_passive(double radius) {
            return 2*radius;
        }
        static double offset(double radius) {
            return 2*radius;
        }
        //force multiplier
        static double force_mult(double radius) {
            return 0.5625*1.0/(6*PI*radius*visc);
        }
        static double torque_mult(double radius, double delta_t) {
            return 0.875*(4*radius*radius)*delta_t/(8*PI*pow(radius,3)*visc);
        }
        //diffusion
        static double sqr_trans_diffusion(double radius, double delta_t) {
            return sqrt(2*delta_t*(0.5625*(kB*T)/(6*PI*radius*visc)));
        }
        static double sqr_rot_diffusion(double radius, double delta_t) {
            return sqrt(2*delta_t*(0.875*(kB*T)/(8*PI*pow(radius,3)*visc)));
        }
        //sigma
        static double sigma(double radius) {
            return (2*radius)/std::pow(2.0, 1.0/6.0);
        }
        //well depth
        static double LJ_depth(double lj_dep) {
            return lj_dep*kB*T;
        }
        static double tor_depth(double tor_dep) {
            return tor_dep*kB*T;
        }
        //rebuild_counter
        static int rebuild_counter(double radius, double act_vel, double delta_t) {
            return static_cast<int>((0.5*radius)/(act_vel*delta_t));
        }
        //calculating power
        static double power(double value, int pows) {
            double res = 1.0;
            while(pows > 0){
                res *= value;
                --pows;
            }
            return res;
        }
    };

    //various simulation variables
    const int num_active_particles_;
    const int num_passive_particles_;
    double radius_;
    const double act_part_vel_;
    const double sqr_trans_diffusion_;
    const double sqr_rot_diffusion_;
    const double size_box_;
    const double inv_size_box_;
    const double sigma_;
    const double sigma_6_;
    const double sigma_12_;
    const double lj_depth_;
    const double tor_depth_;
    const double cutoff_;
    const double tor_cutoff_;
    const double cutoff_passive_;
    const double offset_;
    const double delta_t_;
    int global_step_count_ = 0;
    const int construct_neigh_list_counter_;
    const double force_multiplier_;
    const double torque_multiplier_;
    std::random_device rd_trans_;
    std::mt19937 generator_trans_;
    std::normal_distribution<double> distribution_trans_;
    std::random_device rd_angle_;
    std::mt19937 generator_angle_;
    std::normal_distribution<double> distribution_angle_;

    //arrays for storing positions and various variables
    std::vector<double> positions_x_ = std::vector<double>((num_passive_particles_+num_active_particles_));
    std::vector<double> positions_y_ = std::vector<double>((num_passive_particles_+num_active_particles_));
    std::vector<double> forces_x_ = std::vector<double>((num_passive_particles_+num_active_particles_));
    std::vector<double> forces_y_ = std::vector<double>((num_passive_particles_+num_active_particles_));
    std::vector<double> torques_ = std::vector<double>(num_active_particles_);
    std::vector<double> orientations_ = std::vector<double>(num_active_particles_);
    std::vector<int> neighbours_lj_;
    std::vector<int> neighbours_torque_;
    std::vector<int> place_holder_lj_ = std::vector<int>((num_passive_particles_+num_active_particles_));
    std::vector<int> place_holder_torque_ = std::vector<int>((num_active_particles_));

    //private-helper constructor, here is where all of the other parameters get calculated and set for the simulation
    Simulator(double radius, int num_act_parts, int num_pass_parts, double act_vel, double sqr_trans, double sqr_rot, double size_box, double sig, double lj, double tor, double cut, double t_cut, double pass_cut, double off, int construct_neigh_list_counter, double force_multiplier, double sigma_12, double sigma_6, double torque_multiplier, double delta_t)
    :   radius_(radius),
        num_active_particles_(num_act_parts),
        num_passive_particles_(num_pass_parts),
        act_part_vel_(act_vel),
        sqr_trans_diffusion_(sqr_trans),
        sqr_rot_diffusion_(sqr_rot),
        size_box_(size_box),
        sigma_(sig),
        lj_depth_(lj),
        tor_depth_(tor),
        cutoff_(cut),
        tor_cutoff_(t_cut),
        cutoff_passive_(pass_cut),
        offset_(off),
        inv_size_box_(1/size_box),
        construct_neigh_list_counter_(construct_neigh_list_counter),
        force_multiplier_(force_multiplier),
        generator_trans_(rd_trans_()),
        distribution_trans_(0.0,1.0),
        generator_angle_(rd_angle_()),
        distribution_angle_(0.0,1.0),
        sigma_12_(sigma_12),
        sigma_6_(sigma_6),
        torque_multiplier_(torque_multiplier),
        delta_t_(delta_t){}

    //positions are relaxed by first shrinking the particles, then slowly growing them to their original size
    bool relax_by_growth(double target_radius, double start_fraction = 0.7, int growth_steps = 20) {
        double original_radius = radius_;
        radius_ = target_radius * start_fraction;

        for (int step = 1; step <= growth_steps; ++step) {
            radius_ = target_radius * (start_fraction + (1.0 - start_fraction) * step / growth_steps);

            bool ok = false;
            for (int attempt = 0; attempt < 50 && !ok; ++attempt) {
                ok = relax_positions_iterative(2000);
            }
            if (!ok) {
                return false;
            }
        }
        radius_ = target_radius;
        return true;
    }
    //here we relax positions by displacing particles slowly along their centres of overlap
    bool relax_positions_iterative(int max_iterations = 2000) {
        bool existing_overlaps;

        double dx = 0.0;
        double dy = 0.0;
        double new_x = 0.0;
        double new_y = 0.0;
        double min_dist = (1.99*radius_);
        double slop = 0.01*min_dist;
        int total_overlaps = 0;

        for (int iter = 0; iter < max_iterations; iter ++) {
            existing_overlaps = false;
            for (int part_1 = 0; part_1 < (num_active_particles_+num_passive_particles_-1); part_1++) {
                for (int part_2 = part_1+1; part_2 < (num_active_particles_+num_passive_particles_); part_2++) {
                    dx = positions_x_[part_2]-positions_x_[part_1];
                    dy = positions_y_[part_2]-positions_y_[part_1];
                    double dist = sqrt(dx*dx+dy*dy);
                    //particles overlap
                    if (dist < (min_dist-slop)) {
                        double overlap = min_dist - dist;
                        existing_overlaps = true;

                        double ux = dx / dist;
                        double uy = dy / dist;
                        //displacement based
                        if ((part_1 < num_passive_particles_) || true) {
                            total_overlaps++;
                            positions_x_[part_1] -= 0.5 * overlap * ux;
                            positions_y_[part_1] -= 0.5 * overlap * uy;
                        }
                        if ((part_2 < num_passive_particles_) || true) {
                            positions_x_[part_2] += 0.5 * overlap * ux;
                            positions_y_[part_2] += 0.5 * overlap * uy;
                        }
                        //adjusting for boundaries
                        positions_x_[part_1] = boundary_condition(positions_x_[part_1]);
                        positions_y_[part_1] = boundary_condition(positions_y_[part_1]);
                        positions_x_[part_2] = boundary_condition(positions_x_[part_2]);
                        positions_y_[part_2] = boundary_condition(positions_y_[part_2]);
                    }
                }
            }
        }
        return !existing_overlaps;
    }
    //constructs the neighbour list
    void build_neighbour_lists() {
        int count_position_lj_list = 0;
        int count_position_torque_list = 0;
        neighbours_lj_.clear();
        neighbours_torque_.clear();
        double cut_off_lj = 0.0;
        double cut_off_torque = ((tor_cutoff_+offset_)*(tor_cutoff_+offset_));
        double dx = 0.0;
        double dy = 0.0;
        //LJ neighbour list
        for (int part_1 = 0; part_1 < (num_active_particles_+num_passive_particles_); part_1 ++) {
            place_holder_lj_[part_1] = count_position_lj_list;
            if (part_1 < num_passive_particles_) {
                cut_off_lj = ((cutoff_passive_+offset_)*(cutoff_passive_+offset_));
            } else {
                cut_off_lj = ((cutoff_+offset_)*(cutoff_+offset_));
            }
            for (int part_2 = part_1+1; part_2 < (num_active_particles_+num_passive_particles_); part_2 ++) {
                dx = image_distance(positions_x_[part_2]-positions_x_[part_1]);
                dy = image_distance(positions_y_[part_2]-positions_y_[part_1]);
                //lj interaction happens between all particles
                if ((dx*dx+dy*dy) < cut_off_lj) {
                    //for lj interaction
                    count_position_lj_list++;
                    neighbours_lj_.push_back(part_2);
                }
            }
        }
        //creating neighbour list for torques
        for (int part_1 = num_passive_particles_; part_1 < (num_passive_particles_+num_active_particles_); part_1++) {
            place_holder_torque_[part_1-num_passive_particles_] = count_position_torque_list;
            for (int part_2 = 0; part_2 < num_passive_particles_; part_2++) {
                dx = image_distance(positions_x_[part_2]-positions_x_[part_1]);
                dy = image_distance(positions_y_[part_2]-positions_y_[part_1]);
                //lj interaction happens between all particles
                if ((dx*dx+dy*dy) < cut_off_torque) {
                    //for lj interaction
                    count_position_torque_list++;
                    neighbours_torque_.push_back(part_2);
                }
            }
        }
    }
    //computes the lennard jones forces, we scan all particles and apply the interaction
    void calculate_forces() {
        std::fill(forces_x_.begin(), forces_x_.end(),0.0);
        std::fill(forces_y_.begin(), forces_y_.end(),0.0);
        //loop through all the particles
        for (int part_1 = 0; part_1 < (num_passive_particles_+num_active_particles_-1); part_1++) {
            double pos_1_x = positions_x_[part_1];
            double pos_1_y = positions_y_[part_1];

            double relevant_cutoff = 0.0;

            if (part_1 < num_passive_particles_) {
                relevant_cutoff = cutoff_passive_*cutoff_passive_;
            } else {
                relevant_cutoff = cutoff_*cutoff_;
            }

            for (int neighbour = place_holder_lj_[part_1]; neighbour < place_holder_lj_[part_1+1]; neighbour++) {
                double dx = pos_1_x-positions_x_[neighbours_lj_[neighbour]];
                double dy = pos_1_y-positions_y_[neighbours_lj_[neighbour]];

                dx = image_distance(dx);
                dy = image_distance(dy);

                double sq_distance = dx*dx+dy*dy;
                if (sq_distance < relevant_cutoff) {
                    double distance = sqrt(sq_distance);
                    double dx_project = dx/distance;
                    double dy_project = dy/distance;
                    
                    //if particles get too close, we have a hard-core repulsion failsafe, to prevent the simulation from breaking
                    if (distance < 1.5*radius_){
                        double overlap = (2*radius_) - distance;
                        double adjustment_x = overlap*0.5*dx_project;
                        double adjustment_y = overlap*0.5*dy_project;

                        forces_x_[part_1] += adjustment_x/(delta_t_*force_multiplier_);
                        forces_y_[part_1] += adjustment_y/(delta_t_*force_multiplier_);
                        forces_x_[neighbours_lj_[neighbour]] -= adjustment_x/(delta_t_*force_multiplier_);
                        forces_y_[neighbours_lj_[neighbour]] -= adjustment_y/(delta_t_*force_multiplier_);
                    } else {
                        //standard force computation
                        double force_magnitude = lennard_jones(distance);
                        forces_x_[part_1] += force_magnitude*dx_project;
                        forces_y_[part_1] += force_magnitude*dy_project;
                        forces_x_[neighbours_lj_[neighbour]] -= force_magnitude*dx_project;
                        forces_y_[neighbours_lj_[neighbour]] -= force_magnitude*dy_project;
                    }
                }
            }
        }
    }
    //we iterate through the active particles and check for the passive neighbours and calculate the torques accordingly
    void calculate_torques() {
        std::fill(torques_.begin(), torques_.end(),0.0);

        for (int part_1 = num_passive_particles_; part_1 < num_active_particles_+num_passive_particles_; part_1++) {
            double pos_1_x = positions_x_[part_1];
            double pos_1_y = positions_y_[part_1];
            double unit_x = cos(orientations_[part_1-num_passive_particles_]);
            double unit_y = sin(orientations_[part_1-num_passive_particles_]);

            double relevant_cutoff = tor_cutoff_*tor_cutoff_;

            int end_index = 0;
            if (part_1 < num_active_particles_+num_passive_particles_-1) {
                end_index = place_holder_torque_[part_1-num_passive_particles_+1];
            } else {
                end_index = neighbours_torque_.size();
            }

            for (int neighbour = place_holder_torque_[part_1-num_passive_particles_]; neighbour < end_index; neighbour++) {

                double dx = pos_1_x-positions_x_[neighbours_torque_[neighbour]];
                double dy = pos_1_y-positions_y_[neighbours_torque_[neighbour]];

                dx = image_distance(dx);
                dy = image_distance(dy);

                double sq_distance = dx*dx+dy*dy;
                if (sq_distance < relevant_cutoff) {
                    double distance = sqrt(sq_distance);
                    torques_[part_1-num_passive_particles_] += torque(distance, dx, dy, unit_x, unit_y);
                }
            }
        }
    }
    //calculating lennard jones interactions
    double lennard_jones(double distance) {
        double first = (2*sigma_12_) / Constants::power(distance, 13);
        double second = (sigma_6_) / Constants::power(distance,7);
        return 24*lj_depth_*(first-second);
    }
    //calculating the torque interaction
    double torque(double total_distance, double dist_x, double dist_y, double unit_x, double unit_y) {
        double exponent = -(exp(-(0.25/(2*radius_))*total_distance)/(total_distance*total_distance))*((0.25/(2*radius_))+(1/total_distance));
        double x_pot = dist_x*exponent;
        double y_pot = dist_y*exponent;
        return (unit_x*y_pot)-(unit_y*x_pot);
    }

    //integrates positions in timestep, we calculate forces and torques first
    void integrate_positions() {
        //calculate forces
        calculate_forces();
        //calculate torques
        calculate_torques();

        //adjust positions
        for (int part = 0; part < num_active_particles_+num_passive_particles_; part ++) {
            //we apply forces to the particles and noise to change their positions
            double new_x = positions_x_[part] + forces_x_[part]*delta_t_*force_multiplier_ + sqr_trans_diffusion_*distribution_trans_(generator_trans_);
            double new_y = positions_y_[part] + forces_y_[part]*delta_t_*force_multiplier_ + sqr_trans_diffusion_*distribution_trans_(generator_trans_);

            //if the particles are active partilces
            if (part >= num_passive_particles_) {
                //change their orientation according to the following terms
                orientations_[part-num_passive_particles_] += sqr_rot_diffusion_*distribution_angle_(generator_angle_); //noise term
                orientations_[part-num_passive_particles_] += torques_[part-num_passive_particles_]*torque_multiplier_*tor_depth_; //torque term
                //propel the particles along their orientation by the velocity over the timestep
                new_x += act_part_vel_*delta_t_*cos(orientations_[part-num_passive_particles_]);
                new_y += act_part_vel_*delta_t_*sin(orientations_[part-num_passive_particles_]);
            }
            //check if any particles left the box
            new_x = boundary_condition(new_x);
            new_y = boundary_condition(new_y);
            positions_x_[part] = new_x;
            positions_y_[part] = new_y;
        }
    }
    //boundary condition - more efficient than the one below
    double image_distance(double d_1d) {
        return d_1d -= size_box_*round(d_1d*inv_size_box_);
    }
    //checking if the particles are outside of the simulation to translate them back
    double boundary_condition(double pos_1d) {
        if (pos_1d < 0) pos_1d += size_box_;
        else if (pos_1d >= size_box_) pos_1d -= size_box_;
        return pos_1d;
    }

public:
    //this is the public facing constructor, the one you directly call from main(), it calls the private constructor above
    Simulator (double radius, double active_dens, double passive_dens, double lj_strength, double tor_strength, double v0_star, double boundary_length, double delta_t)
    :   Simulator(radius, //radius
                Constants::num_particles(active_dens, radius, boundary_length), //number of active particles
                Constants::num_particles(passive_dens, radius, boundary_length), //number of passive particles
                v0_star, //active particle velocity
                Constants::sqr_trans_diffusion(radius, delta_t), // translational diffusion parameter
                Constants::sqr_rot_diffusion(radius, delta_t), //rotational diffusion parameter
                boundary_length*radius, //size of the simulation
                Constants::sigma(radius), //distance at which the LJ strength is at a minimum
                Constants::LJ_depth(lj_strength), //strength of LJ in kBT
                Constants::tor_depth(tor_strength), //strength of torque in kBT
                Constants::cutoff(radius), //cuttoff for the LJ interaction
                Constants::tor_cutoff(radius), //cutoff for the torque interaction
                Constants::cutoff_passive(radius), //cutoff for the passive-passive interaction
                Constants::offset(radius), //offset for all interactions
                Constants::rebuild_counter(radius, v0_star, delta_t), //how often do we need to rebuild the neighbour lists
                Constants::force_mult(radius), //force multiplier - just helps us not to have to recompute it
                Constants::power(Constants::sigma(radius), 12), //helps us not have to recompute the power terms
                Constants::power(Constants::sigma(radius), 6), //helps us not have to recompute the power terms
                Constants::torque_mult(radius, delta_t), //torque multiplier - just helps us not to have to recompute it
                delta_t //the timestep
                ) {}
    
    //initializing particle positions, since this results in overlaps, we need to also relax the positions
    void initialize_positions(double density) {
        std::random_device rand_place_device;
        std::mt19937 generator_position(rand_place_device());
        std::uniform_real_distribution<double> dis(0, size_box_);
        //positions for passive particles
        for (int part = 0; part < (num_passive_particles_); part ++) {
            double x_pos = dis(generator_position);
            double y_pos = dis(generator_position);
            positions_x_[part] = x_pos;
            positions_y_[part] = y_pos;
        }
        
        for (int part = num_passive_particles_; part < num_passive_particles_ + num_active_particles_; part++) {
            double x_pos = dis(generator_position);
            double y_pos = dis(generator_position);
            
            positions_x_[part] = x_pos;
            positions_y_[part] = y_pos;
        }
            
        bool positions_relaxed = false;
        int temp_counter = 0;
        while (!positions_relaxed) {
            if (density > 0.55) {
                std::cout << "This will take a few minutes ... dense system." << std::endl;
                positions_relaxed = relax_by_growth(radius_, 0.7, 30);
            } else {
                positions_relaxed = relax_by_growth(radius_, 0.7, 10);
            }
            temp_counter++;
            if (temp_counter > 10) {
                std::cout << "Hmm, for some reason the positions could not be relaxed. Is the system too dense?" << std::endl;
            }
        }
    }
    //initializing particle orientations in (-pi, pi]
    void initialize_orientations() {
        std::random_device rand_dev_orient_;
        std::mt19937 generator_orient_(rand_dev_orient_());
        std::uniform_real_distribution<double> dis_orient_(0.0,1.0);
        for (int part = 0; part < num_active_particles_; part++) {
            orientations_[part] = (2*Constants::PI*dis_orient_(generator_orient_))-Constants::PI;
        }
    }
    //positions are saved to the specified filename in txt (produces very large files)
    void save_positions(std::string filename, bool is_checkpoint = false, bool record_passive = true) {

        std::ofstream save_file;
        if (is_checkpoint) {
            save_file.open(filename, std::ofstream::trunc);
        } else {
            save_file.open(filename, std::ios_base::app);
        }
        for (int part = 0; part < (num_active_particles_+num_passive_particles_); part ++) {

            if ((!record_passive) && (part < num_passive_particles_)) {
                continue;
            }
            
            save_file << positions_x_[part] << " " << positions_y_[part] << " ";
            
            if (part >= num_passive_particles_) {
                //continue;
                save_file << (ceil(remainder(orientations_[part-num_passive_particles_], 2*Constants::PI)*100.0))/100.0;//orientations_[part-num_passive_particles_]; //<< " " << run_lengths_[part-num_passive_particles_] << " " << alphas_[part-num_passive_particles_];
            }
            if (!is_checkpoint) {
                save_file << ";";
            } else {
                save_file << " ";
            }
        }
        save_file.close();
    }

    //positions are saved to the specified filename in binary
    void save_positions_bin(std::string filename) {

        std::vector<float> buffer;
        buffer.reserve((2*num_passive_particles_)+(3*num_active_particles_));
        //we put the passive positions into the buffer
        for (int part = 0; part < num_passive_particles_; part++){
            buffer.push_back((float)positions_x_[part]);
            buffer.push_back((float)positions_y_[part]);
        }
        //we put the active positions and orientations into the buffer
        for (int part = num_passive_particles_; part < num_passive_particles_+num_active_particles_; part++){
            buffer.push_back((float)positions_x_[part]);
            buffer.push_back((float)positions_y_[part]);
            buffer.push_back((float)orientations_[part-num_passive_particles_]);
        }

        std::ofstream out(filename, std::ios::binary | std::ios::app);
        out.write(reinterpret_cast<const char*>(buffer.data()),buffer.size() * sizeof(float));
    }

    //running simulation in time mode
    void run_time_simulation(int num_steps){
        for (int step = 0; step < num_steps; step++) {
            if (global_step_count_ % construct_neigh_list_counter_ == 0) {
                build_neighbour_lists();
            }
            integrate_positions();
            global_step_count_++;
        }
    }

};


int main(int argc, char *argv[]) {
    
    //The arguments arrive in order as strings, so we need to cast them back to the correct types. (0th argument is the script name.)
    double particle_radius = std::atof(argv[1]);
    double density_active_particles = std::atof(argv[2]);
    double density_passive_particles = std::atof(argv[3]);
    double lj_strength = std::atof(argv[4]);
    double torque_strength = std::atof(argv[5]);
    double active_velocity = std::atof(argv[6]);
    int time_to_simulate = std::atoi(argv[7]);
    double delta_t = std::atof(argv[8]);
    double record_intervals = std::atof(argv[9]);
    int boundary_size = std::atoi(argv[10]);
    std::string filename = argv[11];

    int steps_to_run = (int) time_to_simulate/delta_t; //the simulation time is changed to number of steps
    int recording_increment = (int) (record_intervals/delta_t); //we record positions to the file every recording_increment steps
    
    std::cout << "Setting up the simulation." << std::endl;
    Simulator new_simulation(particle_radius, density_active_particles, density_passive_particles, lj_strength, torque_strength, active_velocity, boundary_size, delta_t);
    std::cout << "Initializing positions." << std::endl;
    new_simulation.initialize_positions(density_passive_particles);
    std::cout << "Initializing orientations." << std::endl;
    new_simulation.initialize_orientations();
    int steps_taken = 0;
    while (steps_taken < steps_to_run) {
        std::cout << "\r" << "Time: " << steps_taken*delta_t << " out of " << time_to_simulate << std::flush;
        new_simulation.run_time_simulation(recording_increment);
        new_simulation.save_positions_bin(filename);
        steps_taken += recording_increment;
    }
    return 0;
}
