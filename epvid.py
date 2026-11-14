import os
os.environ["MIEPOM_VECTORIZED_MIE"] = "1"

import miepom.core as mp
import matplotlib.pyplot as plt
import numpy as np
import numpy as xp

np.random.seed(0)
xp.random.seed(0)

USEGPU = False

try:
    import mlx.core as xp
    mp.set_backend(xp)
    USEGPU = True
except ImportError:
    pass
    
try:
    import cupy as xp
    mp.set_backend(xp)
    USEGPU = True
except ImportError:
    pass

if USEGPU == False:
    print("GPU backend not available, using CPU backend. For better performance, install mlx or cupy.")

# Number frames to generate plots and for testing
test_count = 4000

# Total number of time steps
count = 2**16

# shape of the imaging sensor and the shape of the computational grid
sensor_shape = (512, 512)

# create dust images for the sensor transmittance function to simulate the effect of dust on the sensor
dust1 = mp.create_dust_image(sensor_shape)
dust2 = mp.create_dust_image(sensor_shape)
dust = dust1 * dust2

# sensor settings define the properties of the imaging sensor
sensor_settings = {"shape": sensor_shape ,
                   # effective pixel size = (pixel_height*pixel_width)**0.5 of a pixel in meters
                   "pixel_size": 6.9e-6,  
                   "adcbit": 10,
                   # aspect ratio of the pixels (pixel_width/pixel_height)
                   "pixel_aspect" : 1.0, 
                   "full_well_capacity": 2**15,  
                   "transmittance": dust,
                   "flags": 0  # Sensor flags, can be used to disable certain features of the sensor
                   }

# light source settings define the properties of the illumination used in the imaging system
light_source_settings = {# center wavelength of the light source
                        "wavelength": 532e-9,
                        # bandwidth of the light source - assuming Gaussian spectral profile
                       "bandwidth": 29e-9,
                       # intensity at the object plane
                       # because we are using 60x magnification, the intensity at the object plane is 60x60 times higher 
                       # than the intensity at the image plane, so we set it to 60*60 to have unit intensity at the image plane
                       "intensity": 60*60}

microscope_settings = {"magnification": 60, 
                       # tube lens focal length in meters, combined with the magnification, determines the effective focal length of objective lens
                       "tube_lens_focal_length": 200e-3, 
                       # The refractive index of the immersion medium (e.g., water, oil, etc.)
                       # this is for future use, currently not used in the simulation
                       # currently, the simmulation assumes that the medium in which the particles are suspended has the same refractive index as the immersion medium of the objective lens
                       # regardless of what the user specifies here, the simulation will always use the refractive index of the immersion medium of the objective lens
                       # In the future, we will allow the user to specify the refractive index of the immersion medium to be different from the refractive index of the medium in which the particles are suspended, 
                       "immersion_medium_refractive_index": 1.33,
                       "objective_numerical_aperture": 0.9,
                       "condenser_numerical_aperture": 0.1}

# particle settings define the parameters needed for Mie scattering calculations
particle_settings = {"num_particles": 4000,
                     "diameter_mean": 0.3e-6,
                     # 0.5% relative standard deviation for the particle diameter
                     "diameter_std": 0.3e-6 * 0.005,
                     "mobility_mean": -4e-8,
                     # 2% relative standard deviation
                     "mobility_std": 4e-8 * 0.02,
                     # Refractive index of the particle (can be a complex number)
                     "n_particle": 1.6 + 0j,
                     # 0.2% relative standard deviation for the particle refractive index
                     "n_particle_std": 1.6 * 0.002,
                     "n_medium": 1.33,
                     # whether to use the Rayleigh approximation for small particles
                     "use_rayleigh": False
                     }

channel_height = 400e-6 # Height of the microfluidic channel in meters
viscosity = 1e-3 # Pa.s (dynamic viscosity of the fluid)

# central velocity of the flow in the microfluidic channel. This is used to calculate the pressure gradient based on the channel height and fluid viscosity. 
central_velocity = 2e-5

# in the simulator, we set the pressure gradient which induces flow in the channel
pressure_gradient = central_velocity * 8 * viscosity / channel_height**2

# sample settings define the properties of the sample and the microfluidic channel
sample_settings = { 
                    "particle_density": 1000,  # Density of the particles in kg/m^3
                    "fluid_density": 1000,  # Density of the fluid in kg/m^3
                    "temperature": 298,  # Temperature in Kelvin
                    "viscosity": viscosity,  # Viscosity of the fluid in Pa.s
                    "channel_height" : channel_height,  # Height of the microfluidic channel
                    "permittivity": 80,  # Relative permittivity of the fluid
                    "wall_zeta": 80e-3,
                    "pressure_gradient": pressure_gradient
                    }  # Zeta potential of the channel walls

experiment_settings = {
                    "ac_period": 4.8e-3,
                    "ac_voltage": 75,
                    "ac_phase": 0,
                    
}

microscope_simulator_settings = {"simulator_numerical_aperture": 1.0, # the numerical aperture of the simulator, this should be equal to or greater than the numerical aperture of the objective lens
                      "focal_plane": 0e-6, 
                      "simulator_dof": 100e-6,  # Depth of field of the simulator
                      # Whether the particles are considered monodisperse or not - much faster to compute if monodisperse
                      "monodisperse_particles": False}


motion_simulator_settings = {
    "time_step": 1e-4,  # Time step for the simulation in seconds
    "flags" : 0 # Simulation flags, can be used to disable certain features of the motion simulator
}

trigger_settings = {
    "random_sampling_interval": 32, # Random sampling interval in units of simulator time steps
    "regular_sampling_interval": 48,  # Regular sampling interval in units of simulator time steps
}

def create_particles(seed = None):
    # Create Mie particles based on the particle settings and a random seed
    if seed is not None:    
        np.random.seed(seed)
    diameter_mean = particle_settings["diameter_mean"]
    num_particles = particle_settings["num_particles"]
    diameter_std = particle_settings["diameter_std"]
    n_mean = np.real(particle_settings["n_particle"])
    n_complex = np.imag(particle_settings["n_particle"])
    n_std = particle_settings["n_particle_std"]
    mobility_mean = particle_settings["mobility_mean"]
    mobility_std = particle_settings["mobility_std"]
    n_medium = particle_settings["n_medium"]
    use_rayleigh = particle_settings["use_rayleigh"]

    diameter = np.random.normal(loc = diameter_mean, scale = diameter_std, size = num_particles)
    n = np.random.normal(loc = n_mean, scale = n_std, size = num_particles) + 1j * n_complex
    mobility = np.random.normal(loc = mobility_mean, scale = mobility_std, size = num_particles)  

    if use_rayleigh:
        particles = mp.create_rayleigh_particles(count = num_particles, diameter = diameter, n_particle = n, n_medium = n_medium)
    else:
        particles = mp.create_mie_particles(count = num_particles, diameter = diameter, n_particle = n, n_medium = n_medium)
    particles.mobility = mobility
    return particles

def random_sampling_mask(n = count, interval = trigger_settings["random_sampling_interval"], seed = None):
    """Returns a boolean mask array of data sampling indices"""
    out = np.zeros((n,), dtype = np.bool_)
    if seed is not None:
        np.random.seed(seed)
    r = np.random.randint(low = 0, high = interval//2 ,size = n//interval)
    for i,j in enumerate(r):
        out[i*interval + j] = 1
    return out

def random_sampling_mask_full(n = count, interval = trigger_settings["random_sampling_interval"], seed = None):
    """Returns a boolean mask array of data sampling indices"""
    out = np.zeros((n,), dtype = np.bool_)
    if seed is not None:
        np.random.seed(seed)
    r = np.random.randint(low = 0, high = interval//2 ,size = n//interval)
    for i,j in enumerate(r):
        for j in range(0,interval//2):
            out[i*interval + j] = 1
    return out

def regular_sampling_mask(n = count, interval = trigger_settings["regular_sampling_interval"], start = 0):
    """Returns a boolean mask array of data sampling indices"""
    out = np.zeros((n,), dtype = np.bool_)
    out[start::interval] = 1
    return out

# first, create the sensor, light source, and microscope objects
sensor = mp.create_sensor(**sensor_settings)
light_source = mp.create_light_source(**light_source_settings)
microscope = mp.create_microscope(light_source=light_source, sensor=sensor, **microscope_settings)

# create the Mie particles and the microscope simulator
particles = create_particles(seed = 0)
microscope_sim = mp.create_simulator(microscope, particles, **microscope_simulator_settings)

# object plane is the plane where the particles are located, it is defined by the microscope settings
# this object contains information about the size of the object plane, which is determined by the microscope settings and the sensor settings
# it can be used to obtain the dimensions of the object plane, which can be used to generate random coordinates for the particles
# it can also be used to show images 
object_plane = microscope.get_object_plane()
# set the units for displaying the object plane images
object_plane.set_imshow_units("um")
# reciprocal plane contains information about the coordinates of the Fourier transform data
reciprocal_plane = microscope.get_reciprocal_plane()

def create_initial_coordinates(seed = None):
    num_particles = particle_settings["num_particles"]
    height = sample_settings["channel_height"]
    channel_width = object_plane.height
    channel_length = object_plane.width
    np.random.seed(seed)
    coordinates = np.zeros((num_particles, 3))
    coordinates[:, 0] = np.random.uniform(low = -channel_length/2, high = channel_length/2, size = (num_particles))
    coordinates[:, 1] = np.random.uniform(low = -channel_width/2, high = channel_width/2, size = (num_particles))
    coordinates[:, 2] = np.random.uniform(low = -height/2, high = height/2, size = (num_particles))
    return coordinates

def create_triggered_video_masks(count, seed = None):
    tmask_random = random_sampling_mask(n = count, interval = trigger_settings["random_sampling_interval"], seed = seed)
    tmask_regular = regular_sampling_mask(n = count, interval = trigger_settings["regular_sampling_interval"])
    return tmask_random, tmask_regular

initial_coordinates = create_initial_coordinates(0)

# set the z-coordinates to specific values for visualization purposes.
initial_coordinates[0,2] = 20e-6 - sample_settings["channel_height"]/2
initial_coordinates[1,2] = 80e-6 - sample_settings["channel_height"]/2
initial_coordinates[2,2] = 200e-6 - sample_settings["channel_height"]/2

import epsim as ep_motion
import numpy as np

particles.set_coordinates(initial_coordinates)
merged_motion_simulator_settings = {**sample_settings, **experiment_settings, **motion_simulator_settings}

ep_sim = ep_motion.ParticleMotionSimulator(diameter = particles.diameter,  
                                           mobility = particles.mobility, 
                                           rng = np.random.default_rng(0),
                                           **merged_motion_simulator_settings)

# initialize the particle motion simulator with the initial coordinates
ep_sim.init(particles.coordinates)

# create sampling masks for the triggered video simulation
tmask_random, tmask_regular = create_triggered_video_masks(count, seed = 0)
tmask_full = np.logical_or(tmask_random, tmask_regular)

from miepom.utils import print_progress

def triggered_video(seed = 0, count = count, use_sensor = True):
    """Generate a triggered video sequence of particle motion.

    Parameters:
    seed (int): Random seed for initial particle coordinates.
    count (int): Number of frames to generate.
    use_sensor (bool): Whether to use the sensor to compute the image. If set to False, the intensity will be computed directly.

    Yields:
    tuple: A tuple containing two frames (im1, im2) corresponding to random and regular sampling masks.
           If a frame is not captured due to the sampling mask, it yields None for that frame.
    """
    initial_coordinates = create_initial_coordinates(seed)
    tmask_random, tmask_regular = create_triggered_video_masks(count, seed = 0)
    tmask_full = np.logical_or(tmask_random, tmask_regular)
    ep_sim.init(initial_coordinates)

    for i,coordinates in enumerate(ep_sim.iter_coordinates(count)):
        print_progress(i, count)
        if tmask_full[i] == False:
            yield None, None
        else:
            particles.set_coordinates(coordinates)
            if use_sensor:
                im = mp.get_array(microscope_sim.compute_image())
            else:
                im = mp.get_array(microscope_sim.compute_intensity())
            im1 = im if tmask_random[i] else None
            im2 = im if tmask_regular[i] else None
            yield im1, im2
    print_progress(count, count)

def save_video(filename, seq, save_mask = True, ):
    """Save a sequence of images to a file.
    This function first evaluates the sequence of images and then saves them to separate numpy files for the two video streams and their corresponding masks.
    """
    vid1 = []
    vid2 = []
    mask1 = []
    mask2 = []
    for i, frames in enumerate(seq):
        a, b = frames
        if a is not None:
            vid1.append(a)
        mask1.append(1) if a is not None else mask1.append(0)
        if b is not None:
            vid2.append(b)
        mask2.append(1) if b is not None else mask2.append(0)

    np.save(f"{filename}_vid1.npy", vid1)
    np.save(f"{filename}_vid2.npy", vid2)
    if save_mask:
        np.save(f"{filename}_mask1.npy", mask1)
        np.save(f"{filename}_mask2.npy", mask2)

def open_video(filename):
    """Load a sequence of images from a file."""
    vid1 = np.load(f"{filename}_vid1.npy", allow_pickle=True)
    vid2 = np.load(f"{filename}_vid2.npy", allow_pickle=True)
    mask1 = np.load(f"{filename}_mask1.npy", allow_pickle=True)
    mask2 = np.load(f"{filename}_mask2.npy", allow_pickle=True)
    j1 = 0
    j2 = 0
    for i in range(len(mask1)):
        a = vid1[j1] if mask1[i] else None
        b = vid2[j2] if mask2[i] else None
        #if dust is not None:
        #    if a is not None:
        #        a = a * dust
        #    if b is not None:
        #        b = b * dust
        if mask1[i]:
            j1 += 1
        if mask2[i]:
            j2 += 1
        yield a, b
