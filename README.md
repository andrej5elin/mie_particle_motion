# mie_particle_motion

A jupyterlab-based tool for simulating optical motion of small particles dispersed in liquid under influence of external electric field. Imaging model is based on [mie_particle_microscopy](https://github.com/andrej5elin/mie_particle_microscopy).

We apply (optional) random camera acquisition to obtain a video at different times relative to the voltage U of the AC electric field as shown in figure below.

![alttext](https://github.com/andrej5elin/mie_particle_motion/blob/main/sampling.png?raw=true)

We use Brownian motion simulator and the imaging model to compute the particle motion video. Below we plot the first and 15th frame obtained using the triggering scheme shown above.

![alttext](https://github.com/andrej5elin/mie_particle_motion/blob/main/particle_video.png?raw=true)



## Requirements

Implementation is done in python for the jupyter lab IDE. You must have a working jupyter lab environment and install the relevant python packages listed below.

* numpy
* scipy
* matplotlib
* miepython

Optional, but highly recommended for faster computation on GPUs:

* cupy (Windows, Linux)
* mlx (Apple)

## Installation and Usage notes

I recommend using anaconda to build [jupyter lab](https://jupyter.org/install).

```console
$ conda create --name microscope
$ conda activate microscope
$ conda install numpy scipy matplotlib numba miepython
$ conda install jupyter
$ jupyter lab
```

Download the jupyterlab notebooks from the repository and run the code.

