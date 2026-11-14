# mie_particle_motion
[![DOI](https://zenodo.org/badge/1071310148.svg)](https://doi.org/10.5281/zenodo.17285112)

A jupyterlab-based tool for simulating motion of small particles dispersed in liquid under influence of external electric field. Imaging model is based on [mie_particle_microscopy](https://github.com/andrej5elin/mie_particle_microscopy). The toolbox defined here was developed to study the role of experimental parameters, such as capillary dimensions and microscope settings on the obtained video and how these settings affect the analysis of videos when measuring flow, particle size, and electrophoretic mobility using the principles of cross-differential dynamic microscopy [1]. 

We apply (optional) random camera acquisition to obtain a video at different times relative to the voltage U of the AC electric field as shown in figure below. With random triggering scheme one is able to study oscilatory motion with fewer number of processed frames, or to overcome camera speed limitations using a dual-camera setup as explained in [1]

![alttext](https://github.com/andrej5elin/mie_particle_motion/blob/main/sampling.png?raw=true)

We use Brownian motion simulator to obtain trajectories of each particle inside the viewing area. We optionally apply pressure flow, to simulate pressure-induced motion of the particles. Below we plot a few trajectories. Because of the AC electric field, particles oscilate, while pressure-driven flow creates a constant drift, depending on the relative distance from the capillary wall.

![alttext](https://github.com/andrej5elin/mie_particle_motion/blob/main/trajectories.png?raw=true)


Finally, we apply the imaging model to compute the particle motion video. Below we plot the first and 15th frame obtained using the triggering scheme shown above.

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

I recommend using [anaconda](https://www.anaconda.com/download) to build [jupyter lab](https://jupyter.org/install).

```console
$ conda create --name microscope
$ conda activate microscope
$ conda install numpy scipy matplotlib numba miepython
$ conda install jupyter
$ jupyter lab
```

Download the jupyterlab notebooks from the repository and run the code.

## References

[[1]](https://doi.org/10.1039/C9SM00121B) Arko, M. & Petelin, A. *Soft Matter*, 2019,**15**, 2791-2797

