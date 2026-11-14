"""Particle motion in a microfluidic channel under an AC electric field.

Brownian motion, plane Poiseuille flow, sedimentation and AC electrophoresis with a wall-distance-dependent
(complex) electroosmotic contribution. 

Coordinate convention: (x, y, z), the same as miepom. x and y are in the
image plane (the field and the flow point along x, 0 = along x), z is the height in the channel with respect to the central plane.

The channel is assumed to have a rectangular cross-section. Currently, electrophoretic motion is modeled with a wall-distance-dependent electroosmotic contribution,
assuming infinite parallel plates. The Poiseuille flow is modeled as Poiseuille flow between the channel walls asuming rectangular cross-section.

Either hard or mirror boundary conditions can be applied at the channel walls. This is set using the channel_width and channel_height parameters.
These can be set to negative values (mirror boundary conditions), or positive values (hard boundary conditions). 
Note that channel_width can also be infinite (the default) to represent an unbounded channel in the y direction.
By default, channel_length is assumed to be infinite, but it can be set to a finite value to represent a channel of limited length. 
A practtical choice is then to set mirror boundary conditions at the channel ends so that 
particles do not aggregate at the channel ends when pressure gradient is applied, to simulate continuous feed-in of particles.

Time step is implemented as an iterative update of particle coordinates based on the current velocity and Brownian motion.

- ``ParticleMotionSimulator.iter_coordinates`` / ``get_next_coordinates``
  (NumPy, or the array backend set with ``set_backend``). Brownian steps are
  drawn from the ``numpy.random.Generator`` passed to the simulator. 

Precision is float32 unless the environment variable

``EPSIM_USE_DOUBLE_PRECISION=1`` is set before this module is imported.
"""

from __future__ import annotations

from cmath import cos
import os
from functools import cache
from pathlib import Path

import numpy as np

# xp stands for the default array backend
# we need to set the default array backend to numpy initially 
import numpy as xp

#: flags to switch off parts of the motion model (combine with |)

#: disableelectroosmotic contribution
DISABLE_WALL_EFFECT = 1
#: disable Poiseuille flow
DISABLE_POISEUILLE_FLOW = 2
#: disable sedimentation due to gravity
DISABLE_SEDIMENTATION = 4
#: disable Brownian motion
DISABLE_BROWNIAN_MOTION = 8
#: disable electrophoresis altogether
DISABLE_ELECTROPHORESIS = 16
#: combination of flags to disable all velocity contributions
DISABLE_VELOCITY = DISABLE_ELECTROPHORESIS | DISABLE_SEDIMENTATION | DISABLE_POISEUILLE_FLOW
#: enable low-frequency approximation, valid if simulation time step multiplied with the AC frequency is much smaller than one
ENABLE_LOW_FREQ_APPROXIMATION = 32
#: enable surface adhesion
ENABLE_SURFACE_ADHESION = 64

#: whether we use double precision in calculations 
USE_DOUBLE_PRECISION = os.environ.get("EPSIM_USE_DOUBLE_PRECISION", "0") == "1"    

try:
    import numba as nb
    # this flag indicates whether numba is available for accelerated computations
    # it also indicates it numba backend is activated or not.
    # see enable_numba for how to activate the numba backend.
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

def set_backend(backend) -> None:
    """Sets the array module (numpy, cupy, mlx.core, ...) used by the ``*_xp`` functions."""
    if backend.__name__ != "numpy":
        if USE_DOUBLE_PRECISION:
            raise ValueError("Only numpy backend is currently supported for double precision calculations")
        if USE_NUMBA:
            raise ValueError("Numba is enabled. Disable numba before using a non-numpy backend")
    globals()["xp"] = backend

def enable_numba() -> None:
    """Enables the use of numba for accelerated computations."""
    global USE_NUMBA, xp
    if xp.__name__ != "numpy":
        raise ValueError("Numba can only be enabled with the numpy backend. Switch to the numpy backend first.")
    try:
        # attempt to import numba to ensure it is available and can be used
        import numba
    except ImportError:
        raise ImportError("Numba is not installed. Install numba to enable it.")
    USE_NUMBA = True

def disable_numba() -> None:
    """Disables the use of numba for accelerated computations."""
    global USE_NUMBA
    USE_NUMBA = False   

if USE_DOUBLE_PRECISION:
    print("epsim: Using double precision")

    def fdtype():
        """backend-specific floating point dtype"""
        return xp.float64

    def cdtype():
        """backend-specific complex floating point dtype"""
        return xp.complex128

    if USE_NUMBA:
        # numba dtypes - needed for compatibility with numba-compiled functions
        nbfloat = nb.float64
        nbcomplex = nb.complex128
        nbint = nb.int32

else:
    print("epsim: Using single precision")
    def fdtype():
        """backend-specific floating point dtype"""
        return xp.float32

    def cdtype():
        """backend-specific complex floating point dtype"""
        return xp.complex64

    if USE_NUMBA:
        nbfloat = nb.float32
        nbcomplex = nb.complex64
        nbint = nb.int32

#: Boltzmann constant, J/K
K_B = 1.380649e-23

#: vacuum permittivity, F/m
EPSILON_0 = 8.85e-12

#: temperature (deg C) vs. dynamic viscosity (Pa s) of water, IAPWS-2008,
#: from https://wiki.anton-paar.com/en/water/
WATER_VISCOSITY_CSV = Path(__file__).resolve().parent / "csv/water_viscosity.csv"

WATER_PERMITTIVITY_CSV = Path(__file__).resolve().parent / "csv/water_permittivity.csv"

WATER_DENSITY_CSV = Path(__file__).resolve().parent / "csv/water_density.csv"

@cache
def _water_viscosity_table(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads (and caches) the water viscosity reference table."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]

@cache
def _water_permittivity_table(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads (and caches) the water permittivity reference table."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]

@cache
def _water_density_table(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads (and caches) the water density reference table."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]

def water_viscosity(temperature: float, csv_path: str | Path = WATER_VISCOSITY_CSV) -> float:
    """Dynamic viscosity of water, by linear interpolation of the IAPWS-2008 table.

    Parameters
    ----------
    temperature : float
        K. Must lie within the table (2-80 deg C).
    csv_path : str or Path
        Reference table with columns temperature (deg C), viscosity (Pa s).

    Returns
    -------
    float
        Pa s.
    """
    temperature_c, viscosity_pa_s = _water_viscosity_table(str(csv_path))
    t = float(temperature) - 273.15
    if not temperature_c[0] <= t <= temperature_c[-1]:
        raise ValueError(
            f"temperature {temperature} K is outside the water viscosity table "
            f"({temperature_c[0]}-{temperature_c[-1]} deg C)"
        )
    return float(np.interp(t, temperature_c, viscosity_pa_s))

def water_permittivity(temperature: float, csv_path: str | Path = WATER_PERMITTIVITY_CSV) -> float:
    """Relative permittivity of water, by linear interpolation of the reference table.

    Parameters
    ----------
    temperature : float
        K. Must lie within the table (0-100 deg C).
    csv_path : str or Path
        Reference table with columns temperature (deg C), permittivity (dimensionless).

    Returns
    -------
    float
        Relative permittivity (dimensionless).
    """
    temperature_c, permittivity = _water_permittivity_table(str(csv_path))
    t = float(temperature) - 273.15
    if not temperature_c[0] <= t <= temperature_c[-1]:
        raise ValueError(
            f"temperature {temperature} K is outside the water permittivity table "
            f"({temperature_c[0]}-{temperature_c[-1]} deg C)"
        )
    return float(np.interp(t, temperature_c, permittivity))

def water_density(temperature: float, csv_path: str | Path = WATER_DENSITY_CSV) -> float:
    """Density of water, by linear interpolation of the reference table.

    Parameters
    ----------
    temperature : float
        K. Must lie within the table (0-100 deg C).
    csv_path : str or Path
        Reference table with columns temperature (deg C), density (kg/m^3).

    Returns
    -------
    float
        Density, kg/m^3.
    """
    temperature_c, density = _water_density_table(str(csv_path))
    t = float(temperature) - 273.15
    if not temperature_c[0] <= t <= temperature_c[-1]:
        raise ValueError(
            f"temperature {temperature} K is outside the water density table "
            f"({temperature_c[0]}-{temperature_c[-1]} deg C)"
        )
    return float(np.interp(t, temperature_c, density))  

def sedimentation_velocity(d : float, nu : float, rho_p : float, rho_w : float, g : float) -> float:
    """Stokes sedimentation velocity of a sphere.

    Parameters
    ----------
    d : array_like
        Particle diameter, m.
    nu : float
        Dynamic viscosity of the fluid, Pa s.
    rho_p, rho_w : float
        Particle and fluid density, kg/m^3.
    g : array_like, shape (3,)
        Gravitational acceleration vector, m/s^2.

    Returns
    -------
    array
        Velocity, m/s (broadcast of ``d`` and ``g``).
    """
    # Downward velocity due to gravity if rho_p > rho_w and g < 0 (g vector pointing downward)
    return (d**2) * (rho_p - rho_w) * g / (18 * nu)

def diffusion_coefficient(d : float, nu : float, T : float) -> float:
    """Stokes-Einstein diffusion coefficient.

    Parameters
    ----------
    d : array_like
        Particle diameter, m.
    nu : float
        Dynamic viscosity of the fluid, Pa s.
    T : float
        Temperature, K.

    Returns
    -------
    array
        Diffusion coefficient, m^2/s.
    """
    return (T * K_B) / (3 * np.pi * nu * d)

def elliptical_poiseuille_velocity(y : float, z : float, w : float, h : float, g : float, mu : float) -> float:
    """Elliptical Poiseuille flow velocity at coordinates ``y`` and ``z`` in an elliptical pipe.

    Parameters
    ----------
    y : array_like
        Distance from the center along the major axis, m.
    z : array_like
        Distance from the center along the minor axis, m.
    w : float
        Major axis diameter, can be infinite, m.
    h : float
        Minor axis diameter, must be finite, m.
    g : float
        Pressure gradient (-dp/dx), Pa/m.
    mu : float
        Dynamic viscosity of the fluid, Pa s.

    Returns
    -------
    array
        Velocity, m/s.
    """
    a = w / 2
    b = h / 2

    ba = b/a

    scale = (b)**2 / (1 + ba**2)
    return (g / (2 * mu)) * scale * (1 - (y/a)**2 - (z/b)**2)

def rectangular_poiseuille_velocity(y : float, z : float, w : float, h : float, g : float, mu : float, terms : int = 100) -> float:
    """Computes the velocity profile u(y, z) for rectangular Poiseuille flow.

    Note that width can be set to infinity to recover the parabolic profile between two infinite plates.
    Height of the channel must be finite.
    
    Parameters
    ----------
    y : array_like
        Distance from the central plane in the channel along the width, m.
    z : array_like
        Distance from the central plane in the channel along the height, m.
    w : float
        Channel width, m.
    h : float
        Channel height, m.
    g : float
        Pressure gradient (-dp/dx), Pa/m.
    mu : float
        Dynamic viscosity of the fluid, Pa s.
    terms : int, optional
        Number of odd Fourier terms to sum over (default is 100).

    Returns
    -------
    array
        Velocity, m/s.
    """
    if xp.__name__ == "numpy":
        return _rectangular_poiseuille_velocity_np(y, z, w, h, g, mu, terms)
    else:
        return _rectangular_poiseuille_velocity_xp(y, z, w, h, g, mu, terms)

def parabolic_velocity(z : float, h : float, u_c : float) -> float:
    """Parabolic velocity profile between two infinite plates.

    Parameters
    ----------
    z : float
        Distance from the center plane, m.
    h : float
        Channel height, m.
    u_c : float
        Central velocity, m/s.

    Returns
    -------
    float
        Velocity, m/s.
    """
    return u_c * (1 - (2 * z / h)**2)

def _rectangular_poiseuille_velocity_np(y : float, z : float, w : float, h : float, g : float, mu : float, terms : int = 100) -> float:
    hh = h/2
    wh = w/2
    hh2 = hh**2

    scale = (g / (2 * mu))

    # base profile for infinite plates
    u_base = scale * (hh2 - z**2)

    u_series = 0.

    factor = (32 * scale * hh2) / (np.pi**3)
    
    for n in range(1, 2 * terms, 2):
        beta_n = (n * np.pi) / h

        term1 = np.exp(beta_n * (y - wh ))
        term2 = np.exp(-beta_n * (y + 3*wh))
        term3 = np.exp(-beta_n * (y + wh ))
        term4 = np.exp(beta_n * (y - 3 * wh))
        
        num = term1 - term2 + term3 - term4
        den = 1.0 - np.exp(-2 * beta_n * w)
        
        wall_correction = num / den
        
        term = (1.0 / (n**3)) * wall_correction * np.sin(beta_n * (z + hh))
        u_series += term
        
    return u_base - (factor * u_series)

#TODO: Implement a more efficient the _rectangular_poiseuille_velocity_xp function for other array libraries like CuPy or MLX
_rectangular_poiseuille_velocity_xp = _rectangular_poiseuille_velocity_np

if USE_NUMBA:
    _rectangular_poiseuille_velocity_nb = nb.njit([nbfloat(nbfloat, nbfloat, nbfloat, nbfloat, nbfloat, nbfloat, nbint)])(_rectangular_poiseuille_velocity_np)

def wall_mobility(eps : float, zeta : float, nu : float) -> float:
    """Helmholtz-Smoluchowski wall (electroosmotic) mobility.

    Parameters
    ----------
    eps : float
        Relative permittivity of the fluid, dimensionless.
    zeta : float
        Wall zeta potential, V.
    nu : float
        Dynamic viscosity, Pa s.

    Returns
    -------
    float
        Mobility, m^2/(V s).
    """
    return eps * EPSILON_0 * zeta / nu

def delta_inv(nu : float, rho_w : float, f : float) -> float:
    """Inverse viscous penetration depth of the oscillating Stokes layer.

    Parameters
    ----------
    nu : float
        Dynamic viscosity, Pa s.
    rho_w : float
        Fluid density, kg/m^3.
    f : float
        AC frequency, Hz.

    Returns
    -------
    float
        Inverse viscous penetration depth, 1/m.
    """
    omega = 2 * np.pi * f
    return np.sqrt(rho_w * omega / (2 * nu))

def lambda_(nu : float, rho_w : float, f : float) -> complex:
    """Complex wavenumber (1 + i) * delta_inv of the oscillating Stokes layer.

    Parameters
    ----------
    nu : float
        Dynamic viscosity, Pa s.
    rho_w : float
        Fluid density, kg/m^3.
    f : float
        AC frequency, Hz.

    Returns
    -------
    complex
        1/m, with delta_inv = sqrt(rho_w omega / (2 nu)) the inverse viscous penetration depth.
    """
    return (1. + 1j) * delta_inv(nu, rho_w, f)

def complex_mobility(m : float, z : float, h : float, w : float, l : complex) -> complex:
    """Particle mobility plus the wall-driven electroosmotic contribution at height ``z``.

    Parameters
    ----------
    m : array_like
        Electrophoretic mobility of the particle, m^2/(V s).
    z : array_like
        Height in the channel, m.
    h : float
        Channel height, m.
    w : float
        Wall mobility (``wall_mobility_xp``), m^2/(V s).
    l : complex
        Stokes-layer wavenumber (``lambda_xp``), 1/m.

    Returns
    -------
    complex array
        Complex mobility, m^2/(V s).
    """
    lh2 = l * h * 0.5
    lz = l * z
    cosh_term = xp.cosh(lz)
    norm = xp.divide(1.,xp.cosh(lh2))
    return m + w * cosh_term * norm

if USE_NUMBA:
    # we have numba, so we can create a JIT-compiled version of complex_mobility
    _complex_mobility_nb = nb.njit([nbcomplex(nbfloat, nbfloat, nbfloat, nbfloat, nbcomplex)], fastmath = True)(complex_mobility)

def _brownian_step_np(d, dt, rng: np.random.Generator):
    """Brownian step over ``dt`` (NumPy).

    Parameters
    ----------
    d : ndarray
        Diffusion coefficients, m^2/s (any shape; one draw per element).
    dt : float
        Time step, s.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray
        Displacement, m, same shape as ``d``.
    """
    scale = np.sqrt((2. * d *  dt))
    return rng.standard_normal(size=scale.shape, dtype=fdtype()) * scale

def _brownian_step_xp(d, dt):
    """Brownian step over ``dt`` with the array backend's global RNG (cupy/mlx, and numba).

    ``d`` in m^2/s, ``dt`` in s; returns m.
    """
    scale = xp.sqrt((2. * d * dt))
    return xp.random.normal(loc=0., scale=scale)

def _brownian_step_mx(d, dt):
    """Brownian step over ``dt``, the mlx.core version

    ``d`` in m^2/s, ``dt`` in s; returns m.
    """
    scale = xp.sqrt((2. * d * dt))
    return xp.random.normal(shape = scale.shape, loc=0., scale=scale)

if USE_NUMBA:
    _brownian_step_nb = nb.njit([nbfloat(nbfloat, nbfloat)], fastmath = True)(_brownian_step_xp)
    _brownian_step_vec_nb = nb.vectorize([nbfloat(nbfloat, nbfloat)], fastmath = True, target = "parallel")(_brownian_step_xp)
    
def brownian_step(d, dt, rng: np.random.Generator | None = None):
    """Brownian step over ``dt``

    On the NumPy backend the draw comes from ``rng`` (required). On other
    backends the backend's own global RNG is used.

    ``d`` in m^2/s, ``dt`` in s; returns m.
    """
    if xp.__name__ == "numpy":
        if USE_NUMBA:
            return _brownian_step_vec_nb(d, dt)
        else:
            if rng is None:
                rng = np.random.default_rng()
            return _brownian_step_np(d, dt, rng)
    elif xp.__name__ == "mlx.core":
        return _brownian_step_mx(d, dt)
    else:
        return _brownian_step_xp(d, dt)

def effective_mobility(m, t, f, p, dt):
    """Effective (time-averaged over one step) mobility for a sinusoidal field.

    Parameters
    ----------
    m : complex array_like
        Complex mobility, m^2/(V s).
    t : float
        Current time, s.
    f : float
        AC frequency, Hz.
    p : float
        AC phase, rad.
    dt : float
        Time step to next time, s.

    Returns
    -------
    array
        Real mobility, m^2/(V s): (1/dt) times the integral of |m| sin(omega t' + phi + p)
        over the step ending at t. Multiply by the field amplitude to get a velocity.
    """
    omega =  f * 2 * xp.pi
    omega_dt = omega * dt
    phi = xp.arctan2(m.imag, m.real)
    phase = omega * t + phi + p

    cosfi = xp.cos(phase + omega_dt)
    cosfi0 = xp.cos(phase)
    step = xp.divide(cosfi0 - cosfi, omega_dt)

    return xp.abs(m) * step 

def effective_mobility_low_frequency(m, t, f, p, dt):
    """Effective (time-averaged over one step) mobility for a sinusoidal field.

    Parameters
    ----------
    m : complex array_like
        Complex mobility, m^2/(V s).
    t : float
        Current time, s.
    f : float
        AC frequency, Hz.
    p : float
        AC phase, rad.
    dt : float
        Time step to next time, s.

    Returns
    -------
    array
        Real mobility, m^2/(V s): (1/dt) times the integral of |m| sin(omega t' + phi + p)
        over the step ending at t. Multiply by the field amplitude to get a velocity.
    """
    omega =  f * 2 * xp.pi
    omega_dt = omega * dt
    phi = xp.arctan2(m.imag, m.real)
    phase = omega * t + phi + p
    s = xp.sin(phase)
    c = xp.cos(phase)
    # Use a third-order Taylor expansion for small omega_dt to avoid numerical issues.
    # another option would be to us sinc function for small omega_dt, but sinc is not available in mlx
    step = s * (1 - omega_dt**2 / 6) + c * (omega_dt / 2 - omega_dt**3 / 24)
    return xp.abs(m) * step 

def particle_velocity(x, t, dt, v, g, mu, w, h, mp, mw, l, e, f, p, flags = 0, terms = 100):
    """Particle velocity (flow + sedimentation + electrophoresis), m/s.

    Parameters
    ----------
    x : array, shape (..., 3)
        Positions, m.
    t : float
        Time, s.
    dt : float
        Time step, s.
    g : float
        Pressure gradient, Pa/m.
    mu : float
        Fluid viscosity, Pa s.
    w : float
        Channel width, m.
    h : float
        Channel height, m.
    v : array, shape (..., 3)
        Sedimentation velocity, m/s.
    mp : array
        Particle mobility, m^2/(V s).
    mw : float
        Wall mobility, m^2/(V s).
    l : complex
        Stokes-layer wavenumber, 1/m.
    e : float
        Electric field amplitude, V/m.
    f : float
        AC frequency, Hz.
    p : float
        AC phase, rad.
    flags : int
        ``DISABLE_*`` flags.
    terms : int
        Number of terms in the series expansion for the Poiseuille flow.
    """
    if USE_NUMBA:
        if flags not in (0, ENABLE_LOW_FREQ_APPROXIMATION):
            raise ValueError("Unsupported flags combination for the numba implementation. DISABLE_ flags are not supported.")
        if flags & ENABLE_LOW_FREQ_APPROXIMATION:
            return _particle_velocity_low_frequency_nb(x, t, dt, v, g, mu, w, h, mp, mw, l, e, f, p, terms)
        else:  
            return _particle_velocity_nb(x, t, dt, v, g, mu, w, h, mp, mw, l, e, f, p, terms)
    else:
        return _particle_velocity_xp(x, t, dt, v, g, mu, w, h, mp, mw, l, e, f, p, flags, terms=terms)
 
def _particle_velocity_xp(x, t, dt, v, g, mu, w, h, mp, mw, l, e, f, p, flags=0, terms = 100):
    z = x[..., 2]
    y = x[..., 1]

    out = xp.zeros(x.shape, x.dtype)

    if not flags & DISABLE_POISEUILLE_FLOW:
        vp = _rectangular_poiseuille_velocity_xp(y, z, w, h, g, mu, terms=terms)
        out[..., 0] += vp

    if not flags & DISABLE_SEDIMENTATION:
        out += v

    if not flags & DISABLE_WALL_EFFECT:
        mp = complex_mobility(mp, z, h, mw, l)

    if not flags & DISABLE_ELECTROPHORESIS:
        if flags & ENABLE_LOW_FREQ_APPROXIMATION:
            m_eff = effective_mobility_low_frequency(mp, t, f, p, dt)
        else:
            m_eff = effective_mobility(mp, t, f, p, dt)
        out[..., 0] += m_eff * e
    return out

if USE_NUMBA:
    _effective_mobility_nb = nb.njit([nbfloat(nbcomplex, nbfloat, nbfloat, nbfloat, nbfloat)], fastmath=True)(effective_mobility)
    _effective_mobility_low_frequency_nb = nb.njit([nbfloat(nbcomplex, nbfloat, nbfloat, nbfloat, nbfloat)], fastmath=True)(effective_mobility_low_frequency)

    @nb.guvectorize(
        [
            (
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbcomplex[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbint[:],
                nbfloat[:],
            )
        ],
        "(n),(),(),(n),(),(),(),(),(),(),(),(),(),(),()->(n)",
        target="parallel",
        fastmath=True
    )
    def _particle_velocity_nb(x, t, dt, v, g, mu, w, h, mp, mw, l, e, f, p, terms, out):
        h0 = h[0]
        l0 = l[0]
        t0 = t[0]
        f0 = f[0]
        p0 = p[0]
        dt0 = dt[0]
        e = e[0]
        z = x[2]
        y = x[1]

        out0 = v[0]

        vp = _rectangular_poiseuille_velocity_nb(y, z, w[0], h[0], g[0], mu[0], terms[0])
        out0 += vp

        mc = _complex_mobility_nb(mp[0], z, h0, mw[0], l0)    
        m_eff =  _effective_mobility_nb(mc, t0, f0, p0, dt0)
    
        out0 += m_eff * e
        
        out[0] = out0
        out[1] = v[1]
        out[2] = v[2]

    @nb.guvectorize(
        [
            (
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbcomplex[:],
                nbfloat[:],
                nbfloat[:],
                nbfloat[:],
                nbint[:],
                nbfloat[:],
            )
        ],
        "(n),(),(),(n),(),(),(),(),(),(),(),(),(),(),()->(n)",
        target="parallel",
        fastmath=True
    )
    def _particle_velocity_low_frequency_nb(x, t, dt, v, g, mu, w, h, mp, mw, l, e, f, p, terms, out):
        h0 = h[0]
        l0 = l[0]
        t0 = t[0]
        f0 = f[0]
        p0 = p[0]
        dt0 = dt[0]
        e = e[0]
        z = x[2]
        y = x[1]

        out0 = v[0]

        vp = _rectangular_poiseuille_velocity_nb(y, z, w[0], h[0], g[0], mu[0], terms[0])
        out0 += vp

        mc = _complex_mobility_nb(mp[0], z, h0, mw[0], l0) 
        m_eff =  _effective_mobility_low_frequency_nb(mc, t0, f0, p0, dt0)   
    
        out0 += m_eff * e
        
        out[0] = out0
        out[1] = v[1]
        out[2] = v[2]

def _apply_boundary_condition_xp(x,h):
    ha = xp.abs(h)
    where = xp.abs(x) > ha/2
    out = x + ha/2
    out = out % ha
    out = out - ha/2
    out = out * where * (-1 * xp.sign(h))
    return xp.where(where, out, x)

if USE_NUMBA:
    @nb.njit([nb.float64(nb.float64, nb.float64)])
    def _apply_boundary_condition_nb(x,h):
        ha = np.abs(h)
        where = np.abs(x) > ha/2
        if where:
            out = x + ha/2
            out = out % ha
            out = out - ha/2
            out = out * where * (-1 * np.sign(h))
            return out
        return x

def compute_next_coordinates(x, v, d, h, dt, flags = 0, rng = None, out = None):
    """
    Computes the next coordinates of particles based on their current positions, velocities, and time step.

    Parameters
    ----------
    x : array_like, shape (num_particles, 3)
        Current positions of the particles, m.
    v : array_like, shape (num_particles, 3)
        Current velocities of the particles, m/s.
    h : float
        Channel height, m.
    dt : float
        Time step, s.
    flags : int, optional
        Flags to control the behavior of the computation. Default is 0.
    rng : object, optional
        Random number generator to use for stochastic components. If None, the default RNG is used.
    out : array_like, shape (num_particles, 3), optional
        Array to store the output coordinates. If None, a new array is created.

    Returns
    -------
    out : array_like, shape (num_particles, 3)
        Updated positions of the particles, m.
    """
    if xp.__name__ == "mlx.core":
        if out is None:
            return _compute_next_coordinates_mx(x, v, d, h, dt, flags)
        else:
            out[:] = _compute_next_coordinates_mx(x, v, d, h, dt, flags)
            return out
    elif xp.__name__ == "numpy":
        if USE_NUMBA:
            return _compute_next_coordinates_nb(x, v, d, h, dt, out)
        else:
            return _compute_next_coordinates_np(x, v, d, h, dt, flags, rng, out)
    else:
        return _compute_next_coordinates_xp(x, v, d, h, dt, flags, out)

def _compute_next_coordinates_np(x, v, d, h, dt, flags = 0, rng = None, out = None):
    """numpy implementation of computing the next coordinates of particles."""
    if flags & ENABLE_SURFACE_ADHESION:
        mask = 1.
        for i in range(3):
            mask *= xp.abs(x[...,i]) < xp.abs(h[i])/2
        v = v * mask[..., None]
        d = d * mask[..., None]

    dx = v * dt
    if not flags & DISABLE_BROWNIAN_MOTION:
        # using brownian_step instead of _brownian_step_np so that we set default rng in case rng is None
        dx = xp.add(dx, brownian_step(d, dt, rng), out = dx)

    out = xp.add(x, dx, out=out)

    for i, hi in enumerate(h):
        if hi < np.inf:
            # for negative hi, we apply mirror boundary conditions
            # for positive hi, we apply elastic boundary conditions or surface adhesion if enabled
            if not flags & ENABLE_SURFACE_ADHESION or hi < 0:
                out[...,i] = _apply_boundary_condition_xp(out[...,i], hi)

    return out

def _compute_next_coordinates_xp(x, v, d, h, dt, flags = 0, out = None):
    """Generic array library implementation of computing the next coordinates of particles."""
    dx = v * dt
    if not flags & DISABLE_BROWNIAN_MOTION:
        dx = xp.add(dx, _brownian_step_xp(d, dt), out = dx)
    out = xp.add(dx, x, out = out)
    for i, hi in enumerate(h):
        out[...,i] = _apply_boundary_condition_xp(out[...,i], hi)
    return out

def _compute_next_coordinates_mx(x, v, d, h, dt, flags = 0):
    """Mlx core implementation of computing the next coordinates of particles."""
    dx = v * dt
    if not flags & DISABLE_BROWNIAN_MOTION:
        dxb = _brownian_step_mx(d, dt)
        dx = dx + dxb
    out = dx + x
    out = _apply_boundary_condition_xp(out, h)
    return out
 
@nb.guvectorize([(nbfloat[:],nbfloat[:],nbfloat[:],nbfloat[:],nbfloat[:],nbfloat[:])],"(n),(n),(n),(n),()->(n)", fastmath = True, target="parallel")
def _compute_next_coordinates_nb(x, v, d, h, dt, out):
    """Numba implementation of computing the next coordinates of particles."""
    dt0 = dt[0] 

    dx0 = _brownian_step_nb(d[0], dt0) 
    dx1 = _brownian_step_nb(d[1], dt0) 
    dx2 = _brownian_step_nb(d[2], dt0) 

    out[0] = dx0 + x[0] + v[0] * dt0
    out[1] = dx1 + x[1] + v[1] * dt0
    out[2] = dx2 + x[2] + v[2] * dt0

    out[0] = _apply_boundary_condition_nb(out[0], h[0])
    out[1] = _apply_boundary_condition_nb(out[1], h[1])
    out[2] = _apply_boundary_condition_nb(out[2], h[2])

class ParticleMotionSimulator:
    """Time stepper for particles in a channel under an AC field.

    Parameters
    ----------
    diameter : array_like, shape (num_particles,)
        Particle diameters, m.
    mobility : array_like, shape (num_particles,)
        Electrophoretic mobilities, m^2/(V s).
    viscosity : float, optional
        Dynamic viscosity of the fluid, Pa s. Default (None): water at ``temperature``
        (``water_viscosity``).
    permittivity : float
        Relative permittivity of the fluid, dimensionless.
    wall_zeta : float
        Wall zeta potential, V.
    temperature : float
        K.
    particle_density, fluid_density : float
        kg/m^3.
    ac_period : float
        Period of the AC field, s.
    ac_phase : float
        Phase of the AC field at t = 0, rad.
    ac_voltage : float
        Amplitude of the AC voltage across the channel, V.
    electric_field : array_like, shape (3,), optional
        Field amplitude, V/m. Default: ac_voltage / channel_length along ``channel_orientation``.
    central_velocity : float
        Poiseuille flow velocity at the channel center, m/s, along ``channel_orientation``.
    gravitational_acceleration : array_like, shape (3,)
        m/s^2.
    coordinates : array_like, shape (num_particles, 3), optional
        Initial positions, m. Default: all at the origin.
    flags : int
        ``DISABLE_*`` flags.
    time_step : float
        s.
    channel_height : float
        m.
    channel_length : float
        Electrode-to-electrode length, m.
    rng : numpy.random.Generator, optional
        Source of the Brownian steps on the NumPy backend. Required unless
        ``DISABLE_BROWNIAN_MOTION`` is set.
    """

    def __init__(
        self,
        diameter=1e-6,
        mobility=5e-8,
        particle_density=None,
        viscosity=None,
        permittivity=None,
        fluid_density=None,
        temperature = None,
        temperature_c = None,
        ac_frequency=None,
        ac_period = None,
        ac_phase = None,
        ac_voltage = 0,
        electrode_distance = 0.0156,
        gravitational_acceleration=(0, 0, -9.81),
        coordinates=None,
        flags=0,
        time_step=1e-4,
        channel_width = np.inf,
        channel_height = 100e-6,
        channel_length= np.inf,
        pressure_gradient=None,
        wall_zeta=80e-3,
        terms=None,
        rng: np.random.Generator | None = None,
    ):

        if temperature is not None and temperature_c is not None:
            raise ValueError("Cannot specify both temperature and temperature_c")
        if ac_period is not None and ac_frequency is not None:
            raise ValueError("Cannot specify both ac_period and ac_frequency")
        
        if rng is None and not flags & DISABLE_BROWNIAN_MOTION and xp.__name__ == "numpy":
            raise ValueError("ParticleMotionSimulator needs rng (a numpy.random.Generator) for Brownian motion")
        
        self.rng = rng

        self.temperature = xp.array(temperature, fdtype()) if temperature is not None else xp.array(298, fdtype())
        if temperature_c is not None:
            self.temperature_c = temperature_c

        viscosity = water_viscosity(self.temperature) if viscosity is None else viscosity
        permittivity = water_permittivity(self.temperature) if permittivity is None else permittivity
        fluid_density = water_density(self.temperature) if fluid_density is None else fluid_density
        particle_density = water_density(self.temperature) if particle_density is None else particle_density
        pressure_gradient = 0 if pressure_gradient is None else pressure_gradient

        self.ac_frequency = xp.array(ac_frequency, fdtype()) if ac_frequency is not None else xp.array(0, fdtype())
        if ac_period is not None:
            self.ac_period = ac_period

        if self.ac_period == xp.inf:
            if ac_phase is not None:
                raise ValueError("Cannot specify ac_phase when ac_period is infinite")
            # If the AC period is infinite, set the phase to pi/2 so that we can deal with it as a DC field and not a sine wave
            ac_phase = xp.pi/2 
            print("AC period is infinite, setting ac_phase to pi/2")
        else:
            ac_phase = ac_phase if ac_phase is not None else 0.

        self.diameter = xp.array(diameter, fdtype())
        self.mobility = xp.array(mobility, fdtype())
        self.viscosity = xp.array(viscosity, fdtype())
        self.permittivity = xp.array(permittivity, fdtype())
        self.wall_zeta = xp.array(wall_zeta, fdtype())
        self.particle_density = xp.array(particle_density, fdtype())
        self.fluid_density = xp.array(fluid_density, fdtype())
        self.terms = terms if terms is not None else (100 if USE_DOUBLE_PRECISION else 20)
        self.time_step = xp.array(time_step, fdtype())
        self.ac_phase = xp.array(ac_phase, fdtype())
        self.ac_voltage = xp.array(ac_voltage, fdtype())
        self.channel_height = xp.array(channel_height, fdtype())
        self.channel_width = xp.array(channel_width, fdtype())
        self.channel_length = xp.array(channel_length, fdtype())
        self.electrode_distance = xp.array(electrode_distance, fdtype())
        self.pressure_gradient = xp.array(pressure_gradient, fdtype())

        self.electric_field = self.ac_voltage / self.electrode_distance
   
        self.wall_mobility = xp.array(wall_mobility(self.permittivity, self.wall_zeta, self.viscosity), fdtype())
        self.lambda_ = xp.array(lambda_(self.viscosity, self.fluid_density, self.ac_frequency), cdtype())
        self.diffusion_coefficients = xp.zeros((len(self.diameter), 3), fdtype())
        self.diffusion_coefficients[:] = xp.array(
            diffusion_coefficient(self.diameter[:, None], self.viscosity, self.temperature), fdtype()
        )
        self.gravitational_acceleration = xp.array(gravitational_acceleration, fdtype())
        self.sedimentation_velocity = xp.array(sedimentation_velocity(
            self.diameter[:, None],
            self.viscosity,
            self.particle_density,
            self.fluid_density,
            self.gravitational_acceleration,
        ), fdtype())

        if self.channel_width == xp.inf:
            # no need to compute correction terms for velocity calculation in an infinitely wide channel
            self.terms = 0 

        self.flags = flags

        threshold = 1e-4 if USE_DOUBLE_PRECISION else 1e-2

        if self.ac_frequency * self.time_step < threshold:
            # force low frequency approximation so that we can deal with the zero frequency limit
            self.flags |= ENABLE_LOW_FREQ_APPROXIMATION
            print("Low frequency approximation enabled due to small AC frequency.")

        if coordinates is None:
            coordinates = xp.zeros(self.diameter.shape +  (3,), dtype=fdtype())
        else:
            coordinates = xp.array(coordinates, fdtype(), copy=True)

        self.init(coordinates)

    @property
    def central_poiseuille_velocity(self):
        """Central Poiseuille velocity, m/s."""
        return rectangular_poiseuille_velocity(0.0, 0.0, self.channel_width, self.channel_height, self.pressure_gradient, self.viscosity, terms = self.terms)

    @property
    def temperature_c(self):
        return xp.array(self.temperature - 273.15, fdtype())

    @temperature_c.setter
    def temperature_c(self, value):
        self.temperature = xp.array(value + 273.15, fdtype())

    @property
    def ac_period(self):
        """Returns the AC period, s."""
        if self.ac_frequency == 0.:
            return xp.array(np.inf, fdtype())
        return xp.array(1.0 / self.ac_frequency, fdtype())

    @ac_period.setter
    def ac_period(self, value):
        """Sets the AC period, s."""
        self.ac_frequency = xp.array(1.0 / value, fdtype())

    @property
    def current_time(self):
        """Returns the current simulation time, s."""
        if self.count == 0:
            return xp.array(0.0, fdtype())
        return self.time_step * (self.count - 1.)

    @property
    def channel_dimensions(self):
        return xp.array((self.channel_length, self.channel_width, self.channel_height), fdtype())

    def init(self, coordinates) -> None:
        """Resets the simulator to time 0 at ``coordinates`` (num_particles, 3), m."""
        self.set_current_coordinates(coordinates)
        self.count = 0

    def print_info(self) -> None:
        """Prints the experiment settings and the simulator state."""
        print("Experiment settings:")
        print("--------------------")
        print(f"Diameter mean {np.mean(self.diameter)}")
        print(f"Mobility mean {np.mean(self.mobility)}")
        print(f"Particle density mean {np.mean(self.particle_density)}")
        print(f"Channel width {self.channel_width}")
        print(f"Channel height {self.channel_height}")
        print(f"Channel length {self.channel_length}")
        print(f"Pressure gradient {self.pressure_gradient}")
        print(f"Fluid permittivity {self.permittivity}")
        print(f"Fluid density {self.fluid_density}")
        print(f"Fluid viscosity {self.viscosity}")
        print(f"Temperature {self.temperature}")
        print(f"Gravitational acceleration {self.gravitational_acceleration}")
        print(f"Sedimentation velocity mean {np.mean(self.sedimentation_velocity, axis=0)}")
        print(f"Central Poiseuille velocity {self.central_poiseuille_velocity}")
        print()
        print(f"Wall zeta {self.wall_zeta}")
        print(f"Wall mobility {self.wall_mobility}")
        print(f"AC voltage {self.ac_voltage}")
        print(f"AC frequency {self.ac_frequency}")
        print(f"AC phase {self.ac_phase}")
        print(f"Electric field {self.electric_field}")
        print()
        print("Simulator state:")
        print("----------------")
        print(f"Time step {self.time_step}")
        print(f"Flags {self.flags}")
        print(f"Current time {self.current_time}")
        print(f"Count {self.count}")
        print(f"Total simulation time {self.time_step * self.count}")
        print(f"Terms {self.terms}")

    def set_current_coordinates(self, coordinates) -> None:
        """Sets the current positions, (num_particles, 3), m."""
        self.current_coordinates = xp.array(coordinates, fdtype())

    def get_brownian_step(self):
        return brownian_step(self.diffusion_coefficients, self.time_step, rng = self.rng)

    def get_next_coordinates(self):
        """Advances one time step and returns the positions (num_particles, 3), m.

        The first call returns the initial positions unchanged. 
        """
        if self.count == 0:
            self.count += 1
            return self.current_coordinates

        if self.flags & DISABLE_VELOCITY:
            v = xp.array((0.,0.,0.), fdtype())
        else:
            v = self.get_current_velocity()

        self.current_coordinates = compute_next_coordinates(
            self.current_coordinates, v, self.diffusion_coefficients, self.channel_dimensions, self.time_step, self.flags)
        
        try:
            xp.eval(self.current_coordinates)
        except Exception as e:
            pass
        
        self.count += 1
        return self.current_coordinates

    def get_complex_mobility(self, mobility=None, z=None):
        """Complex mobility, m^2/(V s), at heights ``z`` (m; default: current particle heights)."""
        if mobility is None:
            mobility = self.mobility
        if z is None:
            z = self.current_coordinates[..., 2]
        return complex_mobility(mobility, z, self.channel_height, self.wall_mobility, self.lambda_)

    def get_diffusion_coefficients(self, viscosity = None, temperature = None):
        """Diffusion coefficients, m^2/s."""
        viscosity = self.viscosity if viscosity is None else viscosity
        temperature = self.temperature if temperature is None else temperature

        diffusion_coefficients = xp.zeros((len(self.diameter), 3), fdtype())
        diffusion_coefficients[:] = xp.array(
            diffusion_coefficient(self.diameter[:, None], viscosity, temperature), fdtype()
        )
        return diffusion_coefficients

    def get_current_coordinates(self):
        """Current positions, (num_particles, 3), m."""
        return self.current_coordinates

    def get_current_poiseuille_velocity(self):
        """Poiseuille velocity at the current positions, (num_particles, 3), m/s."""
        y = self.current_coordinates[..., 1]
        z = self.current_coordinates[..., 2]
        w = self.channel_width
        h = self.channel_height
        g = self.pressure_gradient
        mu = self.viscosity
        terms = self.terms
        return rectangular_poiseuille_velocity(y, z, w, h, g, mu, terms=terms)

    def get_current_velocity(self):
        """Deterministic velocity at the current positions and time, (num_particles, 3), m/s."""
        x = self.current_coordinates
        t = self.current_time
        dt = self.time_step
        v = self.sedimentation_velocity
        mp = self.mobility
        h = self.channel_height
        mw = self.wall_mobility
        l = self.lambda_
        e = self.electric_field
        f = self.ac_frequency
        p = self.ac_phase
        h = self.channel_height
        w = self.channel_width
        g = self.pressure_gradient
        mu = self.viscosity
        return particle_velocity(x=x, t=t, dt=dt, v=v, g = g, mu = mu, h=h, w=w, mp=mp, mw=mw, l=l, e=e, f=f, p=p, flags=self.flags, terms = self.terms)

    def iter_coordinates(self, count: int):
        """Yields the positions (num_particles, 3), m, for ``count`` consecutive time steps.

        The yielded array is updated in place on the next step; copy it to keep it.
        """
        for _ in range(count):
            yield self.get_next_coordinates()

# disable Numba - revert to default mode
disable_numba()