from mpi4py import MPI  # must always be imported first

import os
import sys
import time
from traceback import print_stack
from math import pi, fsum
from dataclasses import dataclass

import numpy as np
import h5py

from shenfun import (Array, Function,
                     FunctionSpace, TensorProductSpace, VectorSpace)

comm = MPI.COMM_WORLD


def dot(a, b):
    return np.einsum('i...,i...', a, b)


@dataclass
class Config:
    """A dataclass for keeping track of all the LES settings.

    """
    # Notional simulation grid size (3/2 dealising pads this by 1.5)
    N: int = 32

    # Physical kinematic viscosity
    nu: float = 0.000185

    # Spectral forcing energy injection rate
    epsilon: float = 0.103

    # Spectral forcing low-wavenumber cutoff
    kfLow: int = 0

    # Spectral forcing high-wavenumber cutoff
    kfHigh: int = 2

    # simulation unique identification string
    pid: str = os.path.basename(os.getcwd())

    # Output directory
    odir: str = os.getcwd()

    # Abritrary time of start of simulation
    t_init: float = 0.0

    # Initial dt to use
    dt_init: float = 0.0

    # Simulation time limit
    tlimit: float = 20.0

    # Simulation limit on number of time steps taken
    cycle_limit: int = 5000

    # Time between pdf and spectrum outputs
    dt_stat: float = np.inf

    # Time between full 3D volume restart outputs
    dt_rst: float = np.inf

    # Initial condition choices are 'random' or from 'file'
    init_cond: str = 'random'

    # Input directory if init_cond == 'file'
    idir: str = os.getcwd()

    # Filename (not including idir) if init_cond == 'file'
    init_file: str = 'les_init_cond.h5'

    # Initial TKE if init_cond == 'random'
    KE_init: float = 1.4

    # Initial E(k) slope if init_cond == 'random'
    kExp_init: float = 1/3

    # Advection CFL number
    cfl: float = 0.9

    # Diffusion Courant number
    Co_diff: float = 1.0

    # LES model choices are 'smag', 'dyn_smag', '4term', 'dns'
    model: str = '4term'

    # Smagorinsky constant
    Cs: float = 0.01

    # Generalized Eddy Viscosity constants
    C1: float = -0.03
    C2: float = 0.0
    C3: float = 0.0
    C4: float = 0.0

    # Test filter shape if model == 'dyn_smag'
    test_filter: str = 'gaussian'

    # Ratio of dynamic test filter width to dealiasing cutoff filter width
    test_ratio: int = 2


class MPI_Debugging:
    """A "Mixin" class to add simple debug methods to any class with an
    `MPI.Comm` instance stored as the attribute `self.comm`.

    """
    def disable_print(self):
        sys.stdout = open(os.devnull, 'w')

    def enable_print(self):
        sys.stdout = sys.__stdout__

    def redirect_print(self, filename):
        sys.stdout = open(filename, 'w')

    def print(self, string, flush=False, **kwargs):
        """Have rank 0 only call print()

        """
        if self.comm.rank == 0:
            print(string, flush=flush, **kwargs)

    def soft_abort(self, string, code=1):
        """All tasks exit program with exit code `code` after rank 0 prints
        `string` and a stack trace to `stdout`.

        """
        if self.comm.rank == 0:
            print(string, flush=True)
            print_stack()

        sys.exit(code)

    def all_assert(self, test, string=""):
        """Allreduce `test` and exit program via `soft_abort` if False.

        """
        test = self.comm.allreduce(test, op=MPI.LAND)
        if test is False:
            self.soft_abort(string)

    def soft_assert(self, test, string=""):
        """If ``test is False`` then exit program via `soft_abort`.

        .. warning:: `soft_assert` assumes every MPI task receives identical
        value for `test`. If not, this function seems to hang indefinitely
        for some reason I have yet to fathom, even if `test` evaluates to
        False on all processes! Needs further investigation.

        """
        if test is False:
            self.soft_abort(string)


class SpectralLES(MPI_Debugging):
    """
    A new spectralLES solver meant to be called by "outer-loop"
    algorithms for LES model parameter estimation.

    """
    def __init__(self, config, comm=comm):
        # -------------------------------------------------------------------
        # Configure the FFTs and allocate solution arrays
        # -------------------------------------------------------------------
        self.comm = comm
        self.config = config
        self.nu = config.nu
        N = [config.N]*3

        fft0 = FunctionSpace(N[0], padding_factor=1.5, dtype='c16')
        fft1 = FunctionSpace(N[1], padding_factor=1.5, dtype='c16')
        fft2 = FunctionSpace(N[2], padding_factor=1.5, dtype='f8')
        fft_params = dict(planner_effort='FFTW_MEASURE',
                          slab=False,
                          collapse_fourier=False,
                          )

        self.fft = TensorProductSpace(comm, (fft0, fft1, fft2), **fft_params)
        self.vfft = VectorSpace(self.fft)

        # nx == 1.5 * N when 3/2 dealiasing is used
        self.N = np.array(N, dtype='i8')
        self.nx = np.array(self.fft.global_shape(False))
        self.kmax = self.N // 2
        self.nk = np.array(self.fft.global_shape(True))

        """self.K is a list of three sparse arrays that are "broadcastable"
        in the numpy terminology and contain only the wavenumber values for
        the MPI subdomain of each MPI task."""
        self.K = self.fft.local_wavenumbers()
        self.Ksq = self.K[0]**2 + self.K[1]**2 + self.K[2]**2

        # This K is the dense broadcasting of self.K
        K = np.asarray(np.broadcast_arrays(*self.K))

        self.K_Ksq = K / np.where(self.Ksq == 0, 1, self.Ksq)
        self.iK = 1j * K
        self.wavemodes = np.sqrt(self.Ksq).astype('i4')
        self.num_wavemodes = int(sum(self.kmax**2)**0.5) + 1

        """Real-valued (MPI-local) buffer array `r` is the direct input to
        ``FFT.forward()`` and the direct output from ``FFT.backward()``.
        WARNING: FFT buffers are overwritten by some `shenfun` backends
        and are therefore potentially unsafe for use as data arrays."""
        self.r = self.fft.forward.input_array

        """Complex-valued (MPI-local) buffer array `c` is the direct output
        from ``FFT.forward()`` and the direct input to ``FFT.backward()``.
        WARNING: FFT buffers are overwritten by some `shenfun` backends
        and are therefore potentially unsafe for use as data arrays."""
        self.c = self.fft.forward.output_array

        self.u_hat = Function(self.vfft)  # fourier-space solution vector
        self.u = Array(self.vfft)         # physical-space solution vector
        self.w_hat = Function(self.vfft)  # fourier-space work vector
        self.w = Array(self.vfft)         # physical-space work vector

        # --------------------------------------------------------------
        # Configure the dealiasing filter
        # --------------------------------------------------------------
        self.k_dealias = self.kmax[0]  # 3/2 dealiasing
        self.Kdealias = ((np.abs(self.K[0]) < self.k_dealias)
                         * (np.abs(self.K[1]) < self.k_dealias)
                         * (np.abs(self.K[2]) < self.k_dealias))

        # --------------------------------------------------------------
        # Configure the turbulence model
        # --------------------------------------------------------------
        self.Delta2 = (pi / self.k_dealias)**2

        if config.model == 'dyn_smag':
            self.Cs = 0.01  # dummy value

            self.work = np.empty([5, *self.r.shape], self.r.dtype)
            self.nuT = self.work[4]  # just an alias

            if config.test_filter == "spectral":
                kf = self.k_dealias / config.test_ratio
                self.Ktest = ((np.abs(self.K[0]) < kf)
                              * (np.abs(self.K[1]) < kf)
                              * (np.abs(self.K[2]) < kf))

            elif config.test_filter == 'gaussian':
                delta = config.test_ratio * pi / self.k_dealias
                self.Ktest = self.gaussian_filter_kernel(delta)

            self.update_turbulence_model = self.update_dyn_smag_viscosity
            self.rhs_turbulence_model = self.rhs_smagorinsky

        elif config.model == 'smag':
            self.Cs = config.Cs  # Smagorinsky constant

            self.work = np.empty([2, *self.r.shape], self.r.dtype)
            self.nuT = self.work[1]  # just an alias

            self.update_turbulence_model = self.update_smag_viscosity
            self.rhs_turbulence_model = self.rhs_smagorinsky

        elif config.model == '4term':
            self.Cs = np.array((config.C1, config.C2, config.C3, config.C4))
            self.Cs *= -1 * self.Delta2

            self._mk = np.array([[0, 1, 2],
                                [1, 3, 4],
                                [2, 4, 5]])
            self._Tik = np.array([[+1, +1, +1],
                                 [-1, +1, +1],
                                 [-1, -1, +1]])
            self._Tkj = np.array([[+1, -1, -1],
                                 [+1, +1, -1],
                                 [+1, +1, +1]])

            self.Smat = np.empty([6, *self.r.shape], self.r.dtype)
            self.Rmat = np.empty([6, *self.r.shape], self.r.dtype)
            self.work = np.empty([2, *self.r.shape], self.r.dtype)
            self.nuT = self.work[1]  # just an alias

            self.update_turbulence_model = self.update_4term_viscosity
            self.rhs_turbulence_model = self.rhs_4term

        else:  # 'DNS', do nothing here.
            pass

        # --------------------------------------------------------------
        # Configure the spectral forcing
        # --------------------------------------------------------------
        k = np.arange(self.num_wavemodes)

        # form bandpass from 10th-order Linkwitz-Riley filters
        with np.errstate(divide='ignore', invalid='ignore'):
            self._bandpass = 1 - 1/(1 + (k/config.kfLow)**10)  # high pass
            self._bandpass *= 1/(1 + (k/config.kfHigh)**10)    # low pass

        self._bandpass[0] = 0.0                      # ensure zero mean
        self._bandpass[self.k_dealias:] = 0.0        # enforce dealiasing
        self._bandpass *= 1.0/self._bandpass.max()   # normalize to max of 1

        # -------------------------------------------------------------------
        # Initialize the velocity field
        # -------------------------------------------------------------------
        self._dx = 2 * pi / self.nx[0]
        self._dt_diff = 0.0
        self._dt_hydro = 0.0
        self.istep = 0

        if config.init_cond == 'random':
            energy = config.KE_init
            exponent = config.kExp_init
            rseed = None  # this let's numpy be random from run to run
            self.initialize_random_spectrum(energy, exponent, rseed)

        elif config.init_cond == 'file':
            filename = f'{config.idir}/{config.init_file}'
            self.initialize_from_file(filename)

        else:
            self.soft_abort('ERROR: incorrect initial condition')

        self.t_rst = config.t_init
        self.t_stat = config.t_init
        self.istat = 0
        self.frame = 0

        # --------------------------------------------------------------
        # Initialize turbulence model as necessary
        # --------------------------------------------------------------
        self.update_turbulence_model()

        # --------------------------------------------------------------
        # Initialize the spectral forcing
        # --------------------------------------------------------------
        Ek_actual = self.compute_energy_spectrum(self.u_hat)
        Ek_forcing = np.zeros_like(Ek_actual)
        with np.errstate(divide='ignore', invalid='ignore'):
            Ek_forcing = np.sqrt(self._bandpass / Ek_actual)
        idx = np.where(~np.isfinite(Ek_forcing))  # fixes an edge case concern
        Ek_forcing[idx] = 0.0

        self.w_hat[:] = self.u_hat * Ek_forcing[self.wavemodes]
        self.vfft.backward(self.w_hat, self.w)
        self._fscale = config.epsilon * self.nx.prod()
        self._fscale *= 1/self.allsum(self.u * self.w)

        comm.Barrier()

        # --------------------------------------------------------------
        # Create statistics file and output initial statistics
        # --------------------------------------------------------------
        self.fh_stat = None
        self.stat_file = f"{config.odir}/{config.pid}.statistics.h5"
        MiB = 1048576  # 1 mibibyte (Lustre stripe size)

        if comm.rank == 0:
            os.makedirs(f'{config.odir}', exist_ok=True)
            self.fh_stat = h5py.File(self.stat_file, 'w', driver='core',
                                     block_size=MiB)

        self.save_statistics_to_file(config.t_init)

        self.t_rst += config.dt_rst
        self.t_stat += config.dt_stat

        self.disable_print()  # disable screen outputs by default

        return

    def __del__(self):
        if self.comm.rank == 0:
            self.fh_stat.close()

        self.enable_print()

        return

    ###########################################################################
    # General Methods
    ###########################################################################
    def allsum(self, data):
        return self.comm.allreduce(fsum(data.flat), op=MPI.SUM)

    def allmin(self, data):
        fill = np.ma.minimum_fill_value(data)
        return self.comm.allreduce(np.amin(data, initial=fill), op=MPI.MIN)

    def allmax(self, data):
        fill = np.ma.maximum_fill_value(data)
        return self.comm.allreduce(np.amax(data, initial=fill), op=MPI.MAX)

    def div(self, u_hat, out=None):
        if out is None:
            out = np.empty_like(self.r)

        out[:] = self.fft.backward(dot(self.iK, u_hat))

        return out

    def curl(self, u_hat, out=None):
        if out is None:
            out = np.empty([3, *self.r.shape], dtype=self.r.dtype)

        out[0] = self.fft.backward(self.iK[1]*u_hat[2] - self.iK[2]*u_hat[1])
        out[1] = self.fft.backward(self.iK[2]*u_hat[0] - self.iK[0]*u_hat[2])
        out[2] = self.fft.backward(self.iK[0]*u_hat[1] - self.iK[1]*u_hat[0])

        return out

    def strain_squared(self, u_hat, out=None):
        if out is None:
            out = np.empty_like(self.r)

        out[:] = 0.0
        for i in range(3):
            for j in range(i, 3):
                self.fft.backward(self.iK[j]*u_hat[i] + self.iK[i]*u_hat[j])
                # output of FFT is in self.r
                out += (1 + (i != j)) * self.r**2
        out *= 0.5  # leaves 2 * Sij * Sij

        return out

    def compute_energy_spectrum(self, u_hat):
        """Compute the discrete 1D energy spectrum of the input.

        Parameters
        ----------
        u_hat : array_like
            Input array, must be a scalar or vector field defined on the
            Fourier domain `self.fft`.

        Returns
        -------
        E1d : 1-dimensional `ndarray`
            The discrete 1D energy spectrum of `u_hat`
        """
        if u_hat.ndim == 4:
            E3d = np.sum(np.real(u_hat * np.conj(u_hat)), axis=0)
        else:
            E3d = np.real(u_hat * np.conj(u_hat))

        K2 = self.K[2].reshape(-1)
        E3d[:, :, K2 == 0] *= 0.5
        E3d[:, :, K2 == self.kmax[2]] *= 0.5

        E1d = np.zeros(self.num_wavemodes, dtype=E3d.dtype)
        modes = self.wavemodes
        for k in range(self.num_wavemodes):
            E1d[k] = fsum(E3d[modes == k].flat)

        self.comm.Allreduce(MPI.IN_PLACE, E1d, op=MPI.SUM)

        return E1d

    def filter(self, x, kernel):
        """Filter the physical-space array x in-place using the given
        spectral-space kernel.
        """
        x_hat = self.fft.forward(x)  # x --> self.c
        x_hat *= kernel
        x[:] = self.fft.backward(x_hat)  # self.c --> x

        return x

    def gaussian_filter_kernel(self, delta, C=6):
        """Provides Gaussian filter kernel computed pointwise directly from the
        spectral domain analytical formula.
        """
        C = 1 / (4 * C)
        return np.exp(-C * delta**2 * self.Ksq)

    ###########################################################################
    # Initialization Methods
    ###########################################################################
    def initialize_random_spectrum(self, energy, exponent, rseed=None):
        """Generate a random velocity field with a prescribed isotropic spectrum.
        """
        u_hat = self.u_hat
        u = self.u

        self.all_assert(isinstance(rseed, int) or rseed is None)
        rng = np.random.default_rng(rseed)

        # --------------------------------------------------------------
        # Give all wavenumbers a random uniform magnitude and phase
        u_hat[:] = rng.random(u_hat.shape, 'f8')
        theta = rng.random(u_hat.shape, 'f8')
        u_hat *= np.cos(2 * pi * theta) + 1j*np.sin(2 * pi * theta)

        # --------------------------------------------------------------
        # Parseval's Theorem Fix:
        #  Correct those wavenumbers that should be real-valued/Hermitian using
        #  a round-trip FFT. Not efficient, but this is one-time-only.
        #  Q: Also possibly alters the uniform distribution?
        #  A: Meh, seems to work.
        self.vfft.backward(u_hat, u)
        self.vfft.forward(u, u_hat)

        # --------------------------------------------------------------
        # Solenoidally-project before rescaling
        u_hat += 1j * self.iK * dot(u_hat, self.K_Ksq)

        # --------------------------------------------------------------
        # Re-scale to a simple energy spectrum
        nk = self.num_wavemodes
        kd = self.k_dealias
        delta = pi / self.kmax[0]
        k = np.arange(nk)
        k[0] = 1

        Ek_target = k**exponent * np.exp(-(1/6) * (4 * delta * k)**2)
        Ek_target[0] = 0.0
        Ek_target[kd:] = 0.0
        Ek_target *= energy / np.sum(Ek_target)

        Ek_actual = self.compute_energy_spectrum(u_hat)
        self.soft_assert(np.all(Ek_actual > 1e-20))

        with np.errstate(divide='ignore', invalid='ignore'):
            scaling = np.sqrt(Ek_target / Ek_actual)
        scaling[0] = 0.0     # remove the spatial mean
        scaling[kd:] = 0.0   # remove dealiased modes

        u_hat *= scaling[self.wavemodes]
        self.vfft.backward(u_hat, u)

        return

    def initialize_from_file(self, filename, restart=False):
        fh = h5FileIO(filename, 'r', self.comm)
        Np = fh["U_hat0"].shape[0]
        self.all_assert(Np == self.N[0])

        for i in range(3):
            fh.read(f'U_hat{i}', self.u_hat[i])

        self.u_hat *= self.Kdealias
        self.vfft.backward(self.u_hat, self.u)

        return

    ###########################################################################
    # Time Integration Methods
    ###########################################################################
    def run_verbose(self):
        # --------------------------------------------------------------
        # Setup
        # --------------------------------------------------------------
        self.enable_print()

        start = time.strftime('%H:%M:%S')
        self.print(
            "\n------------------------------------------------------\n"
            "MPI-parallel Python LES simulation of HIT \n"
            f"started with {self.comm.size} tasks at {start}.\n", True)

        lines = ['\n Simulation Configuration # -------------------------']
        for k, v in vars(self.config).items():
            lines.append(f"{k:15s} = {v}")
        self.print('\n'.join(lines), True)

        self.comm.Barrier()
        wt1 = time.perf_counter()

        success = self.run_quiet()

        self.comm.Barrier()
        wt2 = time.perf_counter()

        minutes, seconds = divmod(int(wt2 - wt1), 60)
        hours, minutes = divmod(minutes, 60)
        tot_sec = wt2 - wt1
        dof_cycles = self.N.prod() * self.istep / tot_sec
        cell_cycles = self.nx.prod() * self.istep / tot_sec

        self.print(
            "----------------------------------------------------------\n"
            f"Cycles completed = {self.istep:4d}\n"
            f"Total Runtime (H:M:S) = {hours}:{minutes:02d}:{seconds:02d}\n"
            f"Total cycles per second = {self.istep / tot_sec:0.6g}\n"
            f"-- DOF cycles per second = {dof_cycles:0.6e}\n"
            f"-- cell cycles per second = {cell_cycles:0.6e}\n\n"
            f"Simulation finished at {time.strftime('%H:%M:%S')}.\n"
            "----------------------------------------------------------\n")

        return success

    def run_quiet(self):
        # --------------------------------------------------------------
        # Setup
        # --------------------------------------------------------------
        config = self.config

        a = [0.5, 0.5, 1.0]
        b = [1./6., 1./3., 1./3., 1./6.]

        t_sim = config.t_init
        tlimit = config.tlimit
        dt = max(1e-6, config.dt_init)

        u_hat0 = np.empty_like(self.u_hat)  # previous solution register
        u_hat1 = np.empty_like(self.u_hat)  # full-step accumulation register
        du = np.zeros_like(self.u_hat)      # RHS evaluation register

        # --------------------------------------------------------------
        # Time integration loop
        # --------------------------------------------------------------
        try:
            while t_sim < tlimit:
                if tlimit - t_sim < dt:
                    dt = tlimit - t_sim

                u_hat0[:] = u_hat1[:] = self.u_hat

                # Stages 1 to 3
                for rk in range(3):
                    self.compute_rhs(du)
                    u_hat1 += b[rk] * dt * du
                    self.u_hat[:] = u_hat0 + a[rk] * dt * du
                    self.vfft.backward(self.u_hat, self.u)

                # Stage 4
                self.compute_rhs(du)
                u_hat1 += b[3] * dt * du
                self.u_hat[:] = u_hat1[:]
                self.vfft.backward(self.u_hat, self.u)

                # Update t_sim
                t_sim += dt

                # Update dt, output statistics, etc.
                dt = self.step_update(du, t_sim, dt)

        except SystemExit:
            self.enable_print()
            self.print(f"SpectralLES Warning: Simulation {config.pid=} "
                       "aborted before reaching tlimit!",
                       flush=True)
            success = False

        except Exception as err:
            self.enable_print()
            self.print(f"SpectralLES Warning: Unexpected {err=}\n"
                       f"{config.pid=} interrupted before reaching tlimit!",
                       flush=True)
            success = False

        else:
            success = True

        # --------------------------------------------------------------
        # Finish (executes in all cases, even unknown Exception)
        # --------------------------------------------------------------
        self.finalize(t_sim)

        return success

    def compute_rhs(self, du):
        """Incompressible Navier-Stokes advection in rotation form with
        possible turbulence model and spectral forcing.
        """
        iK = self.iK
        u_hat = self.u_hat
        u = self.u
        w = self.w

        # take curl of velocity and inverse transform to get vorticity
        w[0] = self.fft.backward(iK[1] * u_hat[2] - iK[2] * u_hat[1])
        w[1] = self.fft.backward(iK[2] * u_hat[0] - iK[0] * u_hat[2])
        w[2] = self.fft.backward(iK[0] * u_hat[1] - iK[1] * u_hat[0])

        # take cross-product of vorticity and velocity and transform back
        # NOTE: u x w = - w x u (Since Du/Dt = du/dt + w x u)
        du[0] = self.fft.forward(u[1] * w[2] - u[2] * w[1])
        du[1] = self.fft.forward(u[2] * w[0] - u[0] * w[2])
        du[2] = self.fft.forward(u[0] * w[1] - u[1] * w[0])

        # add on turbulence stress
        self.rhs_turbulence_model(du)

        # dealias everything that was nonlinear
        du *= self.Kdealias

        # project off the normal stresses as the pressure poisson solution
        du += 1j * self.iK * dot(du, self.K_Ksq)

        # add linear viscous diffusion
        du -= self.nu * self.Ksq * u_hat

        # add spectral forcing
        Ek_actual = self.compute_energy_spectrum(u_hat)
        with np.errstate(divide='ignore', invalid='ignore'):
            Ek_forcing = self._fscale * np.sqrt(self._bandpass / Ek_actual)
        idx = np.where(~np.isfinite(Ek_forcing))  # fixes an edge case concern
        Ek_forcing[idx] = 0.0
        du += Ek_forcing[self.wavemodes] * u_hat

        return

    def rhs_smagorinsky(self, du):
        """18 total scalar FFTs in function.
        """
        iK = self.iK
        Cs = self.Cs
        u_hat = self.u_hat

        # S is an alias to work[0]
        S = self.strain_squared(u_hat, out=self.work[0])
        S = np.sqrt(S)

        for i in range(3):
            for j in range(i, 3):
                Sij = self.fft.backward(iK[j] * u_hat[i] + iK[i] * u_hat[j])
                Rij_hat = self.fft.forward(Cs * self.Delta2 * S * Sij)

                du[i] += iK[j] * Rij_hat
                du[j] += (i != j) * iK[i] * Rij_hat

        return du

    def rhs_4term(self, du):
        """Remarkably this function only has 18 FFTs, but requires lots of
        extra memory. See `update_4term_viscosity` for extra notes, as it
        performs all the same calculations to get an effective eddy viscosity.
        """
        iK = self.iK
        u_hat = self.u_hat
        C = self.Cs
        Smat = self.Smat
        Rmat = self.Rmat
        Ssq = self.w[0]
        Rsq = self.w[1]
        tau_ij = self.w[2]

        Ssq[:] = 0.0
        Rsq[:] = 0.0
        m = 0
        for i in range(3):
            for j in range(i, 3):
                Smat[m] = self.fft.backward(iK[j]*u_hat[i] + iK[i]*u_hat[j])
                Ssq += (1 + (i != j)) * Smat[m]**2

                Rmat[m] = self.fft.backward(iK[j]*u_hat[i] - iK[i]*u_hat[j])
                Rsq += (1 + (i != j)) * Rmat[m]**2

                m += 1
        Smat *= 1/2
        Rmat *= 1/2
        Ssq *= 1/4
        Rsq *= 1/4

        m = 0
        for i in range(3):
            for j in range(i, 3):
                ik = self._mk[i]
                kj = self._mk[j]
                tik = self._Tik[i].reshape(3, 1, 1, 1)
                tkj = self._Tkj[j].reshape(3, 1, 1, 1)

                # G_ij^(1) = |S| S_ij, note inclusion of 2 here
                tau_ij[:] = C[0] * np.sqrt(2*Ssq) * Smat[m]

                for k in range(3):
                    # G_ij^(2) = S_ik R_kj - R_ik S_kj
                    tau_ij += C[1] * 0.5 * (dot(Smat[ik], tkj*Rmat[kj])
                                            - dot(tik*Rmat[ik], Smat[kj]))

                    # G_ij^(3) = S_ik S_kj
                    tau_ij += C[2] * dot(Smat[ik], Smat[kj])

                    # G_ij^(4) = R_ik R_kj
                    tau_ij += C[3] * dot(tik * Rmat[ik], tkj * Rmat[kj])

                # - 1/3 delta_ij (C2*SijSij + C3*RijRij), note lack of 2 here
                tau_ij -= (1/3)*(i == j) * (C[2]*Ssq + C[3]*Rsq)

                tau_hat = self.fft.forward(tau_ij)
                du[i] += iK[j] * tau_hat
                du[j] += (i != j) * iK[i] * tau_hat

                m += 1

        return du

    ###########################################################################
    # Inter-step Update Methods
    ###########################################################################
    def step_update(self, du, t_sim, dt):
        """Things to do at the end of each complete time step
        """
        config = self.config
        dx = self._dx
        u_hat = self.u_hat
        u = self.u
        w_hat = self.w_hat
        w = self.w

        # --------------------------------------------------------------
        # Update turbulence model as necessary for new dt and outputs
        # --------------------------------------------------------------
        self.update_turbulence_model()

        # --------------------------------------------------------------
        # Update spectral forcing
        # --------------------------------------------------------------
        Ek_actual = self.compute_energy_spectrum(u_hat)
        Ek_forcing = np.zeros_like(Ek_actual)
        with np.errstate(divide='ignore', invalid='ignore'):
            Ek_forcing = np.sqrt(self._bandpass / Ek_actual)
        idx = np.where(~np.isfinite(Ek_forcing))  # fixes an edge case concern
        Ek_forcing[idx] = 0.0

        w_hat[:] = u_hat * Ek_forcing[self.wavemodes]
        self.vfft.backward(w_hat, w)
        self._fscale = config.epsilon * self.nx.prod()
        self._fscale *= 1/self.allsum(u * w)

        # --------------------------------------------------------------
        # Compute new dt
        # --------------------------------------------------------------
        buffer = np.empty(2)
        buffer[0] = np.max(np.sum(np.abs(self.u), axis=0))
        buffer[1] = np.max(self.nuT)
        self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)

        u_max = max(1e-20, buffer[0])
        nuT_max = max(1e-20, buffer[1])

        self._dt_hydro = config.cfl * dx / u_max
        self._dt_diff = config.Co_diff * dx**2 / (6 * (self.nu + nuT_max))

        new_dt = min(2.0 * dt, self._dt_hydro, self._dt_diff)

        # --------------------------------------------------------------
        # Generate Outputs
        # --------------------------------------------------------------
        self.istep += 1
        t_next = t_sim + new_dt

        if sys.stdout.name != '/dev/null':
            self.w[0] = dot(self.u, self.u)
            TKE = 0.5 * self.allsum(self.w[0]) / self.nx.prod()
            self.print(f"cycle = {self.istep:4d}, time = {t_sim:< 14.6e}, "
                       f"dt = {dt:< 14.6e}, TKE = {TKE:< 14.6e}",
                       flush=(self.istep % 25 == 0))

        if t_sim + 1e-14 >= self.t_stat:
            self.t_stat += max(config.dt_stat, t_next - self.t_stat)
            self.save_statistics_to_file(t_sim)

        rst_output = False
        if t_sim + 1e-14 >= self.t_rst:
            config.dt_init = new_dt  # this modifies the Config object in place
            self.t_rst += max(config.dt_rst, t_next - self.t_rst)
            self.write_restart_to_file()
            rst_output = True

        if rst_output or (self.istep % 200 == 0):
            if self.comm.rank == 0:
                self.fh_stat.flush()

        if self.istep > config.cycle_limit:
            self.finalize(t_sim)
            self.soft_abort("Error: hit cycle limit!")

        return new_dt

    def update_smag_viscosity(self):
        S = self.strain_squared(self.u_hat, out=self.work[0])
        S = np.sqrt(S)
        self.nuT[:] = self.Cs * self.Delta2 * S

        return

    def update_dyn_smag_viscosity(self):
        """51 total scalar FFTs in function.
        """
        n2 = self.config.test_ratio**2

        # pointing working memory to readable names
        S_hat = self.w_hat[0]
        S = self.work[0]
        St = self.work[1]
        Mij = self.work[2]
        LijMij = self.work[3]
        MijMij = self.work[4]

        self.w_hat[:] = self.Ktest * self.u_hat
        self.vfft.backward(self.w_hat, self.w)

        S[:] = 0.0
        St[:] = 0.0
        for i in range(3):
            for j in range(i, 3):
                S_hat[:] = self.iK[j]*self.u_hat[i] + self.iK[i]*self.u_hat[j]
                Sij = self.fft.backward(S_hat)  # == 2Sij
                S += (1 + (i != j)) * Sij**2  # == 4SijSij

                S_hat *= self.Ktest
                Sij = self.fft.backward(S_hat)  # == 2Sij
                St += (1 + (i != j)) * Sij**2  # == 4SijSij

        S[:] = np.sqrt(0.5 * S)    # == sqrt(2 Sij Sij)
        St[:] = np.sqrt(0.5 * St)  # == sqrt(2 Sij Sij)

        LijMij[:] = 0.0
        MijMij[:] = 0.0
        for i in range(3):
            for j in range(i, 3):
                S_hat[:] = self.iK[j]*self.u_hat[i] + self.iK[i]*self.u_hat[j]
                Mij[:] = S * self.fft.backward(S_hat)
                Mij[:] = self.filter(Mij, self.Ktest)
                Mij[:] -= n2 * St * self.fft.backward(self.Ktest * S_hat)

                Lij = self.filter(self.u[i] * self.u[j], self.Ktest)
                Lij -= self.w[i] * self.w[j]

                LijMij += (1 + (i != j)) * Lij * Mij  # note sign
                MijMij += (1 + (i != j)) * Mij**2

        buffer = np.empty(2)
        buffer[0] = fsum(LijMij.flat)  # 0.5 needed for Sij
        buffer[1] = fsum(MijMij.flat)
        self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

        self.Cs = buffer[0] / (self.Delta2 * buffer[1])
        self.nuT[:] = self.Cs * self.Delta2 * S

        return

    def update_4term_viscosity(self):
        """This function has 12 FFTs.

        For the 4-term nonlinear model, we set nuT == tau_ij Sij / Skl Skl for
        both compute_dt (Courant limit) and statistical outputs of Pi and nuT.

        SPARSE SYMMETRIC TENSOR INDEXING:
        m == 0 -> ij == 00
        m == 1 -> ij == 01
        m == 2 -> ij == 02
        m == 3 -> ij == 11
        m == 4 -> ij == 12
        m == 5 -> ij == 22

        ik indexing (T means transpose):
        ------------
        0k -> m = [0,  1,  2]
        1k -> m = [1T, 3,  4]
        2k -> m = [2T, 4T, 5]

        kj indexing:
        ------------
        k0 -> m = [0,  1T, 2T]
        k1 -> m = [1,  3,  4T]
        k2 -> m = [2,  4,  5]

        """
        iK = self.iK
        u_hat = self.u_hat
        C = self.Cs
        Pi = self.nuT  # re-using nuT memory for Pi

        # These are the sparse tensor work arrays for S_ij and R_ij
        Smat = self.Smat
        Rmat = self.Rmat

        # more work arrays
        Ssq = self.w[0]
        Rsq = self.w[1]
        tau_ij = self.w[2]

        # m indices to create vectors X_ik or X_kj (X = S or R)
        mk = np.array([[0, 1, 2],
                       [1, 3, 4],
                       [2, 4, 5]])

        # transpose signs to create vector Rik = - Rki (S is symmetric)
        # (+1 gives no transpose, -1 gives the transpose term as required)
        Tik = np.array([[+1, +1, +1],
                        [-1, +1, +1],
                        [-1, -1, +1]])

        # transpose sign to create vector Rkj = - Rjk (S is symmetric)
        Tkj = np.array([[+1, -1, -1],
                        [+1, +1, -1],
                        [+1, +1, +1]])

        Ssq[:] = 0.0
        Rsq[:] = 0.0
        m = 0
        for i in range(3):
            for j in range(i, 3):
                Smat[m] = self.fft.backward(iK[j]*u_hat[i] + iK[i]*u_hat[j])
                Ssq += (1 + (i != j)) * Smat[m]**2

                Rmat[m] = self.fft.backward(iK[j]*u_hat[i] - iK[i]*u_hat[j])
                Rsq += (1 + (i != j)) * Rmat[m]**2

                m += 1
        Smat *= 1/2
        Rmat *= 1/2
        Ssq *= 1/4
        Rsq *= 1/4

        # --------------------------------------------------------------
        # Compute Pi = tau_ij * S_ij
        Pi[:] = 0.0
        m = 0
        for i in range(3):
            for j in range(i, 3):
                ik = mk[i]  # This is a shape (3,) array for advanced indexing
                kj = mk[j]  # This is a shape (3,) array for advanced indexing
                tik = Tik[i].reshape(3, 1, 1, 1)
                tkj = Tkj[j].reshape(3, 1, 1, 1)

                # G_ij^(1) = |S| S_ij, note inclusion of 2 here
                tau_ij[:] = C[0] * np.sqrt(2*Ssq) * Smat[m]

                for k in range(3):
                    # G_ij^(2) = S_ik R_kj - R_ik S_kj
                    tau_ij += C[1] * 0.5 * (dot(Smat[ik], tkj*Rmat[kj])
                                            - dot(tik*Rmat[ik], Smat[kj]))

                    # G_ij^(3) = S_ik S_kj
                    tau_ij += C[2] * dot(Smat[ik], Smat[kj])

                    # G_ij^(4) = R_ik R_kj
                    tau_ij += C[3] * dot(tik * Rmat[ik], tkj * Rmat[kj])

                # - 1/3 delta_ij (C2*SijSij + C3*RijRij), note lack of 2 here
                tau_ij -= (1/3)*(i == j) * (C[2]*Ssq + C[3]*Rsq)

                # Pi = tau_ij * Sij
                Pi += (1 + (i != j)) * tau_ij * Smat[m]

                m += 1

        # Compute nuT = Pi / Ssq (same memory space as Pi)
        self.nuT /= Ssq

        return

    ###########################################################################
    # Output Methods
    ###########################################################################
    def finalize(self, t_sim):
        self.save_statistics_to_file(t_sim)
        self.write_restart_to_file(checkpoint=True)

        if self.comm.rank == 0:
            self.fh_stat.flush()
            print(' *** satistics flushed to disk '
                  f'({self.istat} entries)', flush=True)
            self.fh_stat.close()

        self.enable_print()

        return

    def save_statistics_to_file(self, t_sim):
        iK = self.iK
        u_hat = self.u_hat
        work = self.work[0]

        grp = None
        if self.comm.rank == 0:
            grp = self.fh_stat.create_group(f"{self.istat:03d}")
            grp.attrs['t_sim'] = t_sim

        # save the energy spectrum to the statistics file
        Ek1d = self.compute_energy_spectrum(u_hat)
        if self.comm.rank == 0:
            grp['Ek'] = Ek1d

        work[:] = self.strain_squared(u_hat, out=work) + 1e-99
        work *= self.nuT  # convert Ssq to Pi
        if self.config.model == '4term':
            # because Ssq = 2SijSij here, but not in 4term model!
            work *= 0.5
        self.save_histogram_to_file(grp, work, 'Pi')

        # turbulent stress tensor statistics
        if self.config.model in ['smag', 'dyn_smag']:
            self.w_hat[:] = 0.0
            for j in range(3):
                Sij = self.fft.backward(iK[j]*u_hat[0] + iK[0]*u_hat[j])
                self.w[0] = self.nuT * Sij
                self.save_histogram_to_file(grp, self.w[0], f'sigma_1{j+1}')

        else:  # 4term model
            # When this function is called from `step_update` or `finalize`,
            # the following things are true:
            # - Smat and Rmat will already be computed and correct
            # - Ssq and Rsq (aka w[0] and w[1]) will have been overwritten
            C = self.Cs
            Smat = self.Smat
            Rmat = self.Rmat
            Ssq = self.w[0]
            Rsq = self.w[1]
            tau_ij = self.w[2]

            Ssq[:] = 0.0
            Rsq[:] = 0.0
            m = 0
            for i in range(3):
                for j in range(i, 3):
                    Ssq += (1 + (i != j)) * Smat[m]**2
                    Rsq += (1 + (i != j)) * Rmat[m]**2
                    m += 1

            i = 0  # i = 0 corresponds to tau_11, tau_12, tau_13
            for j in range(3):
                ik = self._mk[i]
                kj = self._mk[j]
                tik = self._Tik[i].reshape(3, 1, 1, 1)
                tkj = self._Tkj[j].reshape(3, 1, 1, 1)

                tau_ij[:] = C[0] * np.sqrt(2*Ssq) * Smat[j]
                tau_ij -= (1/3)*(i == j) * (C[2]*Ssq + C[3]*Rsq)
                for k in range(3):
                    tau_ij += C[1] * 0.5 * dot(Smat[ik], tkj*Rmat[kj])
                    tau_ij -= C[1] * 0.5 * dot(tik*Rmat[ik], Smat[kj])
                    tau_ij += C[2] * dot(Smat[ik], Smat[kj])
                    tau_ij += C[3] * dot(tik * Rmat[ik], tkj * Rmat[kj])

                self.save_histogram_to_file(grp, tau_ij, f'sigma_1{j+1}')

        self.istat += 1

        return

    def save_histogram_to_file(self, fh, data, key):
        # get view of data as contiguous 1D numpy array
        data = np.ravel(data, order='K')

        # get data range
        gmin = self.allmin(data)
        gmax = self.allmax(data)
        xrange = (gmin, gmax)

        # construct simple logarithmic bin edges for strictly positive data
        if gmin > 0.0:
            edges = np.geomspace(gmin, gmax, 101)

        # otherwise construct "symlog" bin edges for data with negative
        # values in order to vastly improve kernel density estimation
        # in post-processing
        else:
            medians = self.comm.allgather(np.median(np.abs(data)))
            abs_med = np.median(medians)
            abs_max = max(abs(gmin), gmax)
            lin_edges = np.linspace(-abs_med, abs_med, 51)
            log_edges = np.geomspace(abs_med, abs_max, 26)
            # this gives 101 edges spanning -abs_max to abs_max with half
            # of all bins spanning -abs_med to abs_med
            edges = np.unique(np.r_[-log_edges, lin_edges, log_edges])

        temp, _ = np.histogram(data, bins=edges, density=True)
        hist = np.ascontiguousarray(temp)

        if self.comm.rank == 0:
            self.comm.Reduce(MPI.IN_PLACE, hist, op=MPI.SUM)
        else:
            self.comm.Reduce(hist, None, op=MPI.SUM)

        if self.comm.rank == 0:
            grp = fh.create_group(key)
            grp['hist'] = hist
            grp['edges'] = edges
            grp.attrs['range'] = xrange

        self.comm.Barrier()

        return

    def write_restart_to_file(self, checkpoint=False):
        config = self.config

        if checkpoint:
            filename = f"{config.odir}/{config.pid}.checkpoint.h5"
        else:
            filename = f"{config.odir}/{config.pid}.{self.frame:03d}.h5"

        with h5FileIO(filename, 'w', self.comm) as fh:
            fh.write("U_hat", self.u_hat, self.vfft, kwargs=vars(config))

        self.frame += 1

        return


class h5FileIO(h5py.File, MPI_Debugging):
    """Class for reading/writing a single snapshot of 3D data to the parallel
    HDF5 format. This is a much simplified form of the original
    :class:`HDF5File` provided by the :module:`shenfun` package.

    """
    def __init__(self, filename, mode='r', comm=comm, **h5_kw):
        super().__init__(filename, mode, driver="mpio", comm=comm, **h5_kw)
        self.comm = comm

        return

    def load_config(self):
        """Convert file attrs into a Config instance

        """
        config = dict(self.attrs.items())
        for key, val in config:
            if val is np.bool_(1):    # if val is "True" as saved by h5py
                config[key] = True

            elif val is np.bool_(0):  # if val is "False" as saved by h5py
                config[key] = False

            elif not np.issscalar(val):
                del config[key]

        return Config(**config)

    def read(self, name, U, T=None):
        """
        Read ``U``'s `local_slice()` from HDF5 file. `U` must either be a
        :class:`shenfun.Array` or :class:`shenfun.Function` object or a
         :class:`shenfun.TensorProductSpace` must be provided.
        Parameters
        ----------
        name: str
            Base string for tensor field `U`. For example, if `U` is rank 1,
            then scalar fields will be read in from datasets "name0", "name1",
            etc.
        U: :class:`numpy.ndarray`-like or :class:`shenfun.Array`-like
            The data field to be read from the named dataset.
        T: :class:`shenfun.TensorProductSpace`, optional
            The `shenfun` basis space that defines `U`'s global shape and
            local slice of the global array. Must be same Tensor-rank as `U`.

        """
        rank = len(U.shape) - 3
        if hasattr(U, 'local_slice'):
            local_slice = U.local_slice()[-3:]

        else:  # CompositeSpaces are tricksy hobbits, FYI
            Tshape = T.shape(False)[-3:]
            Ushape = U.shape[-3:]
            fwd_out = not Tshape == Ushape
            local_slice = T.local_slice(fwd_out)[-3:]

        if rank == 0:
            dset = self[name]
            with dset.collective:
                U[:] = dset[local_slice]

        else:  # if rank == 1:
            for i in range(U.shape[0]):
                dset = self[f"{name}{i}"]
                with dset.collective:
                    U[i] = dset[local_slice]

        return

    def write(self, name, U, T=None, kwargs={}):
        """Write ``U`` to HDF5 file.

        Parameters
        ----------
        name: str
            Base string for tensor field `U`. For example, if `U` is rank 1,
            then scalar fields will be stored as datasets "name0", "name1",
            etc.
        U: :class:`numpy.ndarray`-like or :class:`shenfun.Array`-like
            The data field to be stored as the named dataset.
        T: :class:`shenfun.TensorProductSpace`, optional
            The `shenfun` basis space that defines `U`'s global shape and
            local slice of the global array. Must be same Tensor-rank as `U`.

        """
        rank = len(U.shape) - 3
        if hasattr(U, 'local_slice'):
            local_slice = U.local_slice()[-3:]
            global_shape = U.global_shape[-3:]  # NOTE the lack of () here.

        else:  # CompositeSpaces are tricksy hobbits, FYI
            Tshape = T.shape(False)[-3:]
            Ushape = U.shape[-3:]
            fwd_out = not Tshape == Ushape
            local_slice = T.local_slice(fwd_out)[-3:]
            global_shape = T.global_shape(fwd_out)[-3:]

        if rank == 0:
            dset = self.require_dataset(
                name, shape=global_shape, dtype=U.dtype)
            with dset.collective:
                dset[local_slice] = U

        else:  # if rank == 1:
            for i in range(U.shape[0]):
                dset = self.require_dataset(
                    f"{name}{i}", shape=global_shape, dtype=U.dtype)
                with dset.collective:
                    dset[local_slice] = U[i]

        for k, v in kwargs.items():
            if v is not None:
                self.attrs[k] = v

            else:
                self.print('SpectralLES:h5FileIO:write() '
                           f'FIXME: found a None keyword value! key={k}')

        return
