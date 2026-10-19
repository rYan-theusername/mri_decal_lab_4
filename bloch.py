import numpy as np

# hydrogen gyromagnetic ratio in rad / (G·s)
GAMMA = 2 * np.pi * 4257.7


def _validate_vector(value, name, nonzero=False):
    array = np.asarray(value, dtype=float)

    if array.ndim != 1 or array.shape != (3,):
        raise ValueError(f"{name} must be a 3-component vector")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    if nonzero and np.linalg.norm(array) == 0:
        raise ValueError(f"{name} must be a non-zero vector")

    return array.copy()


def _validate_scalar(value, name, positive=False):
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric scalar") from exc

    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite number")

    if positive and scalar <= 0:
        raise ValueError(f"{name} must be positive")

    return scalar


def _relax(M, M0, T1, T2, dt):
    """ applies one T1/T2 relaxation step using
        the exact Bloch relaxation solution. """

    E1 = np.exp(-dt / T1)
    M[2] = M[2] * E1 + M0 * (1.0 - E1)

    E2 = np.exp(-dt / T2)
    M[0] *= E2
    M[1] *= E2


def _effective_B(B, off_resonance_hz):
    """ return the effective B. considers the effect
        of off-resonance as a z component """

    delta_Bz = -2 * np.pi * off_resonance_hz / GAMMA
    return B + np.array([0.0, 0.0, delta_Bz])


def _precess(M, B_eff, dt):
    """ rotate magnetization clockwise about B_eff
        over time dt via Rodrigues' formula. """

    B_mag = np.linalg.norm(B_eff)
    theta = -GAMMA * B_mag * dt

    if B_mag == 0:
        return

    axis = B_eff / B_mag
    M[:] = (
        M * np.cos(theta)
        + np.cross(axis, M) * np.sin(theta)
        + axis * np.dot(axis, M) * (1.0 - np.cos(theta))
    )


class Spin:
    """ a single spin with fixed position and evolving magnetization.

        has position, magnetization, and parameters
        M0, T1, T2, and off_resonance (default=0)

        units: position in cm. magnetic field in Gauss.
               time in seconds. off_resonance in Hz.
               gradient G in G/cm. """

    def __init__(self, position, magnetization, M0, T1, T2, off_resonance=0.0):
        self.position = _validate_vector(position, "position")
        self.magnetization = _validate_vector(magnetization, "magnetization")
        self.M0 = _validate_scalar(M0, "M0", positive=True)
        self.T1 = _validate_scalar(T1, "T1", positive=True)
        self.T2 = _validate_scalar(T2, "T2", positive=True)
        self.off_resonance = _validate_scalar(off_resonance, "off_resonance")

    @property
    def mxy(self):
        """ transverse magnetization as the complex number Mx + j My. """
        return self.magnetization[0] + 1j * self.magnetization[1]

    def tip_90(self):
        """ instant 90 degree tip: all of M0 goes into the transverse plane. """
        self.magnetization = np.asarray([self.M0, 0.0, 0.0])

    def update(self, B, dt, G=None):
        """ advance magnetization over time step of duration dt
            with external field B. if G is given, the spin also
            feels an extra z field G · r from the gradient. """

        B = _validate_vector(B, "B")
        dt = _validate_scalar(dt, "dt", positive=True)

        if G is not None:
            G = _validate_vector(G, "G")
            B = B + np.array([0.0, 0.0, np.dot(G, self.position)])

        B_eff = _effective_B(B, self.off_resonance)

        _relax(self.magnetization, self.M0, self.T1, self.T2, dt / 2)
        _precess(self.magnetization, B_eff, dt)
        _relax(self.magnetization, self.M0, self.T1, self.T2, dt / 2)

    def reset(self):
        """ resets magnetization to the fully relaxed state. that is, no
            transverse component and a longitudinal component of M0. """

        self.magnetization = np.asarray([0.0, 0.0, self.M0])
