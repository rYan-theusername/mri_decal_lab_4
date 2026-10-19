import numpy as np
from bloch import *


def make_line_spins(x, M0, T1=1e9, T2=1e9):
    """ spawn spins along x. x and M0 are length-N arrays (cm, a.u.). """

    x = np.asarray(x, dtype=float)
    M0 = np.asarray(M0, dtype=float)
    if M0.ndim == 0:
        M0 = np.full_like(x, float(M0))

    spins = []
    for xi, density in zip(x, M0):
        density = float(density)
        stored_M0 = density if density > 0 else 1e-12
        spin = Spin([xi, 0.0, 0.0], [0.0, 0.0, density], stored_M0, T1, T2)
        spin.density = density
        spins.append(spin)
    return spins


def make_grid_spins(x, y, M0, T1=1e9, T2=1e9):
    """ spawn a 2D grid of spins. M0 has shape (len(y), len(x)). """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    M0 = np.asarray(M0, dtype=float)

    spins = []
    for iy, yi in enumerate(y):
        for ix, xi in enumerate(x):
            density = float(M0[iy, ix])
            stored_M0 = density if density > 0 else 1e-12
            spin = Spin([xi, yi, 0.0], [0.0, 0.0, density], stored_M0, T1, T2)
            spin.density = density
            spins.append(spin)
    return spins


def two_block_object(x):
    """ a simple 1D object: two rectangular 'tubes' of different brightness. """

    x = np.asarray(x, dtype=float)
    M0 = np.zeros_like(x)
    M0[(x >= -0.7) & (x <= -0.3)] = 1.0
    M0[(x >= 0.2) & (x <= 0.55)] = 0.6
    return M0


def plus_phantom(ny, nx):
    """ a chunky plus-sign phantom on an (ny, nx) grid. """

    M0 = np.zeros((ny, nx))
    y0, x0 = ny // 2, nx // 2
    half_bar = max(1, ny // 8)
    half_len = max(2, ny // 3)
    M0[y0 - half_bar:y0 + half_bar + 1, x0 - half_len:x0 + half_len + 1] = 1.0
    M0[y0 - half_len:y0 + half_len + 1, x0 - half_bar:x0 + half_bar + 1] = 1.0
    return M0


def instant_90(spins):
    """ tip every spin 90 degrees into the transverse plane. """

    for spin in spins:
        spin.tip_90()


def reset_and_tip(spins):
    """ fully relax, then instant-90. blank pixels stay blank. """

    for spin in spins:
        spin.reset()
    instant_90(spins)


def receive_signal(spins):
    """ coil that just sums the transverse magnetization of every spin. """

    return sum((spin.mxy for spin in spins), start=0j)


def _as_3_by_n(vec, n_steps, name):
    array = np.asarray(vec, dtype=float)
    if array.shape == (3,):
        return np.broadcast_to(array[:, None], (3, n_steps)).copy()
    if array.shape == (3, n_steps):
        return array
    raise ValueError(f"{name} must have shape (3,) or (3, {n_steps}), got {array.shape}")


def simulate_waveform(spins, G, dt, B=None, record_mxy=False, k0=None):
    """ play a gradient (and optional B) waveform, sampling the signal
        at the start of each dwell — i.e. at the current k-space location
        before that step's gradient area is applied.

        G: (3, n_steps) or (n_steps,) interpreted as Gx.
        B: (3,) or (3, n_steps). defaults to zero (rotating frame).
        k0: starting k-space location (3,) or scalar kx. default 0.

        returns s, k, mxy
            s:   (n_steps,) complex received signal
            k:   (3, n_steps) k-space location of each sample, 1/cm
            mxy: (n_spins, n_steps) or None """

    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = np.vstack([G, np.zeros_like(G), np.zeros_like(G)])

    n_steps = G.shape[1]
    if B is None:
        B = np.zeros((3, n_steps))
    else:
        B = _as_3_by_n(B, n_steps, "B")

    s = np.zeros(n_steps, dtype=complex)
    k = np.zeros((3, n_steps))
    if k0 is None:
        k_curr = np.zeros(3)
    else:
        k0 = np.asarray(k0, dtype=float)
        k_curr = np.array([float(k0), 0.0, 0.0]) if k0.ndim == 0 else k0.astype(float).copy()
    mxy = np.zeros((len(spins), n_steps), dtype=complex) if record_mxy else None

    for i in range(n_steps):
        s[i] = receive_signal(spins)
        k[:, i] = k_curr
        if record_mxy:
            for j, spin in enumerate(spins):
                mxy[j, i] = spin.mxy

        Gi = G[:, i]
        Bi = B[:, i]
        for spin in spins:
            spin.update(Bi, dt, G=Gi)
        k_curr = k_curr + (GAMMA / (2 * np.pi)) * Gi * dt

    return s, k, mxy


def centered_fft(m):
    """ fft of a spatially-centered object. k = 0 lands in the middle. """

    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(m)))


def centered_ifft(s):
    """ inverse fft of centered k-space. x = 0 lands in the middle. """

    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(s)))


def centered_fft2(m):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(m)))


def centered_ifft2(s):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(s)))

