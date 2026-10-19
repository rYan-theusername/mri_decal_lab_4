import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def plot_vector_components(v, t, title='Vector Components over Time', labels=('$x$', '$y$', '$z$'),
                           linestyles=('b-', 'r-', 'g-'), names=None):
    """ plot Mx, My, Mz vs time. v is (3, T), or a list of (3, T) arrays
        to overlay several spins on one axes. if a list is given, `names`
        labels each spin in the legend (e.g. isocenter vs off-center). """

    plt.figure(figsize=(8, 4))
    series = v if isinstance(v, (list, tuple)) else [v]
    spin_names = names if names is not None else [None] * len(series)
    dash_styles = ('-', '--', ':', '-.')
    comp_labels = ('$M_x$', '$M_y$', '$M_z$')
    comp_colors = ('C0', 'C3', 'C2')

    for j, Mj in enumerate(series):
        Mj = np.asarray(Mj)
        for i in range(Mj.shape[0]):
            spin_name = spin_names[j]
            if spin_name:
                label = f'{spin_name} {comp_labels[i] if i < 3 else f"comp {i}"}'
            else:
                label = labels[i] if i < len(labels) else f'Component {i+1}'
            style = dash_styles[j] if len(series) > 1 else linestyles[i]
            color = comp_colors[i] if i < 3 else None
            plt.plot(t, Mj[i, :], linestyle=style, color=color, label=label, linewidth=2)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def plot_1D_phantom(x, M0, title='1D object'):
    """ stem plot of a discrete 1D spin phantom. """

    x = np.asarray(x)
    M0 = np.asarray(M0)
    plt.figure(figsize=(8, 4))
    plt.stem(x, M0, basefmt='k-')
    plt.xlabel('$x$ (cm)')
    plt.ylabel('$M_0$')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_2d_phantom(x, y, M0, title='2D object'):
    """ grayscale image of a discrete 2D spin phantom. """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    M0 = np.asarray(M0, dtype=float)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(
        M0, cmap='gray', origin='lower',
        extent=[x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2],
    )
    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('$y$ (cm)')
    ax.set_title(title)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.show()


def _image_extent(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    return [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]


def _grid_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X, Y = np.meshgrid(x, y)
    return x, y, X.ravel(), Y.ravel()


def _phase_arrow_uv(mxy, arrow_length):
    mags = np.abs(mxy)
    mag_safe = np.maximum(mags, 1e-12)
    ux = np.real(mxy) / mag_safe * arrow_length
    uy = np.imag(mxy) / mag_safe * arrow_length
    return ux, uy, mags


def plot_object_and_recon(x, m0, img, title='Object vs Reconstruction'):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, m0, 'ko-', label='object $M_0(x)$', linewidth=2)
    ax.plot(x, np.real(img), 'C0s--', label='recon (real)', linewidth=2)
    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('amplitude')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.show()


def _render_signal_and_k(fig, t, s, k, title='Received Signal and $k(t)$'):
    t = np.asarray(t)
    s = np.asarray(s)
    k = np.asarray(k)
    if k.ndim == 2:
        k = k[0]

    fig.clear()
    axes = fig.subplots(2, 1, sharex=True)
    axes[0].stem(t, np.abs(s), linefmt='C0-', markerfmt='C0o', basefmt='k-')
    axes[0].set_ylabel(r'$|s(t)|$')
    axes[0].set_title(title)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].stem(t, k, linefmt='C1-', markerfmt='C1o', basefmt='k-')
    axes[1].axhline(0, color='gray', linewidth=1)
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel(r'$k_x$ (1/cm)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    if getattr(fig, 'canvas', None) is not None:
        fig.canvas.draw()
    return fig


def plot_signal_and_k(t, s, k, title='Received Signal and $k(t)$'):
    """ |s|(t) on top, k(t) on the bottom. k may be 1D or the x-row of (3, n). """

    fig, _axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    _render_signal_and_k(fig, t, s, k, title=title)
    plt.show()


def plot_kspace_and_image(kspace, image, extent=None,
                          titles=('|k-space|', 'reconstruction')):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(np.abs(kspace), cmap='gray', origin='lower')
    axes[0].set_title(titles[0])
    axes[0].set_xlabel('$k_x$ index')
    axes[0].set_ylabel('$k_y$ index')
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.abs(image), cmap='gray', origin='lower', extent=extent)
    axes[1].set_title(titles[1])
    if extent is None:
        axes[1].set_xlabel('$x$ index')
        axes[1].set_ylabel('$y$ index')
    else:
        axes[1].set_xlabel('$x$ (cm)')
        axes[1].set_ylabel('$y$ (cm)')
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    plt.show()


def animate_running_recon(k, s, x, m0, fps=6):
    """ add k-space samples from left to right and watch the 1D recon grow. """

    from helpers import centered_ifft

    k = np.asarray(k)
    s = np.asarray(s)
    x = np.asarray(x)
    m0 = np.asarray(m0)
    n = len(s)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.set_tight_layout(True)
    stem_container = axes[0].plot(k, np.abs(s), 'C0o-', linewidth=1, alpha=0.3)
    filled, = axes[0].plot([], [], 'C3o', markersize=6)
    axes[0].set_xlabel(r'$k_x$ (1/cm)')
    axes[0].set_ylabel(r'$|s(k)|$')
    axes[0].set_title('k-space samples so far')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(x, m0, 'k-', linewidth=2, label='object')
    recon_line, = axes[1].plot(x, np.zeros_like(x), 'C0--', linewidth=2, label='recon')
    axes[1].set_xlabel('$x$ (cm)')
    axes[1].set_title('running reconstruction')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    ymax = max(np.max(m0), 1e-6) * 1.3
    axes[1].set_ylim(-0.2 * ymax, ymax)

    def update(frame):
        i = frame + 1
        filled.set_data(k[:i], np.abs(s[:i]))
        s_partial = np.zeros_like(s)
        s_partial[:i] = s[:i]
        img = centered_ifft(s_partial)
        recon_line.set_ydata(np.real(img))
        return [filled, recon_line]

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_running_recon_2d(kspace, M0, extent=None, fps=4):
    """ add phase-encode lines from most-negative ky upward and watch
        the 2D recon grow. """

    from helpers import centered_ifft2

    kspace = np.asarray(kspace)
    M0 = np.asarray(M0)
    n_pe, n_read = kspace.shape
    vmax_k = max(float(np.max(np.abs(kspace))), 1e-12)
    vmax_i = max(float(np.max(np.abs(M0))), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.set_tight_layout(True)
    shown = np.zeros_like(kspace)
    im0 = axes[0].imshow(np.abs(shown), cmap='gray', origin='lower',
                         vmin=0, vmax=vmax_k)
    axes[0].set_title('k-space lines so far')
    axes[0].set_xlabel('$k_x$ index')
    axes[0].set_ylabel('$k_y$ index')
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.zeros_like(M0), cmap='gray', origin='lower',
                         vmin=0, vmax=vmax_i, extent=extent)
    axes[1].set_title('running reconstruction')
    if extent is None:
        axes[1].set_xlabel('$x$ index')
        axes[1].set_ylabel('$y$ index')
    else:
        axes[1].set_xlabel('$x$ (cm)')
        axes[1].set_ylabel('$y$ (cm)')
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    def update(frame):
        n = frame + 1
        shown[:] = 0
        shown[:n, :] = kspace[:n, :]
        im0.set_data(np.abs(shown))
        im1.set_data(np.abs(centered_ifft2(shown)))
        axes[0].set_title(f'k-space after {n} / {n_pe} phase encodes')
        return [im0, im1]

    anim = FuncAnimation(fig, update, frames=n_pe, interval=1000 / fps)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def interactive_dephasing(x, M0, Gx=0.05, duration=0.01, dt=0.001,
                          tip_fn=None, fps=8):
    """ jshtml animation of spins dephasing under a constant Gx.
        change Gx in the calling cell and re-run to try another value. """

    from helpers import make_line_spins, instant_90, simulate_waveform
    if tip_fn is None:
        tip_fn = instant_90

    x = np.asarray(x, dtype=float)
    M0 = np.asarray(M0, dtype=float)
    n_steps = max(2, int(round(duration / dt)))
    vmax = max(float(np.max(M0)), 1e-12)
    arrow_length = 0.35
    alive = M0 > 0
    x_plot = x[alive]

    spins = make_line_spins(x, M0)
    tip_fn(spins)
    s, _, mxy = simulate_waveform(
        spins, np.full(n_steps, Gx), dt, record_mxy=True
    )

    def arrows_for(i):
        m = mxy[alive, i]
        mags = np.abs(m)
        mag_safe = np.maximum(mags, 1e-12)
        ux = np.real(m) / mag_safe * arrow_length
        uy = np.imag(m) / mag_safe * arrow_length
        return ux, uy, mags

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.set_tight_layout(True)
    ux, uy, mags = arrows_for(0)
    ax.axhline(0, color='gray', linewidth=1)
    q = ax.quiver(
        x_plot, np.zeros_like(x_plot), ux, uy, mags,
        cmap='viridis', clim=(0, vmax),
        angles='xy', scale_units='xy', scale=1, width=0.006,
    )
    fig.colorbar(q, ax=ax, label=r'$|M_{xy}|$', fraction=0.046)
    ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('transverse plane')
    title = ax.set_title(rf'$t$ = 0.00 ms, $G_x$ = {Gx:.2f} G/cm')
    sig_text = ax.text(
        0.02, 0.95, rf'$|s| = |\sum M_{{xy}}| = {np.abs(s[0]):.2f}$',
        transform=ax.transAxes, va='top', fontsize=11,
    )
    ax.grid(True, linestyle='--', alpha=0.5)

    def update(i):
        ux, uy, mags = arrows_for(i)
        q.set_UVC(ux, uy, mags)
        t_ms = i * dt * 1e3
        title.set_text(rf'$t$ = {t_ms:.2f} ms, $G_x$ = {Gx:.2f} G/cm')
        sig_text.set_text(rf'$|s| = |\sum M_{{xy}}| = {np.abs(s[i]):.2f}$')
        return [q, title, sig_text]

    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000 / fps)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def interactive_dephasing_2d(x, y, M0, Gx=0.05, duration=0.02, dt=0.001,
                             tip_fn=None, fps=8):
    """ jshtml animation of a 2D grid dephasing under a constant Gx.
        change Gx in the calling cell and re-run to try another value. """

    from helpers import make_grid_spins, instant_90, simulate_waveform
    if tip_fn is None:
        tip_fn = instant_90

    x, y, xs, ys = _grid_xy(x, y)
    M0 = np.asarray(M0, dtype=float)
    m0 = M0.ravel()
    if m0.size != xs.size:
        raise ValueError('M0 must have shape (len(y), len(x))')

    n_steps = max(2, int(round(duration / dt)))
    vmax = max(float(np.max(m0)), 1e-12)
    dx = x[1] - x[0] if len(x) > 1 else 0.2
    arrow_length = 0.4 * dx
    alive = m0 > 1e-6 * vmax
    xs_p, ys_p = xs[alive], ys[alive]

    spins = make_grid_spins(x, y, M0)
    tip_fn(spins)
    s, _, mxy = simulate_waveform(
        spins, np.full(n_steps, Gx), dt, record_mxy=True
    )

    def arrows_for(i):
        return _phase_arrow_uv(mxy[alive, i], arrow_length)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.set_tight_layout(True)
    ax.imshow(M0, cmap='gray', origin='lower',
              extent=_image_extent(x, y), alpha=0.25)
    ux, uy, mags = arrows_for(0)
    q = ax.quiver(
        xs_p, ys_p, ux, uy, mags,
        cmap='viridis', clim=(0, vmax),
        angles='xy', scale_units='xy', scale=1, width=0.006,
    )
    fig.colorbar(q, ax=ax, label=r'$|M_{xy}|$', fraction=0.046)
    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('$y$ (cm)')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    title = ax.set_title(rf'$t$ = 0.00 ms, $G_x$ = {Gx:.2f} G/cm')
    sig_text = ax.text(
        0.02, 0.97, rf'$|s| = |\sum M_{{xy}}| = {np.abs(s[0]):.2f}$',
        transform=ax.transAxes, va='top', fontsize=11,
    )

    def update(i):
        ux, uy, mags = arrows_for(i)
        q.set_UVC(ux, uy, mags)
        t_ms = i * dt * 1e3
        title.set_text(rf'$t$ = {t_ms:.2f} ms, $G_x$ = {Gx:.2f} G/cm')
        sig_text.set_text(rf'$|s| = |\sum M_{{xy}}| = {np.abs(s[i]):.2f}$')
        return [q, title, sig_text]

    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000 / fps)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_2d_encoding(x, y, M0, G, dt, tip_fn=None, fps=8):
    """ jshtml: k-space coordinates on the left, 2D phase arrows on the right.

        G: (n_steps,) as Gx, or (2, n) / (3, n) as a waveform.
        change G in the calling cell and re-run. """

    from helpers import make_grid_spins, instant_90, simulate_waveform
    if tip_fn is None:
        tip_fn = instant_90

    x, y, xs, ys = _grid_xy(x, y)
    M0 = np.asarray(M0, dtype=float)
    m0 = M0.ravel()
    if m0.size != xs.size:
        raise ValueError('M0 must have shape (len(y), len(x))')

    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = np.vstack([G, np.zeros_like(G), np.zeros_like(G)])
    elif G.shape[0] == 2:
        G = np.vstack([G, np.zeros(G.shape[1])])

    spins = make_grid_spins(x, y, M0)
    tip_fn(spins)
    s, k, mxy = simulate_waveform(spins, G, dt, record_mxy=True)
    n_steps = s.size
    vmax = max(float(np.max(m0)), 1e-12)
    dx = x[1] - x[0] if len(x) > 1 else 0.2
    arrow_length = 0.4 * dx
    alive = m0 > 1e-6 * vmax
    xs_p, ys_p = xs[alive], ys[alive]

    def arrows_for(i):
        return _phase_arrow_uv(mxy[alive, i], arrow_length)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.set_tight_layout(True)
    ax_k, ax_m = axes

    kx, ky = k[0], k[1]
    kspan = max(float(np.max(np.abs(kx))), float(np.max(np.abs(ky))), 0.5)
    klim = 1.15 * kspan
    ax_k.axhline(0, color='gray', linewidth=1)
    ax_k.axvline(0, color='gray', linewidth=1)
    ax_k.plot(kx, ky, color='C1', alpha=0.25, linewidth=1)
    path, = ax_k.plot([kx[0]], [ky[0]], color='C1', linewidth=2)
    here, = ax_k.plot([kx[0]], [ky[0]], 'ro', markersize=8)
    ax_k.set_xlim(-klim, klim)
    ax_k.set_ylim(-klim, klim)
    ax_k.set_xlabel(r'$k_x$ (1/cm)')
    ax_k.set_ylabel(r'$k_y$ (1/cm)')
    ax_k.set_aspect('equal', adjustable='box')
    ax_k.set_box_aspect(1)
    ax_k.grid(True, linestyle='--', alpha=0.5)
    k_title = ax_k.set_title(rf'$k = ({kx[0]:.2f},\; {ky[0]:.2f})$')

    ax_m.imshow(M0, cmap='gray', origin='lower',
                extent=_image_extent(x, y), alpha=0.25)
    ux, uy, mags = arrows_for(0)
    q = ax_m.quiver(
        xs_p, ys_p, ux, uy, mags,
        cmap='viridis', clim=(0, vmax),
        angles='xy', scale_units='xy', scale=1, width=0.006,
    )
    fig.colorbar(q, ax=ax_m, label=r'$|M_{xy}|$', fraction=0.046)
    ax_m.set_xlabel('$x$ (cm)')
    ax_m.set_ylabel('$y$ (cm)')
    ax_m.set_aspect('equal')
    ax_m.grid(True, linestyle='--', alpha=0.4)
    m_title = ax_m.set_title(r'transverse $M_{xy}(x,y)$')
    sig_text = ax_m.text(
        0.02, 0.97, rf'$|s| = {np.abs(s[0]):.2f}$',
        transform=ax_m.transAxes, va='top', fontsize=11,
    )

    def update(i):
        ux, uy, mags = arrows_for(i)
        q.set_UVC(ux, uy, mags)
        path.set_data(kx[:i + 1], ky[:i + 1])
        here.set_data([kx[i]], [ky[i]])
        k_title.set_text(rf'$k = ({kx[i]:.2f},\; {ky[i]:.2f})$')
        sig_text.set_text(rf'$|s| = {np.abs(s[i]):.2f}$')
        return [q, path, here, k_title, sig_text, m_title]

    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000 / fps)
    plt.close(fig)
    return HTML(anim.to_jshtml())
