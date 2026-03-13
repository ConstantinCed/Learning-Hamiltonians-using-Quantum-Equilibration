import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_n(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def one_site_op(n, site, op):
    ops = [I2] * n; ops[site] = op; return kron_n(ops)

def two_site_op(n, s1, op1, s2, op2):
    ops = [I2] * n; ops[s1] = op1; ops[s2] = op2; return kron_n(ops)

def nullspace_1d_constraint(c, rtol=1e-12):
    c = c.reshape(1, -1)
    _, S, Vh = np.linalg.svd(c, full_matrices=True)
    rank = int(np.sum(S > rtol * S[0])) if S.size else 0
    return Vh[rank:].T

_cache: dict = {}

def get_cache(n: int) -> dict:
    if n in _cache:
        return _cache[n]
    d, m = 2**n, (n-1)+n
    B_arr = np.empty((m, d, d), dtype=complex)
    for i in range(n-1):
        B_arr[i]         = two_site_op(n, i, Z, i+1, Z)
    for i in range(n):
        B_arr[n-1+i]     = one_site_op(n, i, X)
    states = np.arange(d, dtype=int)
    ZZ = np.empty((n-1, d))
    for i in range(n-1):
        zi  = 1 - 2*((states >> (n-1-i)) & 1)
        zi1 = 1 - 2*((states >> (n-2-i)) & 1)
        ZZ[i] = zi * zi1
    XP = np.empty((n, d), dtype=int)
    for i in range(n):
        XP[i] = states ^ (1 << (n-1-i))
    _cache[n] = dict(B_arr=B_arr, ZZ_diags=ZZ, X_perms=XP)
    return _cache[n]

def compute_B_tilde(ZZ: np.ndarray, XP: np.ndarray,
                    Vd: np.ndarray, V: np.ndarray) -> np.ndarray:
    nZZ, d = ZZ.shape;  nX = XP.shape[0]
    DiV2d = (ZZ[:, :, None] * V[None]).transpose(1, 0, 2).reshape(d, nZZ*d)
    BZZ   = (Vd @ DiV2d).reshape(d, nZZ, d).transpose(1, 0, 2)
    Vdp   = Vd[:, XP].transpose(1, 0, 2).reshape(nX*d, d)
    BX    = (Vdp @ V).reshape(nX, d, d)
    return np.concatenate([BZZ, BX], axis=0)

def make_sample_state(n: int, Bt: np.ndarray, evals: np.ndarray, g: np.ndarray):
    d = 2**n
    m = Bt.shape[0]
    Bt_flat = Bt.reshape(m, d*d)
    return dict(
        Bt_r     = np.ascontiguousarray(Bt_flat.real),
        Bt_i     = np.ascontiguousarray(Bt_flat.imag),
        delta_kl = (evals[:, None] - evals[None, :]).ravel(),
        g        = g,
        m        = m,
        d        = d,
    )

def compute_inf_fast(state: dict, t: float,
                     Bwr: np.ndarray, Bwi: np.ndarray) -> float:
    delta_kl = state["delta_kl"]
    d        = state["d"]

    sq = np.sqrt(np.maximum(2.0 - 2.0*np.cos(delta_kl * t), 0.0))

    np.multiply(state["Bt_r"], sq, out=Bwr)
    np.multiply(state["Bt_i"], sq, out=Bwi)

    Q = (Bwr @ Bwr.T + Bwi @ Bwi.T) / d
    Q = 0.5*(Q + Q.T)

    g = state["g"]
    if np.linalg.norm(g) < 1e-14:
        return float(np.min(np.linalg.eigvalsh(Q)))
    N  = nullspace_1d_constraint(g)
    Qs = N.T @ Q @ N
    return float(np.min(np.linalg.eigvalsh(0.5*(Qs + Qs.T))))

def threshold(n: int) -> float:
    return float(np.sqrt(n / np.log(n)))

def fit_power(ns_arr: np.ndarray, y: np.ndarray):
    mask = y > 1e-14
    if mask.sum() < 2:
        return np.nan, np.nan
    a, b = np.polyfit(np.log(ns_arr[mask]), np.log(y[mask]), 1)
    return float(np.exp(b)), float(a)

def make_t_grid(ns, num_frames=80, t_min=0.02, t_max=10.0):
    return np.geomspace(t_min, t_max, num_frames)

SAMPLES_FOR_N = {4: 50, 5: 50, 6: 50, 7: 30, 8: 15, 9: 8}

def compute_all(ns, tgrid, seed=0):
    rng    = np.random.default_rng(seed)
    ns_arr = np.array(ns, dtype=float)
    T, N   = len(tgrid), len(ns)

    means = np.zeros((T, N))
    stds  = np.zeros((T, N))

    for ni, n in enumerate(ns):
        S = SAMPLES_FOR_N.get(n, 50)
        print(f"\nn = {n}  ({S} samples × {T} frames)", flush=True)

        cache  = get_cache(n)
        B_arr  = cache["B_arr"]
        ZZ, XP = cache["ZZ_diags"], cache["X_perms"]

        d, m = 2**n, (n-1)+n

        Bwr = np.empty((m, d*d))
        Bwi = np.empty((m, d*d))

        frame_sum  = np.zeros(T)
        frame_sum2 = np.zeros(T)

        for si in range(S):
            J = rng.normal(size=n-1)
            h = rng.normal(size=n)
            g = np.concatenate([J, h])
            H            = np.tensordot(g, B_arr, axes=[[0], [0]])
            evals, evecs = np.linalg.eigh(H)
            Vd           = evecs.conj().T
            Bt           = compute_B_tilde(ZZ, XP, Vd, evecs)    # once per sample
            state        = make_sample_state(n, Bt, evals, g)

            for ti, t in enumerate(tgrid):
                v = compute_inf_fast(state, t, Bwr, Bwi)
                frame_sum[ti]  += v
                frame_sum2[ti] += v * v

            if (si+1) % 5 == 0 or si == S-1:
                print(f"  sample {si+1}/{S}", flush=True)

        means[:, ni] = frame_sum  / S
        stds[:, ni]  = np.sqrt(np.maximum(frame_sum2/S - (frame_sum/S)**2, 0.0))

    C_arr = np.zeros(T)
    a_arr = np.full(T, np.nan)
    for ti in range(T):
        C_arr[ti], a_arr[ti] = fit_power(ns_arr, means[ti])

    return means, stds, C_arr, a_arr

def animate(ns, tgrid, means, stds, C_arr, a_arr, gif_path, fps=5):
    ns_arr    = np.array(ns, dtype=float)
    tstar     = np.array([threshold(n) for n in ns])
    tstar_max = tstar.max()

    pos_means = means[means > 0]
    y_lo = pos_means.min() * 0.35 if pos_means.size else 1e-4
    y_hi = means.max() * 3.0

    valid_a = a_arr[np.isfinite(a_arr)]
    a_lo = (valid_a.min() - 0.4) if valid_a.size else -3.0
    a_hi = (valid_a.max() + 0.4) if valid_a.size else  1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("#0d0d1a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#131328")
        ax.tick_params(colors="#bbbbbb")
        ax.xaxis.label.set_color("#bbbbbb")
        ax.yaxis.label.set_color("#bbbbbb")
        ax.title.set_color("#eeeeee")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    ax1.set_yscale("log")
    ax1.set_xlim(ns_arr[0] - 0.5, ns_arr[-1] + 0.5)
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_xlabel("n  (qubits)", fontsize=11)
    ax1.set_ylabel(r"$\inf_A\;\frac{1}{d}\|[U,A]\|_F^2$", fontsize=11)

    thr_vlines = [
        ax1.axvline(nv, color="#ff6b6b", lw=2.5, alpha=0.05, zorder=1)
        for nv in ns_arr
    ]

    line_mean, = ax1.plot([], [], "o-", color="#4cc9f0", lw=2.2, ms=7,
                          label="sample mean", zorder=3)
    line_fit,  = ax1.plot([], [], "--", color="#f72585", lw=1.8,
                          label="power-law fit", zorder=2)

    fill_holder = [ax1.fill_between([], [], [], alpha=0.0)]

    ax1.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#555577",
               labelcolor="#cccccc", loc="upper right")

    ax2.set_xlim(tgrid[0], tgrid[-1])
    ax2.set_ylim(a_lo, a_hi)
    ax2.set_xscale("log")
    ax2.set_xlabel("t  (log scale)", fontsize=11)
    ax2.set_ylabel(r"power-law exponent  $\alpha$", fontsize=11)
    ax2.set_title(r"$\langle\mathrm{inf}\rangle \approx C\,n^\alpha$", fontsize=11)

    ax2.axvspan(tstar.min(), tstar.max(),
                color="#ff6b6b", alpha=0.10, label=r"$t^*(n)$ band")
    ax2.axhline(0, color="#666688", lw=0.8, ls=":")
    ax2.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#555577",
               labelcolor="#cccccc")

    line_alpha, = ax2.plot([], [], "o-", color="#06d6a0", lw=2.0, ms=4)
    t_marker     = ax2.axvline(tgrid[0], color="#f8961e", lw=1.5, ls="--")

    info_box = ax1.text(
        0.03, 0.05, "", transform=ax1.transAxes, va="bottom",
        color="#f8961e", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="#1a1a2e", edgecolor="#555577", alpha=0.88)
    )

    plt.tight_layout(pad=2.0)

    def update(i):
        t  = tgrid[i]
        m  = means[i]
        s  = stds[i]
        ai = a_arr[i]
        ci = C_arr[i]

        line_mean.set_data(ns_arr, m)

        fill_holder[0].remove()
        fill_holder[0] = ax1.fill_between(
            ns_arr,
            np.maximum(m - s, y_lo * 1.01),
            m + s,
            alpha=0.22, color="#4cc9f0", zorder=2
        )

        if np.isfinite(ai):
            line_fit.set_data(ns_arr, ci * ns_arr**ai)
        else:
            line_fit.set_data([], [])

        for vl, ts in zip(thr_vlines, tstar):
            frac = min(t / ts, 1.0)
            vl.set_alpha(0.05 + 0.85 * frac)
            vl.set_color("#ff6b6b" if t >= ts else "#ffaaaa")

        valid = np.isfinite(a_arr[:i+1])
        line_alpha.set_data(tgrid[:i+1][valid], a_arr[:i+1][valid])
        t_marker.set_xdata([t, t])

        a_str = f"{ai:+.3f}" if np.isfinite(ai) else "—"
        info_box.set_text(f"t  = {t:.3f}\nα  = {a_str}\nt* = {tstar_max:.2f}")

        ax1.set_title(
            f"TFIM  |  n = {ns[0]}…{ns[-1]}  |  t = {t:.3f}",
            fontsize=11, color="#eeeeee"
        )

    anim = FuncAnimation(fig, update, frames=len(tgrid),
                         interval=int(1000/fps), blit=False)

    print(f"\nRendering {len(tgrid)}-frame GIF → {gif_path}", flush=True)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    print("Done.")
    plt.close(fig)

if __name__ == "__main__":

    ns         = list(range(4, 10))
    num_frames = 120
    fps        = 5
    gif_path   = "/mnt/user-data/outputs/tfim_time_sweep.gif"

    tgrid = make_t_grid(ns, num_frames, t_max=20.0)

    tstar_info = {n: f"{threshold(n):.2f}" for n in ns}
    print("=== TFIM time-sweep animation ===")
    print(f"n range      : {ns[0]} … {ns[-1]}")
    print(f"Frames       : {num_frames}  (t: {tgrid[0]:.4f} → {tgrid[-1]:.4f})")
    print(f"Thresholds   : {tstar_info}")
    print(f"Samples/n    : {SAMPLES_FOR_N}")
    print()

    means, stds, C_arr, a_arr = compute_all(ns, tgrid)

    npz_path = gif_path.replace(".gif", "_data.npz")
    np.savez(npz_path, tgrid=tgrid, means=means, stds=stds,
             C_arr=C_arr, a_arr=a_arr, ns=np.array(ns))
    print(f"Data saved to {npz_path}")

    print("\n=== Rendering animation ===")
    animate(ns, tgrid, means, stds, C_arr, a_arr, gif_path, fps=fps)
