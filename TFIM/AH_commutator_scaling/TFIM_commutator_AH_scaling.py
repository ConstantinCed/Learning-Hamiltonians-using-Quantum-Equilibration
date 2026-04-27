import os
import csv
import numpy as np
from google.colab import files

# Output directory in Colab
OUTDIR = "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)


def compute_Q(n, J, h):
    """
    Build Q = V V^T for the TFIM-style coefficient vector.

    Parameters
    ----------
    n : int
        Number of spins/sites.
    J : np.ndarray, shape (n-1,)
        Coupling coefficients.
    h : np.ndarray, shape (n,)
        Field coefficients.

    Returns
    -------
    Q : np.ndarray, shape (2n-1, 2n-1)
        Positive semidefinite matrix encoding the quadratic form.
    """

    m = 2 * n - 1
    V = np.zeros((m, 2 * (n - 1)))

    ks = np.arange(n - 1)

    # h terms
    V[ks, ks] = 2.0 * h[:-1]
    V[ks, n - 1 + ks] = 2.0 * h[1:]

    # boundary J terms
    if n >= 2:
        V[n - 1, 0] = 2.0 * J[0]
        V[2 * n - 2, 2 * n - 3] = 2.0 * J[n - 2]

    # bulk J terms
    if n >= 3:
        js = np.arange(1, n - 1)
        V[n - 1 + js, js] = 2.0 * J[js]
        V[n - 1 + js, n - 2 + js] = 2.0 * J[js - 1]

    return V @ V.T


def constrained_min_eig(Q, g):
    """
    Compute the minimum eigenvalue of Q restricted to the subspace
    orthogonal to g.

    The observable vector O is constrained by

        ||O|| = 1,
        <O, g> = 0,

    where g is the Hamiltonian coefficient vector.

    This function enforces orthogonality by adding a large penalty
    in the g direction:

        Q_def = Q + (mu / ||g||^2) g g^T.

    Since the penalty is positive and large, the smallest eigenvector
    of Q_def lies approximately in the subspace orthogonal to g.

    Parameters
    ----------
    Q : np.ndarray, shape (m, m)
        Positive semidefinite matrix.
    g : np.ndarray, shape (m,)
        Hamiltonian coefficient vector.

    Returns
    -------
    float
        Approximate constrained minimum eigenvalue.
    """

    g2 = np.dot(g, g)

    if g2 < 1e-28:
        return float(np.linalg.eigvalsh(Q)[0])

    # Penalty scale. This is chosen larger than the natural scale of Q.
    mu = float(np.trace(Q)) + 1.0

    Qdef = Q + (mu / g2) * np.outer(g, g)

    return float(np.linalg.eigvalsh(Qdef)[0])


def sample_inf_value_unnormalized(n, rng, distribution):
    """
    Draw unnormalized Hamiltonian coefficients and compute the constrained
    minimum eigenvalue.

    The Hamiltonian coefficients are NOT normalized.

    However, the observable is still normalized and constrained to be
    orthogonal to H.

    Parameters
    ----------
    n : int
        Number of sites.
    rng : np.random.Generator
        Random number generator.
    distribution : str
        Either "gaussian" or "uniform".

    Returns
    -------
    float
        Constrained minimum eigenvalue.
    """

    if distribution == "gaussian":
        J = rng.normal(loc=0.0, scale=1.0, size=n - 1)
        h = rng.normal(loc=0.0, scale=1.0, size=n)

    elif distribution == "uniform":
        J = rng.uniform(low=-1.0, high=1.0, size=n - 1)
        h = rng.uniform(low=-1.0, high=1.0, size=n)

    else:
        raise ValueError("distribution must be either 'gaussian' or 'uniform'")

    # Hamiltonian coefficient vector, unnormalized
    g = np.concatenate([J, h])

    Q = compute_Q(n, J, h)

    return constrained_min_eig(Q, g)


def run_experiment(
    ns,
    num_samples,
    seed,
    distribution,
    outdir=OUTDIR,
):
    """
    Run Monte Carlo experiment for a given coefficient distribution.

    Saves a CSV with columns:

        n, m=2n-1, mean, std, sem

    Parameters
    ----------
    ns : list[int]
        System sizes.
    num_samples : int
        Number of random Hamiltonian samples per n.
    seed : int
        Random seed.
    distribution : str
        Either "gaussian" or "uniform".
    outdir : str
        Output directory.

    Returns
    -------
    csv_path : str
        Path to saved CSV.
    means, stds, sems : np.ndarray
        Summary statistics.
    """

    rng = np.random.default_rng(seed)

    vals = {n: [] for n in ns}

    for n in ns:
        print(f"Running n = {n}, distribution = {distribution}")

        for _ in range(num_samples):
            val = sample_inf_value_unnormalized(
                n=n,
                rng=rng,
                distribution=distribution,
            )
            vals[n].append(val)

    means = np.array([np.mean(vals[n]) for n in ns])
    stds = np.array([np.std(vals[n], ddof=1) for n in ns])
    sems = stds / np.sqrt(num_samples)

    csv_path = os.path.join(
        outdir,
        f"tfim_data_unnormalized_{distribution}.csv"
    )

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow([
            "n",
            "m=2n-1",
            "distribution",
            "mean",
            "std",
            "sem",
            "num_samples",
        ])

        for i, n in enumerate(ns):
            w.writerow([
                n,
                int(2 * n - 1),
                distribution,
                f"{means[i]:.10e}",
                f"{stds[i]:.10e}",
                f"{sems[i]:.10e}",
                num_samples,
            ])

    return csv_path, means, stds, sems


# -----------------------------
# Main run
# -----------------------------

seed = 42

ns = list(range(4, 201))
num_samples = 1000

# Run Gaussian N(0,1)
csv_gaussian, means_gaussian, stds_gaussian, sems_gaussian = run_experiment(
    ns=ns,
    num_samples=num_samples,
    seed=seed,
    distribution="gaussian",
)

# Run Uniform[-1,1]
csv_uniform, means_uniform, stds_uniform, sems_uniform = run_experiment(
    ns=ns,
    num_samples=num_samples,
    seed=seed + 1,
    distribution="uniform",
)

print()
print("Saved files:")
print(csv_gaussian)
print(csv_uniform)

# Download CSV files from Colab
files.download(csv_gaussian)
files.download(csv_uniform)
