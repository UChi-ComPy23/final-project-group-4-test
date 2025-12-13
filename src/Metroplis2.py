import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""
2D Ising model (nearest-neighbor, periodic BC) using Metropolis.
This version includes:
- Correct definitions for susceptibility and specific heat (scaled by beta, beta^2 and volume V=N^2)
- Blocking (batch means) standard error estimates per temperature
- plot_results_error reads *_err keys directly (no summary_with_error needed)
"""


# ============================================================
#  Utilities: initialization
# ============================================================
def initialize_lattice(N, mode="cold"):
    if mode == "cold":
        lattice = np.ones((N, N), dtype=np.int8)
    elif mode == "hot":
        lattice = np.random.choice([1, -1], size=(N, N)).astype(np.int8)
    else:
        raise ValueError(f"Unrecognized mode: {mode}; must be either 'cold' or 'hot'")
    return lattice


# ============================================================
#  Energy change for a single flip
# ============================================================
def compute_energy_change(lattice, i, j, J=1.0):
    N = lattice.shape[0]
    spin = lattice[i, j]

    up = lattice[(i - 1) % N, j]
    down = lattice[(i + 1) % N, j]
    left = lattice[i, (j - 1) % N]
    right = lattice[i, (j + 1) % N]

    neighbors = up + down + left + right
    dH = 2.0 * J * spin * neighbors
    return dH


# ============================================================
#  Metropolis updates
# ============================================================
def metropolis_single_flip(lattice, beta, J=1.0):
    N = lattice.shape[0]
    i, j = np.random.randint(0, N, size=2)
    dH = compute_energy_change(lattice, i, j, J=J)

    if dH <= 0:
        lattice[i, j] *= -1
    else:
        if np.random.rand() < np.exp(-beta * dH):
            lattice[i, j] *= -1
    return lattice


def metropolis_update(lattice, beta, J=1.0):
    N = lattice.shape[0]
    for _ in range(N * N):
        lattice = metropolis_single_flip(lattice, beta, J=J)
    return lattice


# ============================================================
#  Observables
# ============================================================
def compute_magnetization(lattice):
    return np.sum(lattice, dtype=np.float64)


def compute_energy(lattice, J=1.0):
    # Count each bond once: right and down neighbors
    return -J * np.sum(lattice * (np.roll(lattice, -1, axis=0) + np.roll(lattice, -1, axis=1)), dtype=np.float64)


def compute_observables(mag_samples, energy_samples, N, beta):
    """
    mag_samples: total magnetization M (sum of spins)
    energy_samples: total energy E (with bonds counted once)
    """
    mag_samples = np.asarray(mag_samples, dtype=np.float64)
    energy_samples = np.asarray(energy_samples, dtype=np.float64)
    V = N * N

    m1 = np.mean(mag_samples)
    m2 = np.mean(mag_samples**2)
    m3 = np.mean(mag_samples**3)
    m4 = np.mean(mag_samples**4)

    e1 = np.mean(energy_samples)
    e2 = np.mean(energy_samples**2)

    var_m = m2 - m1**2
    var_e = e2 - e1**2

    # Correct physical scaling for TOTAL M and TOTAL E
    susceptibility = beta * var_m / V
    specific_heat = (beta**2) * var_e / V

    # Central standardized moments for M (beware: can be noisy near Tc)
    if var_m <= 1e-12:
        skewness = np.nan
        kurtosis = np.nan
    else:
        mu3 = m3 - 3*m1*m2 + 2*m1**3
        mu4 = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*m1**4
        skewness = mu3 / (var_m**1.5)
        kurtosis = mu4 / (var_m**2)

    return susceptibility, skewness, kurtosis, specific_heat


# ============================================================
#  Error estimation: blocking (batch means)
# ============================================================
def block_standard_error(x, n_blocks=20):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return np.nan

    n_blocks = int(min(n_blocks, n))
    block_size = n // n_blocks

    if block_size < 2:
        return np.std(x, ddof=1) / np.sqrt(n)

    trimmed = x[:block_size * n_blocks]
    blocks = trimmed.reshape(n_blocks, block_size).mean(axis=1)
    return np.std(blocks, ddof=1) / np.sqrt(n_blocks)


# ============================================================
#  Simulation (stores means + per-temperature error bars)
# ============================================================
def simulate_ising(
    N,
    beta_range,
    n_samples,
    n_therm=1000,
    init_mode="cold",
    J=1.0,
    n_blocks=20,
):
    summary = {
        "temperatures": [],

        "magnetizations": [], "magnetizations_err": [],
        "susceptibilities": [], "susceptibilities_err": [],
        "skewnesses": [], "skewnesses_err": [],
        "kurtoses": [], "kurtoses_err": [],
        "specific_heats": [], "specific_heats_err": [],
    }

    V = N * N

    for beta in tqdm(beta_range, desc="Metropolis Simulating"):
        lattice = initialize_lattice(N, mode=init_mode)

        # thermalization
        for _ in range(n_therm):
            lattice = metropolis_update(lattice, beta, J=J)

        # sampling
        m_samples = np.empty(n_samples, dtype=np.float64)
        e_samples = np.empty(n_samples, dtype=np.float64)

        for t in range(n_samples):
            lattice = metropolis_update(lattice, beta, J=J)
            m_samples[t] = compute_magnetization(lattice)
            e_samples[t] = compute_energy(lattice, J=J)

        # --- Means ---
        # If you want per-spin magnetization squared, divide by V^2:
        m2_series = m_samples**2
        m2_mean = np.mean(m2_series) / (V**2)
        chi, skew, kurt, c = compute_observables(m_samples, e_samples, N, beta)

        # --- Errors (blocking) ---
        m2_err = block_standard_error(m2_series / (V**2), n_blocks=n_blocks)

        m1 = np.mean(m_samples)
        var_m_series = (m_samples - m1)**2
        chi_series = beta * var_m_series / V
        chi_err = block_standard_error(chi_series, n_blocks=n_blocks)

        e1 = np.mean(e_samples)
        var_e_series = (e_samples - e1)**2
        c_series = (beta**2) * var_e_series / V
        c_err = block_standard_error(c_series, n_blocks=n_blocks)

        # Skew/Kurt errors: block-recompute (cheap jackknife-like)
        skew_err = np.nan
        kurt_err = np.nan
        block_size = n_samples // n_blocks
        if block_size >= 10:
            sk_list = []
            ku_list = []
            for b in range(n_blocks):
                xs = m_samples[b*block_size:(b+1)*block_size]
                ys = e_samples[b*block_size:(b+1)*block_size]
                _, sk_b, ku_b, _ = compute_observables(xs, ys, N, beta)
                sk_list.append(sk_b)
                ku_list.append(ku_b)
            sk_list = np.asarray(sk_list, dtype=np.float64)
            ku_list = np.asarray(ku_list, dtype=np.float64)
            if np.all(np.isfinite(sk_list)) and len(sk_list) > 1:
                skew_err = np.std(sk_list, ddof=1) / np.sqrt(len(sk_list))
            if np.all(np.isfinite(ku_list)) and len(ku_list) > 1:
                kurt_err = np.std(ku_list, ddof=1) / np.sqrt(len(ku_list))

        # store
        summary["temperatures"].append(1.0 / beta)

        summary["magnetizations"].append(m2_mean)
        summary["magnetizations_err"].append(m2_err)

        summary["susceptibilities"].append(chi)
        summary["susceptibilities_err"].append(chi_err)

        summary["skewnesses"].append(skew)
        summary["skewnesses_err"].append(skew_err)

        summary["kurtoses"].append(kurt)
        summary["kurtoses_err"].append(kurt_err)

        summary["specific_heats"].append(c)
        summary["specific_heats_err"].append(c_err)

    return summary


def estimate_critical_temperature(results, beta_range):
    susceptibility = np.asarray(results["susceptibilities"], dtype=np.float64)
    max_index = int(np.nanargmax(susceptibility))
    critical_beta = beta_range[max_index]
    print(f"Estimated Critical Beta (1/T): {critical_beta}")
    print(f"Estimated Critical Temperature Tc: {1.0/critical_beta}")
    return critical_beta


# ============================================================
#  Plotting (reads new *_err keys directly)
# ============================================================
def plot_results_error(results, N):
    temperatures = np.asarray(results["temperatures"], dtype=np.float64)

    plt.figure(figsize=(12, 8))

    # Magnetization squared
    plt.subplot(2, 3, 1)
    plt.errorbar(
        temperatures,
        results["magnetizations"],
        yerr=results["magnetizations_err"],
        fmt='o-',
        label="Magnetization Squared (per-spin)"
    )
    plt.xlabel("Temperature (T)")
    plt.ylabel(r"$\langle m^2 \rangle$")  # m = M/V
    plt.legend()
    plt.grid()

    # Susceptibility
    plt.subplot(2, 3, 2)
    plt.errorbar(
        temperatures,
        results["susceptibilities"],
        yerr=results["susceptibilities_err"],
        fmt='o-',
        label="Susceptibility"
    )
    plt.xlabel("Temperature (T)")
    plt.ylabel(r"$\chi$")
    plt.legend()
    plt.grid()

    # Skewness
    plt.subplot(2, 3, 3)
    plt.errorbar(
        temperatures,
        results["skewnesses"],
        yerr=results["skewnesses_err"],
        fmt='o-',
        label="Skewness (M)"
    )
    plt.xlabel("Temperature (T)")
    plt.ylabel("Skewness")
    plt.legend()
    plt.grid()

    # Kurtosis
    plt.subplot(2, 3, 4)
    plt.errorbar(
        temperatures,
        results["kurtoses"],
        yerr=results["kurtoses_err"],
        fmt='o-',
        label="Kurtosis (M)"
    )
    plt.xlabel("Temperature (T)")
    plt.ylabel("Kurtosis")
    plt.legend()
    plt.grid()

    # Specific Heat
    plt.subplot(2, 3, 5)
    plt.errorbar(
        temperatures,
        results["specific_heats"],
        yerr=results["specific_heats_err"],
        fmt='o-',
        label="Specific Heat"
    )
    plt.xlabel("Temperature (T)")
    plt.ylabel(r"$C$")
    plt.legend()
    plt.grid()

    plt.suptitle(f"Extended Observables for Lattice Size N={N}")
    plt.tight_layout()
    plt.show()


# ============================================================
#  Main
# ============================================================
