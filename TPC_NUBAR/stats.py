import awkward as awk
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
import os
from tqdm import tqdm
import multiprocessing as mp

# Constants
POT = 2.6098758621e22
me = 0.511e-3  # Electron mass in GeV/c^2
background = {'nu': 0.563, 'antinu': 0.015}

# Load efficiency data
def load_efficiency_data(filename="kaon_decay_efficiency_data.txt"):
    data = np.loadtxt(filename, skiprows=2)
    m_n, mu_decay, e_decay = data[:, 0], data[:, 1], data[:, 2]
    min_decay = np.minimum(mu_decay, e_decay)
    extended_m_n = np.arange(150, 2001, 25)
    extended_min = np.interp(extended_m_n, m_n, min_decay)
    return interp1d(extended_m_n, extended_min, kind='linear', fill_value='extrapolate')

min_efficiency_interp = load_efficiency_data()

def normalize_vectors(vectors):
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    magnitudes = np.where(magnitudes == 0, 1, magnitudes)
    return vectors / magnitudes

def cdf_sampling(pdf_func, num_samples, m, mu, etotal):
    x_range = (0, etotal)
    x_values = np.linspace(x_range[0], x_range[1], 1000)
    pdf_values = pdf_func(x_values, m, mu, etotal)
    pdf_normalized = pdf_values / np.trapz(pdf_values, x_values)
    cdf_values = np.cumsum(pdf_normalized) * (x_values[1] - x_values[0])
    cdf_values /= cdf_values[-1]
    u = np.random.uniform(0, 1, num_samples)
    return np.interp(u, cdf_values, x_values)

def Gamma_ee(ep, m, mu, etotal):
    alpha = 1/137
    return alpha * mu**2 * m * etotal - 4 * (etotal - ep) * ep / (8 * np.pi**2 * etotal - m / 2)

def costheta_subtended(numtm, Nmtm, Eplus, Eminus):
    pNpnu = numtm[0]*Nmtm[0] - np.dot(numtm[1:], Nmtm[1:])
    pplus = np.sqrt(Eplus**2 - me**2)
    pminus = np.sqrt(Eminus**2 - me**2)
    return (Eplus*Eminus + pNpnu - m**2/2) / (pplus*pminus)

def apply_gaussian_smearing(values, relative_sigma):
    absolute_sigma = values * relative_sigma
    return np.maximum(np.random.normal(values, absolute_sigma), 0)

def apply_costheta_smearing(costheta, sigma):
    return np.clip(np.random.normal(costheta, sigma), -1, 1)

def process_event(etotal, weight, numtm, Nmtm, direction, m, mu, samples_per_etotal=1,
                  relative_sigma_plus=0.078, relative_sigma_minus=0.078, costheta_sigma=0.001):
    num_samples = int(samples_per_etotal * weight)
    samples_eplus = cdf_sampling(Gamma_ee, num_samples, m, mu, etotal)
    samples_eminus = etotal - samples_eplus
    
    smeared_eplus = apply_gaussian_smearing(samples_eplus, relative_sigma_plus)
    smeared_eminus = apply_gaussian_smearing(samples_eminus, relative_sigma_minus)
    
    samples_angle_original = costheta_subtended(numtm, Nmtm, smeared_eplus, smeared_eminus)
    samples_angle_smeared = apply_costheta_smearing(samples_angle_original, costheta_sigma)
    
    return smeared_eplus, smeared_eminus, samples_angle_smeared

def calculate_signal_strength(angles):
    return np.sum(np.vectorize(lambda x: x > np.cos(np.pi/2))(angles))

def calculate_rejection_power(signal, background):
    test_statistic = np.sqrt(2 * ((signal + background) * np.log(1 + signal/background) - signal))
    return 1 - stats.norm.cdf(-test_statistic)

def process_file(args):
    exp, m, mu = args
    path = f"./output/{exp}_Dipole_M{m:.2e}_mu{mu:.2e}_example.parquet"
    if not os.path.isfile(path):
        print(f'File {path} not found.')
        return None

    data = awk.from_parquet(path)
    
    dec_flag = data["primary_type"] == 5914
    fid_flag = data["in_fiducial"][dec_flag]
    weights = np.array(np.squeeze(data['event_weight'] * fid_flag * POT))
    
    nuout_flag = data["secondary_types"] == 5910
    nuout_momenta = np.squeeze(data["secondary_momenta"][nuout_flag][~awk.is_none(data["secondary_momenta"][nuout_flag], axis=1)])
    
    N_flag = data["primary_type"] == 5914
    N_momenta = np.squeeze(data["primary_momentum"][N_flag])
    
    gamma_flag = data["secondary_types"] == 22
    gamma_momenta = np.squeeze(data["secondary_momenta"][gamma_flag][~awk.is_none(data["secondary_momenta"][gamma_flag], axis=1)])
    
    gamma_ETotal = np.array(gamma_momenta[:, 0])
    gamma_direction = normalize_vectors(gamma_momenta[:, 1:])
    
    weights = np.vectorize(lambda x: x > 0.999)(N_momenta[:, 3] / np.linalg.norm(N_momenta[:, 1:], axis=1)) * weights
    upscale = np.minimum(1e5, 1 / np.min(weights[np.nonzero(weights)]))
    weights *= upscale

    all_angles = []
    for etotal, weight, numtm, Nmtm, direction in zip(gamma_ETotal, weights, nuout_momenta, N_momenta, gamma_direction):
        _, _, angles = process_event(etotal, weight, numtm, Nmtm, direction, m, mu)
        all_angles.extend(angles)

    all_angles = np.array(all_angles)
    signal_strength = calculate_signal_strength(all_angles)
    signal_strength = min_efficiency_interp(m * 1000) * signal_strength / upscale
    
    return exp, m, mu, signal_strength

def main():
    n_m, n_mu = 10, 10
    m_sample = np.geomspace(1e-2, 2, n_m)
    mu_sample = np.geomspace(1e-7, 1e-5, n_mu)
    m_sample, mu_sample = np.meshgrid(m_sample, mu_sample)
    m_sample = np.reshape(m_sample, [n_m * n_mu])
    mu_sample = np.reshape(mu_sample, [n_m * n_mu])

    experiments = ['ND280UPGRD','TPC2','TPC4','TPC5']
    
    # Create a list of all parameter combinations
    all_params = [(exp, m, mu) for exp in experiments for m, mu in zip(m_sample, mu_sample)]

    # Set up multiprocessing
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)

    # Process files in parallel
    results = list(tqdm(pool.imap(process_file, all_params), total=len(all_params), desc="Processing files"))

    # Close the pool
    pool.close()
    pool.join()

    # Filter out None results and write to file
    valid_results = [r for r in results if r is not None]
    with open('results.txt', 'w') as f:
        f.write("Experiment,m,mu,Signal,Background,Power\n")
        for result in valid_results:
            f.write(f"{result[0]},{result[1]:.6e},{result[2]:.6e},{result[3]:.6f}\n")

if __name__ == "__main__":
    main()