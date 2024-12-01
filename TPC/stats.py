import numpy as np
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
import awkward as ak

# Load efficiency data
data = np.loadtxt("kaon_decay_efficiency_data.txt", skiprows=2)
m_n, mu_decay, e_decay = data[:, 0], data[:, 1], data[:, 2]
min_decay = np.minimum(mu_decay, e_decay)
extended_m_n = np.arange(150, 2001, 25)
extended_min = np.interp(extended_m_n, m_n, min_decay)
min_efficiency_interp = interp1d(extended_m_n, extended_min, kind='linear', fill_value='extrapolate')

def get_min_efficiency(m_n_value):
    return min_efficiency_interp(m_n_value)

# Constants
POT = 2.6098758621e22
me = 0.511e-3

def normalize_vectors(vectors):
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    magnitudes = np.where(magnitudes == 0, 1, magnitudes)
    return vectors / magnitudes

def cdf_sampling(pdf_func, num_samples, m, mu, etotal):
    x_values = np.linspace(0, etotal, 1000)
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
    return (Eplus*Eminus + pNpnu - m**2/2)/(pplus*pminus)

def apply_gaussian_smearing(values, relative_sigma):
    absolute_sigma = values * relative_sigma
    smeared_values = np.random.normal(values, absolute_sigma)
    return np.maximum(smeared_values, 0)

def apply_costheta_smearing(costheta, sigma):
    smeared_costheta = np.random.normal(costheta, sigma)
    return np.clip(smeared_costheta, -1, 1)

def calculate_signal_strength(m, mu, exp='TPC2'):
    path = f"{exp}_Dipole_M{m:.2e}_mu{mu:.2e}_example.parquet"
    
    if not os.path.isfile('./output/'+path):
        print(f'File {path} not found.')
        return None

    data = ak.from_parquet("output/"+path)
    
    dec_flag = data["primary_type"] == 5914
    fid_flag = data["in_fiducial"][dec_flag]
    weights = np.array(np.squeeze(data['event_weight'] * fid_flag * POT))
    
    N_momenta = np.squeeze(data["primary_momentum"][dec_flag])
    weights = np.vectorize(lambda x: x > 0.999)(N_momenta[:,3] / np.linalg.norm(N_momenta[:,1:], axis=1)) * weights
    
    upscale = np.minimum(1e4,1/np.min(weights[np.nonzero(weights)]))
    weights *= upscale

    nuout_flag = data["secondary_types"] == 5910
    nuout_momenta = data["secondary_momenta"][nuout_flag]
    nuout_momenta = ak.mask(nuout_momenta, ak.num(nuout_momenta, axis=2) > 0)
    nuout_momenta = np.squeeze(nuout_momenta[~ak.is_none(nuout_momenta, axis=1)])

    gamma_flag = data["secondary_types"] == 22
    gamma_momenta = data["secondary_momenta"][gamma_flag]
    gamma_momenta = ak.mask(gamma_momenta, ak.num(gamma_momenta, axis=2) > 0)
    gamma_momenta = np.squeeze(gamma_momenta[~ak.is_none(gamma_momenta, axis=1)])
    
    gamma_ETotal = np.array(gamma_momenta[:,0])
    gamma_direction = normalize_vectors(gamma_momenta[:,1:])

    samples_per_etotal = 1
    relative_sigma_plus = 0.078
    relative_sigma_minus = 0.078
    costheta_sigma = 0.001

    all_samples_angle = []

    for etotal, weight, numtm, Nmtm, direction in zip(gamma_ETotal, weights, nuout_momenta, N_momenta, gamma_direction):
        num_samples = int(samples_per_etotal * weight)
        samples_eplus = cdf_sampling(Gamma_ee, num_samples, m, mu, etotal)
        samples_eminus = etotal - samples_eplus
        
        smeared_eplus = apply_gaussian_smearing(samples_eplus, relative_sigma_plus)
        smeared_eminus = apply_gaussian_smearing(samples_eminus, relative_sigma_minus)
        
        samples_angle_original = costheta_subtended(numtm, Nmtm, smeared_eplus, smeared_eminus)
        samples_angle_smeared = apply_costheta_smearing(samples_angle_original, costheta_sigma)
        
        all_samples_angle.extend(samples_angle_smeared)

    all_samples_angle = np.array(all_samples_angle)
    signal_strength = np.sum(np.vectorize(lambda x: x > np.cos(np.pi/2))(all_samples_angle))
    signal_strength = get_min_efficiency(m*1000) * signal_strength / upscale

    return signal_strength

# Generate sample points
n_m, n_mu = 10, 10
m_sample = np.geomspace(1e-2, 2, n_m)
mu_sample = np.geomspace(1e-7, 1e-5, n_mu)
m_sample, mu_sample = np.meshgrid(m_sample, mu_sample)
m_sample = np.reshape(m_sample, [n_m*n_mu])
mu_sample = np.reshape(mu_sample, [n_m*n_mu])

# Calculate signal strength for each point
output_file = f"signal_strengths_{experiment}.txt"
with open(output_file, "w") as f:
    f.write("index,m,mu,signal_strength\n")  # Write header

for i in tqdm(range(100), desc="Calculating signal strengths"):
    m, mu = m_sample[i], mu_sample[i]
    signal_strength = calculate_signal_strength(m, mu)
    
    # Save the result to the file
    with open(output_file, "a") as f:
        f.write(f"{i},{m},{mu},{signal_strength}\n")

print(f"Signal strengths calculated and saved to {output_file}.")