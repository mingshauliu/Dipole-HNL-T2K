import multiprocessing
import awkward as ak
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import csv
import os

TPC_boxes = {
    'OGTPC1':[0, 0, -0.375, 1.7, 1.96, 0.563],
    'OGTPC2':[0, 0, 0.99, 1.7, 1.96, 0.563],
    'OGTPC3':[0, 0, 2.355, 1.7, 1.96, 0.563],
    'ND280UPGRD':[0, 0, 1.035, 1.7, 1.96, 0.563],
    'TPC2':[0, 0, 2.4, 1.7, 1.96, 0.563],
    'TPC3':[0, 0, 3.765, 1.7, 1.96, 0.563],
    'TPC4':[0, 0.74, -0.795, 1.592, 0.27, 1.709],
    'TPC5':[0, -0.74, -0.795, 1.592, 0.27, 1.709]
} # TPC and HATPC dimensions in ND280 (OGTPC) and ND280+ (TPC)


def is_points_in_boxes(points, boxes):
    points = np.asarray(points)
    boxes = np.asarray(boxes)
    centers, dimensions = boxes[:3], boxes[3:]
    
    # Broadcasting to compare all points with all boxes
    in_bounds = np.all(
        (points[:, np.newaxis, :] >= centers - dimensions/2) & 
        (points[:, np.newaxis, :] <= centers + dimensions/2),
        axis=2
    )

    # Check if each point is in any box
    return np.any(in_bounds, axis=1)

def create_get_min_efficiency(filename="kaon_decay_efficiency_data.txt", skip_rows=2, m_n_min=10, m_n_max=2000, m_n_step=25):
    m_n, mu_decay, e_decay = np.loadtxt(filename, skiprows=skip_rows, usecols=(0, 1, 2), unpack=True)
    min_decay = np.minimum(mu_decay, e_decay)
    extended_m_n = np.arange(m_n_min, m_n_max + m_n_step, m_n_step)
    
    # Perform interpolation only where necessary
    mask = (extended_m_n < m_n[0]) | (extended_m_n > m_n[-1])
    extended_min = np.interp(extended_m_n, m_n, min_decay)
    
    interp_func = interp1d(m_n, min_decay, kind='linear', bounds_error=False, fill_value=(min_decay[0], min_decay[-1]))
    extended_min[~mask] = interp_func(extended_m_n[~mask])
    
    # Create final interpolation function
    min_efficiency_interp = interp1d(extended_m_n, extended_min, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    return min_efficiency_interp

get_min_efficiency = create_get_min_efficiency()

def normalize_vectors(vectors):
    vectors_array = np.asarray(vectors, dtype=np.float64)
    magnitudes = np.linalg.norm(vectors_array, axis=1)
    
    # Use boolean indexing to handle zero magnitudes
    non_zero = magnitudes != 0
    vectors_array[non_zero] /= magnitudes[non_zero, np.newaxis]
    
    return vectors_array

def cdf_sampling(pdf_func, num_samples, m, mu, etotal):
    # Define Energy Sampling Range
    x_range = (me, etotal)
    # Step 1: Create a fine grid for x values
    x_values = np.linspace(x_range[0], x_range[1], 1000)
    # Step 2: Calculate PDF values
    pdf_values = pdf_func(x_values, m, mu, etotal)
    # Step 3: Normalize the PDF
    pdf_normalized = pdf_values / np.trapz(pdf_values, x_values)
    # Step 4: Calculate the CDF using cumulative sum
    cdf_values = np.cumsum(pdf_normalized) * (x_values[1] - x_values[0])
    cdf_values /= cdf_values[-1]  # Ensure the CDF ends at 1
    # Step 5, 6: Random u & Inverse transform sampling
    return np.interp(np.random.uniform(0, 1, num_samples), cdf_values, x_values)

# Branching ratio of ep = E+ and em = E-, etotal = ep + em
def Gamma_ee(ep, m, mu, etotal):
    alpha = 1/137  # Fine-structure constant
    return alpha * mu**2 * m * etotal - 4 * (etotal - ep) * ep / (8 * np.pi**2 * etotal - m / 2)

def costheta_subtended(m, numtm, Nmtm, Eplus, Eminus):
    pNpnu = numtm[0] * Nmtm[0] - np.dot(numtm[1:], Nmtm[1:])
    Eplus_sq = Eplus**2
    Eminus_sq = Eminus**2
    pplus_pminus = np.sqrt((Eplus_sq - me**2) * (Eminus_sq - me**2))
    return (Eplus * Eminus + pNpnu - 0.5 * m**2) / pplus_pminus


def apply_gaussian_smearing(values, relative_sigma):
    absolute_sigma = values * relative_sigma
    smeared_values = np.random.normal(values, absolute_sigma)
    return np.maximum(smeared_values, 0)  # Ensure non-negative energies

def apply_costheta_smearing(costheta, sigma):
    smeared_costheta = np.random.normal(costheta, sigma)
    return np.clip(smeared_costheta, -1, 1) # Clip to cosθ range

def load_data(exp, m, mu, POT):
    path = f"{exp}_Dipole_M{m:2.2e}_mu{mu:2.2e}_example.parquet"
    root_dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/msliu/output_final_bar/'
    if os.path.isfile(root_dir + path):
        data = ak.from_parquet(root_dir + path)
        
        weights = np.array(np.squeeze(data['event_weight'] * is_points_in_boxes(data["vertex"][:,1],TPC_boxes[exp]) * (POT['ND280+'])))
        
        nuout_momenta = process_particle_data(data, "secondary_types", 5910, "secondary_momenta")
        N_momenta = np.squeeze(data["primary_momentum"][data["primary_type"] == 5914])
        gamma_momenta = process_particle_data(data, "secondary_types", 22, "secondary_momenta")
        
        gamma_ETotal = np.array(gamma_momenta[:, 0])
        gamma_direction = normalize_vectors(gamma_momenta[:, 1:])
        
        return weights, nuout_momenta, N_momenta, gamma_ETotal, gamma_direction
    else:
        print('File ' + path + ' not found.')
        return None, None, None, None, None
    
def process_particle_data(data, type_key, type_value, momentum_key):
    flag = data[type_key] == type_value
    momenta = data[momentum_key][flag]
    momenta = ak.mask(momenta, ak.num(momenta, axis=2) > 0)
    return np.squeeze(momenta[~ak.is_none(momenta, axis=1)])

def apply_cuts(weights, N_momenta, gamma_ETotal):
    precut_weights = weights
    weights = np.vectorize(lambda x: x > 0.999)(N_momenta[:, 3] / np.linalg.norm(N_momenta[:, 1:], axis=1)) * weights
    weights = np.vectorize(lambda x: x > 2*me)(gamma_ETotal) * weights
    # print(f'Weights before HNL incident cut: {np.sum(precut_weights)}')
    # print(f'Weights after HNL incident cut: {np.sum(weights)}')
    # print(f'Cut 1 survival rate: {np.sum(weights) / np.sum(precut_weights) * 100:f} %')
    return weights

def scale_weights(weights, target_sum=1e5):
    original_weights = weights
    weightscale = target_sum / np.sum(weights)
    weights *= weightscale
    # print(f'Weights scaled by {weightscale:3f} to reach target weight {target_sum:3f}')
    return weights, weightscale

def generate_samples(gamma_ETotal, weights, nuout_momenta, N_momenta, gamma_direction, m, mu,
                     samples_per_etotal=1, relative_sigma_plus=0.078, relative_sigma_minus=0.078, costheta_sigma=0.01):
    
    total_samples = int(np.sum(weights) * samples_per_etotal)
    
    # Pre-allocate arrays
    all_samples = {
        'eplus_original': np.zeros(total_samples),
        'eminus_original': np.zeros(total_samples),
        'eplus': np.zeros(total_samples),
        'eminus': np.zeros(total_samples),
        'angle_original': np.zeros(total_samples),
        'angle': np.zeros(total_samples),
        'gamma': np.zeros(total_samples)
    }
    
    index = 0
    for etotal, weight, numtm, Nmtm, direction in tqdm(zip(gamma_ETotal, weights, nuout_momenta, N_momenta, gamma_direction), desc='Events Generated', total=100000):
        num_samples = int(samples_per_etotal * weight)
        if num_samples == 0:
            continue
        
        samples_eplus = cdf_sampling(Gamma_ee, num_samples, m, mu, etotal)
        samples_eminus = etotal - samples_eplus
        
        all_samples['eplus_original'][index:index+num_samples] = samples_eplus
        all_samples['eminus_original'][index:index+num_samples] = samples_eminus
        
        smeared_eplus = apply_gaussian_smearing(samples_eplus, relative_sigma_plus)
        smeared_eminus = apply_gaussian_smearing(samples_eminus, relative_sigma_minus)
        
        all_samples['eplus'][index:index+num_samples] = smeared_eplus
        all_samples['eminus'][index:index+num_samples] = smeared_eminus
        all_samples['gamma'][index:index+num_samples] = smeared_eplus + smeared_eminus
        
        samples_angle_original = costheta_subtended(m, numtm, Nmtm, smeared_eplus, smeared_eminus)
        all_samples['angle_original'][index:index+num_samples] = samples_angle_original
        
        all_samples['angle'][index:index+num_samples] = apply_costheta_smearing(samples_angle_original, costheta_sigma)
        
        index += num_samples
    
    # Trim arrays to actual size used
    for key in all_samples:
        all_samples[key] = all_samples[key][:index]
    
    all_samples['diff'] = all_samples['eplus'] - all_samples['eminus']
    
    return all_samples

def calculate_signal_strength(all_samples, weightscale, m):
    # signal_strength = np.sum(np.vectorize(lambda x: x > np.cos(np.pi/2))(all_samples['angle']))
    signal_strength = len(all_samples['angle'])
    # print(f"Weights before pair production angle cut: {len(all_samples['angle']) / weightscale:.1f}")
    # print(f"Weights after pair production angle cut: {signal_strength / weightscale:.1f}")
    # print(f"Cut 1 | Cut 2 survival rate: {signal_strength / len(all_samples['angle']) * 100:.1f} %")
    # print(f"Cut 3 Mass recon survival rate: {get_min_efficiency(m * 1000) * 100:.1f} %")
    return get_min_efficiency(m * 1000) * signal_strength / weightscale

def write_signal_strength(exp, i, m, mu, signal_strength, filename="signal_strengths_nd280_nubar.csv"):
    
    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.isfile(filename)
    
    # Open the file in append mode
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["Experiment", "i", "Mass", "Coupling", "Signal_Strength"])

        writer.writerow([exp,f"{i}",f"{m:.6e}",f"{mu:.6e}",f"{signal_strength:.6f}"])

def run_experiment(args):
    exp, i, m, mu, POT = args
    print(f'Experiment: {exp}, Parameters: HNL mass {m} GeV, Dipole Strength mu {mu}')
   
    weights, nuout_momenta, N_momenta, gamma_ETotal, gamma_direction = load_data(exp, m, mu, POT)
    if weights is None:
        return
    
    weights = apply_cuts(weights, N_momenta, gamma_ETotal)
    weights, weightscale = scale_weights(weights)
    
    all_samples = generate_samples(gamma_ETotal, weights, nuout_momenta, N_momenta, gamma_direction, m, mu)
    
    signal_strength = calculate_signal_strength(all_samples, weightscale, m)
    print(f"Final Signal Strength: {signal_strength:.1f}")
    
    write_signal_strength(exp, i, m, mu, signal_strength)

def main():

    n_m = 10
    n_mu = 10
    m_sample = np.geomspace(1e-2,2,n_m)
    mu_sample = np.geomspace(1e-7,1e-5,n_mu)
    m_sample, mu_sample = np.meshgrid(m_sample, mu_sample)
    m_sample = np.reshape(m_sample,[n_m*n_mu])
    mu_sample = np.reshape(mu_sample,[n_m*n_mu])
    
    # Target signals
    index = np.arange(0,100)
    experiments = ['ND280UPGRD','TPC2', 'TPC3', 'TPC4','TPC5']
    # experiments = ['OGTPC1', 'OGTPC2', 'OGTPC3']
    
    filename = 'signal_strengths_nd280+_nubar.csv'
    
    if os.path.isfile(filename):
        # Read processed signals
        df = pd.read_csv(filename, usecols=['Experiment', 'i'])
        processed_list = np.array([f"{exp}_{i}" for exp, i in zip(df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist())])
        
        # Find unprocessed signals
        unprocessed_list = np.setdiff1d(np.array([f"{exp}_{i}" for exp in experiments for i in index]),processed_list)
        
    else:
        unprocessed_list = np.array([f"{exp}_{i}" for exp in experiments for i in index])
        
    split_arrays = np.char.split(unprocessed_list, sep='_')
    experiment_array = np.array([item[0] for item in split_arrays])
    index_array = np.array([item[1] for item in split_arrays], dtype=int)
    
    ms = m_sample[index_array]
    mus = mu_sample[index_array]

    # Prepare arguments for multiprocessing
    args_list = [(exp, i, m, mu, POT) for exp, i, m, mu in zip(experiment_array, index_array, ms, mus)]

    # Determine the number of CPUs to use
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    # Use multiprocessing to run experiments in parallel
    with multiprocessing.Pool(num_cpus) as pool:
        list(tqdm(pool.imap(run_experiment, args_list), total=len(args_list), desc="Overall Progress"))

if __name__=='__main__':
    ## Constants
    POT = {'ND280+':2.6098758621e22, 'ND280NU':3.6e21}
    me = 0.511e-3 # Electron mass in GeV/c^2
    main()