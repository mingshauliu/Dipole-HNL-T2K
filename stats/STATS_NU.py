import multiprocessing
import awkward as ak
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import os
import csv
from dataclasses import dataclass
from typing import Tuple, Optional, NamedTuple, List, Union
from pathlib import Path

# Remember to check output file's tag u/d and +/- of fractional error

@dataclass
class Constants:
    POT = {'ND280+': 2.6098758621e22, 'ND280NU': 3.6e21}
    ELECTRON_MASS = 0.511e-3  # GeV/c^2
    DEFAULT_TARGET_SUM = 1e5
    DATA_DIR = Path('../MC/output_nu_NEW')
    ERROR_BOOL = True

class ParticleData(NamedTuple):
    """Container for particle-related data"""
    weights: np.ndarray
    nuout_momenta: np.ndarray
    N_momenta: np.ndarray
    gamma_ETotal: np.ndarray
    gamma_direction: np.ndarray

def create_efficiency_interpolator(
    filename: str = "kaon_decay_efficiency_data.txt",
    skip_rows: int = 2,
    m_n_min: float = 10,
    m_n_max: float = 2000,
    m_n_step: float = 25
) -> interp1d:
    """Create an interpolation function for minimum efficiency."""
    m_n, mu_decay, e_decay = np.loadtxt(filename, skiprows=skip_rows, usecols=(0, 1, 2), unpack=True)
    min_decay = np.minimum(mu_decay, e_decay)
    extended_m_n = np.arange(m_n_min, m_n_max + m_n_step, m_n_step)
    
    # Interpolate values
    extended_min = np.interp(extended_m_n, m_n, min_decay)
    return interp1d(extended_m_n, extended_min, kind='linear', bounds_error=False, fill_value='extrapolate')

def create_error_interpolator(
    filename: str = "numu_flux_error.txt",
    skip_rows: int = 0,
    m_n_min: float = 10,
    m_n_max: float = 2000,
    m_n_step: float = 25
) -> interp1d:
    """Create an interpolation function for minimum efficiency."""
    m_n, err_frac = np.loadtxt(filename, skiprows=skip_rows, usecols=(0, 1), unpack=True)
    extended_m_n = np.arange(m_n_min, m_n_max + m_n_step, m_n_step)
    
    # Interpolate values
    extended_min = np.interp(extended_m_n, m_n, err_frac)
    return interp1d(extended_m_n, extended_min, kind='linear', bounds_error=False, fill_value='extrapolate')

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors efficiently handling zero magnitudes."""
    # Handle zero magnitudes
    magnitudes = np.linalg.norm(vectors, axis=1)
    non_zero = magnitudes != 0
    normalized = np.zeros_like(vectors)
    normalized[non_zero] = vectors[non_zero] / magnitudes[non_zero, np.newaxis]
    return normalized

def process_particle_data(data: ak.Array, type_key: str, type_value: int, momentum_key: str) -> np.ndarray:
    """Process particle data based on type and momentum keys."""
    flag = data[type_key] == type_value
    momenta = data[momentum_key][flag]
    valid_momenta = ak.mask(momenta, ak.num(momenta, axis=2) > 0)
    converted_momenta = ak.to_numpy(valid_momenta[~ak.is_none(valid_momenta, axis=1)])
    return np.squeeze(converted_momenta)

def load_data(exp: str, m: float, mu: float, POT: dict) -> Optional[ParticleData]:
    """Load and process particle data from parquet file."""
    path = Constants.DATA_DIR / f"{exp}_Dipole_M{m:2.2e}_mu{mu:2.2e}_example.parquet"
    
    try:
        if not path.exists():
            print(f'File {path} not found.')
            return None
            
        data = ak.from_parquet(str(path))
        
        # Apply masks and filters
        dec_flag = ak.to_numpy(data["primary_type"] == 5914)
        fid_flag = ak.to_numpy(data["in_fiducial"][dec_flag])
        
        # Convert to numpy arrays
        weights = ak.to_numpy(data['event_weight']) * fid_flag * POT['ND280NU']
        weights = np.squeeze(weights)
        
        # Process particle data
        N_momenta = ak.to_numpy(data["primary_momentum"][data["primary_type"] == 5914])
        N_momenta = np.squeeze(N_momenta)
        
        # Process gamma data
        gamma_momenta = process_particle_data(data, "secondary_types", 22, "secondary_momenta")
        if len(gamma_momenta) == 0:
            return None
            
        gamma_ETotal = gamma_momenta[:, 0]
        gamma_direction = normalize_vectors(gamma_momenta[:, 1:])
        
        nuout_momenta = process_particle_data(data, "secondary_types", 5910, "secondary_momenta")
        
        return ParticleData(
            weights=weights,
            nuout_momenta=nuout_momenta,
            N_momenta=N_momenta,
            gamma_ETotal=gamma_ETotal,
            gamma_direction=gamma_direction
        )
        
    except Exception as e:
        print(f'Error processing file {path}: {str(e)}')
        return None

def apply_cuts(particle_data: ParticleData) -> np.ndarray:
    """Apply physics cuts to the data."""
    if len(particle_data.N_momenta) == 0 or len(particle_data.gamma_ETotal) == 0:
        return np.array([])
        
    momentum_cut = particle_data.N_momenta[:, 3] / np.linalg.norm(particle_data.N_momenta[:, 1:], axis=1) > 0.999
    energy_cut = particle_data.gamma_ETotal > 2 * Constants.ELECTRON_MASS
    return particle_data.weights * momentum_cut * energy_cut

def write_signal_strength(
    exp: str,
    i: int,
    m: float,
    mu: float,
    signal_strength: float,
    filename='STATS_NU_ERRd.csv'
) -> None:
    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.isfile(filename)
    
    # Open the file in append mode
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["Experiment", "i", "Mass", "Coupling", "Signal_Strength"])

        writer.writerow([exp,f"{i}",f"{m:.6e}",f"{mu:.6e}",f"{signal_strength:.32f}"])

def process_experiment(args: Tuple) -> None:
    """Process a single experiment with given parameters."""
    exp, i, m, mu, POT, error_flag = args
    print(f'Processing: {exp}, HNL mass {m} GeV, Dipole Strength mu {mu}')
    
    # Load and process data
    particle_data = load_data(exp, m, mu, POT)
    if particle_data is None or np.sum(particle_data.weights) == 0:
        signal_strength = 0
    else:
        weights = apply_cuts(particle_data)
        if len(weights) == 0:
            signal_strength = 0
        else:
            if error_flag:
                signal_strength = get_min_efficiency(m * 1000) * np.sum(weights) * (1 - get_flux_error(m * 1000))
            else:
                signal_strength = get_min_efficiency(m * 1000) * np.sum(weights)
            print(f"Final Signal Strength: {signal_strength:.1f}")
    
    # Write results to CSV
    write_signal_strength(exp, i, m, mu, signal_strength)

def get_unprocessed_experiments(
    filename: str,
    experiments: List[str],
    index: np.ndarray,
    m_sample: np.ndarray,
    mu_sample: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get list of unprocessed experiments."""
    if os.path.isfile(filename):
        df = pd.read_csv(filename, usecols=['Experiment', 'i'])
        processed = {f"{exp}_{i}" for exp, i in zip(df['Experiment'], df['i'])}
        all_exp = {f"{exp}_{i}" for exp in experiments for i in index}
        unprocessed = np.array(list(all_exp - processed))
    else:
        unprocessed = np.array([f"{exp}_{i}" for exp in experiments for i in index])
    
    # Split and process unprocessed experiments
    split_arrays = np.char.split(unprocessed, sep='_')
    exp_array = np.array([item[0] for item in split_arrays])
    idx_array = np.array([int(item[1]) for item in split_arrays])
    
    return exp_array, idx_array, m_sample[idx_array], mu_sample[idx_array]

def main():
    # Initialize parameters
    n_m, n_mu = 10, 10
    m_sample = np.geomspace(1e-2, 2, n_m)
    mu_sample = np.geomspace(1e-7, 1e-5, n_mu)
    m_sample, mu_sample = np.meshgrid(m_sample, mu_sample)
    m_sample = m_sample.ravel()
    mu_sample = mu_sample.ravel()
    
    experiments = ['OGTPC1', 'OGTPC2', 'OGTPC3']
    index = np.arange(100)
    
    # Get unprocessed experiments
    exp_array, idx_array, ms, mus = get_unprocessed_experiments(
        'STATS_NU_ERRd.csv', 
        experiments, 
        index,
        m_sample,
        mu_sample
    )
    
    # Prepare multiprocessing arguments
    args_list = [(exp, i, m, mu, Constants.POT, Constants.ERROR_BOOL) 
                 for exp, i, m, mu in zip(exp_array, idx_array, ms, mus)]
    
    # Run processing in parallel
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    with multiprocessing.Pool(n_cpus) as pool:
        list(tqdm(pool.imap(process_experiment, args_list), 
                 total=len(args_list), 
                 desc="Processing Experiments"))

if __name__ == '__main__':
    # Create efficiency interpolator once
    get_min_efficiency = create_efficiency_interpolator()
    get_flux_error = create_error_interpolator()
    main()