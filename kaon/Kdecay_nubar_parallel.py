import numpy as np
import os
import siren
from siren.SIREN_Controller import SIREN_Controller
import multiprocessing as mp
from functools import partial
import logging
from datetime import datetime
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hnl_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def generate_hnl_events(m, U, events_to_inject=100000, experiment="OGTPC1"):
    """
    Generate HNL events for given mass and coupling parameters.
    
    Args:
        m (float): Mass parameter
        U (float): Coupling parameter
        events_to_inject (int): Number of events to generate
        experiment (str): Experiment name
    
    Returns:
        str: Status message
    """
    try:
        path = f"OGTPC1_Dipole_M{m:.2e}_Ue{U:.2e}_example.parquet"
        if os.path.isfile(f"output_nu/{path}"):
            return f"File {path} already exists"

        model_kwargs = {
            "m4": m,
            "mu_tr_mu4": 1e-6,
            "Ue4": U,
            "Umu4": 0,
            "epsilon": 0.0,
            "gD": 0.0,
            "decay_product": "photon",
            "noHC": True,
            "HNLtype": "dirac",
        }

        controller = SIREN_Controller(events_to_inject, experiment)
        primary_type = siren.dataclasses.Particle.ParticleType.N4
        xs_path = siren.utilities.get_cross_section_model_path(
            f"DarkNewsTables-v{siren.utilities.darknews_version()}", 
            must_exist=False
        )
        table_dir = os.path.join(xs_path, f"Dipole_M{m:.2e}_Ue{U:.2e}")
        
        controller.InputDarkNewsModel(primary_type, table_dir, **model_kwargs)
        
        # Set up distributions
        flux_file = siren.utilities.get_tabulated_flux_file("N_Inject", "numubar")
        mdist = siren.distributions.PrimaryMass(m)
        edist = siren.distributions.TabulatedFluxDistribution(flux_file, True)
        edist_gen = siren.distributions.TabulatedFluxDistribution(m * 1.01, 30, flux_file, False)
        direction_distribution = siren.distributions.FixedDirection(siren.math.Vector3D(0, 0, 1.0))
        position_distribution = siren.distributions.CylinderVolumePositionDistribution(
            siren.geometry.Cylinder(3.8, 0, 7.6)
        )

        # Configure distributions
        primary_injection_distributions = {
            "mass": mdist,
            "energy": edist_gen,
            "direction": direction_distribution,
            "position": position_distribution
        }
        
        primary_physical_distributions = {
            "mass": mdist,
            "energy": edist,
            "direction": direction_distribution
        }

        controller.SetProcesses(
            primary_type, 
            primary_injection_distributions, 
            primary_physical_distributions
        )
        
        controller.Initialize()
        logging.info(f"Initialization complete for m={m:.2e}, U={U:.2e}")
        logging.info(f"Minimum decay width: {controller.DN_min_decay_width}")

        events = controller.GenerateEvents(fill_tables_at_exit=False)
        
        os.makedirs("output_nu", exist_ok=True)
        controller.SaveEvents(
            f"output_nu/{experiment}_Dipole_M{m:.2e}_Ue{U:.2e}_example",
            fill_tables_at_exit=True,
            save_int_probs=True
        )
        
        return f"Completed: m={m:.2e}, U={U:.2e}"
    
    except Exception as e:
        error_msg = f"Error processing m={m:.2e}, U={U:.2e}: {str(e)}"
        logging.error(error_msg)
        return error_msg

def chunk_parameters(m_sample, U_sample, chunk_size=10):
    """Split parameter space into chunks for batch processing."""
    total_points = len(m_sample)
    for i in range(0, total_points, chunk_size):
        yield (
            m_sample[i:i + chunk_size],
            U_sample[i:i + chunk_size]
        )

def process_chunk(m_chunk, U_chunk):
    """Process a chunk of parameters in parallel."""
    with mp.Pool() as pool:
        results = pool.starmap(generate_hnl_events, zip(m_chunk, U_chunk))
    return results

def main():
    # Parameter space setup
    n_m, n_U = 20, 20
    m_sample = np.geomspace(2e-2, 4e-1, n_m)
    U_sample = np.geomspace(1e-7, 1e-2, n_U)
    m_sample, U_sample = np.meshgrid(m_sample, U_sample)
    m_sample = m_sample.flatten()
    U_sample = U_sample.flatten()

    # Create output directory
    os.makedirs("output_nu", exist_ok=True)
    
    # Determine optimal chunk size based on available CPUs
    num_cores = mp.cpu_count() - 1  # Leave one core free
    chunk_size = max(1, min(20, len(m_sample) // (num_cores * 2)))  # Ensure reasonable chunk size
    
    logging.info(f"Starting parameter space exploration with {num_cores} cores")
    logging.info(f"Total parameter points: {len(m_sample)}")
    logging.info(f"Chunk size: {chunk_size}")
    
    # Process parameters in chunks
    start_time = datetime.now()
    completed = 0
    
    for m_chunk, U_chunk in chunk_parameters(m_sample, U_sample, chunk_size):
        results = process_chunk(m_chunk, U_chunk)
        completed += len(results)
        
        # Log progress
        elapsed = datetime.now() - start_time
        progress = completed / len(m_sample) * 100
        logging.info(f"Progress: {progress:.1f}% ({completed}/{len(m_sample)})")
        logging.info(f"Time elapsed: {elapsed}")
        
        # Log results
        for result in results:
            logging.info(result)
    
    total_time = datetime.now() - start_time
    logging.info(f"Parameter space exploration completed in {total_time}")

if __name__ == "__main__":
    main()