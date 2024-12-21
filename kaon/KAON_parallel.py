import numpy as np
import os
import siren
from siren.SIREN_Controller import SIREN_Controller
import multiprocessing as mp
from functools import partial
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hnl_generation.log'),
        logging.StreamHandler()
    ]
)

def generate_hnl_events(params, events_to_inject=100000, experiment="OGTPC3"):
    """
    Generate HNL events for given mass and coupling parameters.
    
    Args:
        params (tuple): Tuple containing (mass, coupling) values
        events_to_inject (int): Number of events to generate
        experiment (str): Experiment identifier
    """
    m, U = params
    try:
        path = f"test/OGTPC3_Dipole_M{m:.2e}_Umu{U:.2e}_example.parquet"
        
        if os.path.isfile(path):
            logging.info(f"File {path} already exists - skipping")
            return f"Skipped: m={m:.2e}, U={U:.2e} (file exists)"

        model_kwargs = {
            "m4": m,
            "mu_tr_mu4": 1e-6,
            "UD4": 0,
            "Umu4": U,
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
        table_dir = os.path.join(xs_path, f"Dipole_M{m:.2e}_Umu{U:.2e}")
        
        controller.InputDarkNewsModel_PrimaryDecay(primary_type, table_dir, use_pickles=False, **model_kwargs)
        
        # Set up distributions
        flux_file = siren.utilities.get_tabulated_flux_file("N_Inject", "numu_PLUS")
        mdist = siren.distributions.PrimaryMass(m)
        edist = siren.distributions.TabulatedFluxDistribution(flux_file, True)
        edist_gen = siren.distributions.TabulatedFluxDistribution(m * 1.01, 30, flux_file, False)
        direction_distribution = siren.distributions.FixedDirection(
            siren.math.Vector3D(0, 0, 1.0)
        )
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
        
        os.makedirs("test", exist_ok=True)
        controller.SaveEvents(
            f"test/{experiment}_Dipole_M{m:.2e}_Umu{U:.2e}_example",
            fill_tables_at_exit=True,
            save_int_probs=True
        )
        
        return f"Completed: m={m:.2e}, U={U:.2e}"
        
    except Exception as e:
        logging.error(f"Error processing m={m:.2e}, U={U:.2e}: {str(e)}")
        return f"Failed: m={m:.2e}, U={U:.2e} - {str(e)}"

def main():
    # Generate parameter grid
    n_m, n_U = 20, 20
    m_sample = np.geomspace(2e-2, 4e-1, n_m)
    U_sample = np.geomspace(1e-7, 1e-2, n_U)
    m_sample, U_sample = np.meshgrid(m_sample, U_sample)
    
    # Create parameter pairs
    param_pairs = list(zip(m_sample.flatten(), U_sample.flatten()))
    
    # Configure multiprocessing
    num_cores = max(mp.cpu_count() - 1, 1)  # Leave one core free, minimum 1
    logging.info(f"Using {num_cores} CPU cores for parallel processing")
    
    # Create a pool of worker processes with progress bar
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(
            pool.imap(generate_hnl_events, param_pairs),
            total=len(param_pairs),
            desc="Generating HNL events")
        )
    
    # Log results
    successful = len([r for r in results if r.startswith("Completed")])
    skipped = len([r for r in results if r.startswith("Skipped")])
    failed = len([r for r in results if r.startswith("Failed")])
    
    logging.info(f"""
    Processing complete:
    - Total parameter points: {len(param_pairs)}
    - Successful: {successful}
    - Skipped: {skipped}
    - Failed: {failed}
    """)

if __name__ == "__main__":
    main()