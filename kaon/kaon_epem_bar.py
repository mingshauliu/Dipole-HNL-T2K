import numpy as np
import os
import siren
from siren.SIREN_Controller import SIREN_Controller
import multiprocessing as mp
from functools import partial

def generate_hnl_events(m, U, events_to_inject=100000, experiment="OGTPC1"):
    """
    Generate Heavy Neutral Lepton (HNL) events using SIREN.
    
    Args:
        m (float): Mass parameter
        U (float): Coupling parameter
        events_to_inject (int): Number of events to generate
        experiment (str): Experiment identifier
        
    Returns:
        str: Status message indicating completion or existing file
    """
    path = f"OGTPC1_Dipole_M{m:.2e}_Umu{U:.2e}_example.parquet"
    
    if os.path.isfile(f"output_epem_bar/{path}"):
        return f"File {path} already exists"
    
    # Model configuration
    model_kwargs = {
        "m4": m,
        "mu_tr_mu4": 2e-6,
        "UD4": 0,
        "Umu4": U,
        "epsilon": 0.0,
        "gD": 0.0,
        "decay_product": "e+e-",
        "noHC": True,
        "HNLtype": "dirac",
    }
    
    # Initialize SIREN controller
    controller = SIREN_Controller(events_to_inject, experiment)
    primary_type = siren.dataclasses.Particle.ParticleType.N4
    
    # Set up cross-section paths
    xs_path = siren.utilities.get_cross_section_model_path(
        f"DarkNewsTables-v{siren.utilities.darknews_version()}", 
        must_exist=False
    )
    table_dir = os.path.join(xs_path, f"Dipole_M{m:.2e}_Umu{U:.2e}")
    controller.InputDarkNewsModel(primary_type, table_dir, **model_kwargs)
    
    # Configure distributions
    flux_file = siren.utilities.get_tabulated_flux_file("N_Inject", "numubar_MINUS")
    mdist = siren.distributions.PrimaryMass(m)
    edist = siren.distributions.TabulatedFluxDistribution(flux_file, True)
    edist_gen = siren.distributions.TabulatedFluxDistribution(m * 1.01, 30, flux_file, False)
    direction_distribution = siren.distributions.FixedDirection(
        siren.math.Vector3D(0, 0, 1.0)
    )
    position_distribution = siren.distributions.CylinderVolumePositionDistribution(
        siren.geometry.Cylinder(3.8, 0, 7.6)
    )
    
    # Set up injection and physical distributions
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
    
    # Initialize and generate events
    controller.SetProcesses(
        primary_type, 
        primary_injection_distributions, 
        primary_physical_distributions
    )
    controller.Initialize()
    
    print(f"Initialization complete for m={m:.2e}, U={U:.2e}")
    print(f"Minimum decay width: {controller.DN_min_decay_width}")
    
    events = controller.GenerateEvents(fill_tables_at_exit=False)
    
    # Save results
    os.makedirs("output_epem_bar", exist_ok=True)
    controller.SaveEvents(
        f"output_epem_bar/{experiment}_Dipole_M{m:.2e}_Umu{U:.2e}_example",
        fill_tables_at_exit=True,
        save_int_probs=True
    )
    
    return f"Completed: m={m:.2e}, U={U:.2e}"

def main():
    """
    Main function to generate HNL events across a grid of mass and coupling parameters.
    Can run either sequentially or in parallel.
    """
    # Generate parameter grid
    n_m, n_U = 30, 20
    m_sample = np.geomspace(2e-2, 4e-1, n_m)
    U_sample = np.geomspace(1e-7, 1e-2, n_U)
    m_sample, U_sample = np.meshgrid(m_sample, U_sample)
    m_sample = m_sample.flatten()
    U_sample = U_sample.flatten()
    
    # Sequential processing
    """
    for m, u in zip(m_sample, U_sample):
        generate_hnl_events(m, u)
    """
    
    # Uncomment for parallel processing
    
    num_cores = mp.cpu_count() - 1  # Leave one core free
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(generate_hnl_events, zip(m_sample, U_sample))
        
    for result in results:
        print(result)
    

if __name__ == "__main__":
    main()