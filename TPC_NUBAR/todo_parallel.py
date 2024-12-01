import numpy as np
import os
import siren
from siren.SIREN_Controller import SIREN_Controller
from multiprocessing import Pool, cpu_count
import traceback
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)

def HNL_siren(params):
    """
    Modified version of HNL_siren that takes a tuple of parameters and includes error handling
    """
    try:
        exp, m4, tr4 = params
        
        # Define a DarkNews model
        model_kwargs = {
            "m4": m4,
            "mu_tr_mu4": tr4,
            "UD4": 0,
            "Umu4": 0,
            "epsilon": 0.0,
            "gD": 0.0,
            "decay_product": "photon",
            "noHC": True,
            "HNLtype": "dirac",
        }

        # Number of events to inject
        events_to_inject = 100000
        
        # Define the controller
        controller = SIREN_Controller(events_to_inject, exp)
        
        # Particle to inject
        primary_type = siren.dataclasses.Particle.ParticleType.NuMu
        xs_path = siren.utilities.get_cross_section_model_path(
            f"DarkNewsTables-v{siren.utilities.darknews_version()}", 
            must_exist=False
        )
        
        # Define DarkNews Model
        table_dir = os.path.join(
            xs_path,
            "Dipole_M%2.2e_mu%2.2e" % (model_kwargs["m4"], model_kwargs["mu_tr_mu4"]),
        )
        controller.InputDarkNewsModel(primary_type, table_dir, **model_kwargs)
        
        # Primary distributions
        primary_injection_distributions = {}
        primary_physical_distributions = {}
        
        # energy distribution
        flux_file = siren.utilities.get_tabulated_flux_file("T2K_OLD","MINUS_numubar")
        edist = siren.distributions.TabulatedFluxDistribution(flux_file, True)
        edist_gen = siren.distributions.TabulatedFluxDistribution(
            model_kwargs["m4"], 20, flux_file, False
        )
        primary_injection_distributions["energy"] = edist_gen
        primary_physical_distributions["energy"] = edist
        
        # direction distribution
        direction_distribution = siren.distributions.FixedDirection(
            siren.math.Vector3D(0, 0, 1.0)
        )
        primary_injection_distributions["direction"] = direction_distribution
        primary_physical_distributions["direction"] = direction_distribution
        
        # position distribution
        decay_range_func = siren.distributions.DecayRangeFunction(
            model_kwargs["m4"], controller.DN_min_decay_width, 3, 284.9
        )
        position_distribution = siren.distributions.RangePositionDistribution(
            3.8, 3.8, decay_range_func, set(controller.GetDetectorModelTargets()[0])
        )
        primary_injection_distributions["position"] = position_distribution
        
        # SetProcesses
        controller.SetProcesses(
            primary_type, primary_injection_distributions, primary_physical_distributions
        )
        controller.Initialize()
        
        def stop(datum, i):
            secondary_type = datum.record.signature.secondary_types[i]
            return secondary_type != siren.dataclasses.Particle.ParticleType.N4
            
        controller.injector.SetStoppingCondition(stop)
        events = controller.GenerateEvents(fill_tables_at_exit=False)
        
        os.makedirs("output_new", exist_ok=True)
        output_file = f"output_new/{exp}_Dipole_M{m4:2.2e}_mu{tr4:2.2e}_example"
        controller.SaveEvents(output_file, fill_tables_at_exit=True)
        
        logging.info(f"Successfully completed simulation for m4={m4}, tr4={tr4}")
        return True, (m4, tr4)
    
    except Exception as e:
        logging.error(f"Error in simulation for m4={m4}, tr4={tr4}: {str(e)}")
        logging.error(traceback.format_exc())
        return False, (m4, tr4)

def main():
    # Create parameter grid
    n_m = 10
    n_mu = 10
    m_sample = np.geomspace(1e-2, 2, n_m)
    mu_sample = np.geomspace(1e-7, 1e-5, n_mu)
    m_sample, mu_sample = np.meshgrid(m_sample, mu_sample)
    m_sample = np.reshape(m_sample, [n_m * n_mu])
    mu_sample = np.reshape(mu_sample, [n_m * n_mu])
    
    # Create parameter tuples for multiprocessing
    params = [('OGTPC1', m, mu) for m, mu in zip(m_sample, mu_sample)]
    
    # Determine number of processes to use (leave one core free for system)
    n_processes = max(1, cpu_count() - 1)
    logging.info(f"Starting parallel processing with {n_processes} processes")
    
    # Start timing
    start_time = datetime.now()
    
    # Create pool and run parallel processes
    with Pool(processes=n_processes) as pool:
        results = pool.map(HNL_siren, params)
    
    # Process results
    successful = [r[1] for r in results if r[0]]
    failed = [r[1] for r in results if not r[0]]
    
    # Log summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logging.info(f"\nSimulation Summary:")
    logging.info(f"Total runtime: {duration}")
    logging.info(f"Total simulations: {len(results)}")
    logging.info(f"Successful: {len(successful)}")
    logging.info(f"Failed: {len(failed)}")
    
    if failed:
        logging.warning("\nFailed simulations:")
        for m4, tr4 in failed:
            logging.warning(f"m4={m4}, tr4={tr4}")

if __name__ == '__main__':
    main()