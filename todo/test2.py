import numpy as np
import os
import siren
from siren.SIREN_Controller import SIREN_Controller
import multiprocessing as mp
from functools import partial

def generate_hnl_events(m, U, events_to_inject=1000, experiment="OGTPC1"):
    
    path="OGTPC1_Dipole_M%2.2e_Umu%2.2e_example.parquet"%(m,U)
    
    if os.path.isfile(f"test/{path}"): 
        return f"File {path} already exists"
    
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
    xs_path = siren.utilities.get_cross_section_model_path(f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False)
    
    table_dir = os.path.join(xs_path, f"Dipole_M{m:.2e}_Umu{U:.2e}")
    controller.InputDarkNewsModel(primary_type, table_dir, **model_kwargs)

    flux_file = siren.utilities.get_tabulated_flux_file("N_Inject", "numu_PLUS")
    mdist = siren.distributions.PrimaryMass(m)
    edist = siren.distributions.TabulatedFluxDistribution(flux_file, True)
    edist_gen = siren.distributions.TabulatedFluxDistribution(m * 1.01, 30, flux_file, False)

    direction_distribution = siren.distributions.FixedDirection(siren.math.Vector3D(0, 0, 1.0))
    
    # decay_range_func = siren.distributions.DecayRangeFunction(m, controller.DN_min_decay_width, 3, 284.9)
    # position_distribution = siren.distributions.RangePositionDistribution(3.8, 3.8, decay_range_func, set(controller.GetDetectorModelTargets()[0]))

    position_distribution = siren.distributions.CylinderVolumePositionDistribution(siren.geometry.Cylinder(3.8,0,7.6))

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

    controller.SetProcesses(primary_type, primary_injection_distributions, primary_physical_distributions)
    controller.Initialize()

    print(f"Initialization complete for m={m:.2e}, U={U:.2e}")
    print(f"Minimum decay width: {controller.DN_min_decay_width}")

    events = controller.GenerateEvents(fill_tables_at_exit=False)
    
    os.makedirs("test", exist_ok=True)
    controller.SaveEvents(f"test/{experiment}_Dipole_M{m:.2e}_Umu{U:.2e}_example", fill_tables_at_exit=True, save_int_probs=True)

    return f"Completed: m={m:.2e}, U={U:.2e}"

def main():
    n_m, n_U = 30, 20
    m_sample = np.geomspace(2e-2, 4e-1, n_m)
    U_sample = np.geomspace(1e-7, 1e-2, n_U)
    m_sample, U_sample = np.meshgrid(m_sample, U_sample)
    m_sample = m_sample.flatten()
    U_sample = U_sample.flatten()
    # # Determine the number of CPU cores to use
    # num_cores = mp.cpu_count() - 1  # Leave one core free
    # print(f"Using {num_cores} CPU cores for parallel processing")

    # # Create a pool of worker processes
    # with mp.Pool(num_cores) as pool:
    #     # Use starmap to pass multiple arguments to the function
    #     results = pool.starmap(generate_hnl_events, zip(m_sample, U_sample))
    
    # for result in results:
    #     print(result)
    
    
    generate_hnl_events(m_sample[0],U_sample[0])

if __name__ == "__main__":
    main() 