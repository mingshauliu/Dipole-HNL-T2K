import awkward as awk
import numpy as np
import os
import siren
import multiprocessing
from siren.SIREN_Controller import SIREN_Controller

def HNL_siren(m4):
    
    U = 0.0004832930238571752
    
    if os.path.exists(f'./T2K/OGTPC3_Dipole_M{m4:2.2e}_Umu{U:2.2e}_example.parquet'):
        return
    

    # Define a DarkNews model
    model_kwargs = {
        "m4": m4,  # 0.140,
        "mu_tr_mu4": 0,  # 1e-6, # GeV^-1
        "UD4": 0,
        "Umu4": U,
        "epsilon": 0.0,
        "gD": 0.0,
        "decay_product": "e+e-",
        "noHC": True,
        "HNLtype": "dirac",
    }

    # Number of events to inject
    events_to_inject = 100000

    # Expeirment to run
    experiment = "OGTPC3"

    # Define the controller
    controller = SIREN_Controller(events_to_inject, experiment)

    # Particle to inject
    primary_type = siren.dataclasses.Particle.ParticleType.N4

    xs_path = siren.utilities.get_cross_section_model_path(f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False)
    # Define DarkNews Model
    table_dir = os.path.join(
        xs_path,
        "Dipole_M%2.2e_Umu%2.2e" % (model_kwargs["m4"], model_kwargs["Umu4"]),
    )
    controller.InputDarkNewsModel(primary_type, table_dir, upscattering=False, **model_kwargs)

    # Primary distributions
    primary_injection_distributions = {}
    primary_physical_distributions = {}

    # energy distribution
    flux_file = siren.utilities.get_tabulated_flux_file("N_Inject", "numu_PLUS")
    edist = siren.distributions.TabulatedFluxDistribution(flux_file, True)
    edist_gen = siren.distributions.TabulatedFluxDistribution(
        model_kwargs["m4"], 20, flux_file, False
    )
    primary_injection_distributions["energy"] = edist_gen
    primary_physical_distributions["energy"] = edist

    # direction distribution
    direction_distribution = siren.distributions.FixedDirection(siren.math.Vector3D(0, 0, 1.0))
    primary_injection_distributions["direction"] = direction_distribution
    primary_physical_distributions["direction"] = direction_distribution

    # position distribution
    decay_range_func = siren.distributions.DecayRangeFunction(
        model_kwargs["m4"], controller.DN_min_decay_width, 3, 284.9
    )
    
    # position_distribution = siren.distributions.RangePositionDistribution(5.0, 9.0, decay_range_func, set(controller.GetDetectorModelTargets()[0]) )
    
    position_distribution = siren.distributions.CylinderVolumePositionDistribution(siren.geometry.Cylinder(3.8,0,7.6))

    
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

    os.makedirs("T2K", exist_ok=True)

    controller.SaveEvents(
        "T2K/OGTPC3_Dipole_M%2.2e_Umu%2.2e_example"
        % (model_kwargs["m4"], model_kwargs["Umu4"]),
        fill_tables_at_exit=True
    )


def parallel_process(data1, data2):
    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count()
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=int(num_cores/2)) as pool:
        # Use starmap to pass multiple arguments to the worker function
        results = pool.starmap(HNL_siren, zip(data1, data2))
        
    return results

def get_new_sample():

    m_sample_new = []
    mu_sample_new = []

    with open('./new_points.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            i, ii = line.strip().split(', ')
            m_sample_new += [float(i)]
            mu_sample_new += [float(ii)]
            
    return m_sample_new, mu_sample_new

if __name__ == '__main__':
    
    m_sample = np.geomspace(1e-2,2,10)

    HNL_siren(m_sample[0])
