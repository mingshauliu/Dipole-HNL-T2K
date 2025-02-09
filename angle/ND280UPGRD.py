import numpy as np
import os
import siren
import multiprocessing
from siren.SIREN_Controller import SIREN_Controller

def HNL_siren(m4, tr4):

    output_destination='./output_2'
    output_file = f"{output_destination}/ND280UPGRD_Dipole_M{m4:2.2e}_mu{tr4:2.2e}_example"
    output_file1 = output_file+".parquet"
    output_file2 = output_file+".parquet.parquet"
    
    if os.path.isfile(output_file1) or os.path.isfile(output_file2):
        print(f'{output_file1} has already been done.')
        return 
    
    try:
        # Define a DarkNews model
        model_kwargs = {
            "m4": m4,  # 0.140,
            "mu_tr_mu4": tr4,  # 1e-6, # GeV^-1
            "UD4": 0,
            "Umu4": 0,
            "epsilon": 0.0,
            "gD": 0.0,
            "decay_product": "e+e-",
            "noHC": True,
            "HNLtype": "dirac",
        }

        # Number of events to inject
        events_to_inject = 100000

        # Expeirment to run
        experiment = "ND280UPGRD"

        # Define the controller
        controller = SIREN_Controller(events_to_inject, experiment)

        # Particle to inject
        primary_type = siren.dataclasses.Particle.ParticleType.NuMu

        xs_path = siren.utilities.get_cross_section_model_path(f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False)
        # Define DarkNews Model
        table_dir = os.path.join(
            xs_path,
            "Dipole_M%2.2e_mu%2.2e" % (model_kwargs["m4"], model_kwargs["mu_tr_mu4"]),
        )
        controller.InputDarkNewsModel(primary_type, table_dir, use_pickles=False, **model_kwargs)

        # Primary distributions
        primary_injection_distributions = {}
        primary_physical_distributions = {}

        # energy distribution
        flux_file = siren.utilities.get_tabulated_flux_file("T2K_NEAR","PLUS_numu")
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
        position_distribution = siren.distributions.RangePositionDistribution(
            2.4, 5.3, decay_range_func, set(controller.GetDetectorModelTargets()[0])
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
        output_destination='./output_2'
        os.makedirs(output_destination, exist_ok=True)

        controller.SaveEvents(
            output_destination+"/ND280UPGRD_Dipole_M%2.2e_mu%2.2e_example"
            % (model_kwargs["m4"], model_kwargs["mu_tr_mu4"]),
            fill_tables_at_exit=True
        )
    except Exception as e:
        print(e)
        return
        
    
def save_results_to_file(results, filename):
    with open(filename, 'w') as file:
        for result in results:
            file.write(f"{result}\n")

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
    
    n_m = 10
    n_mu = 10
    m_sample = np.geomspace(1e-2,2,n_m)
    mu_sample = np.geomspace(1e-7,1e-5,n_mu)
    m_sample, mu_sample = np.meshgrid(m_sample, mu_sample)
    m_sample = np.reshape(m_sample,[n_m*n_mu])
    mu_sample = np.reshape(mu_sample,[n_m*n_mu])

    m_sample = m_sample[0:10]
    mu_sample = mu_sample[0:10]

    results = parallel_process(m_sample, mu_sample)
    
    # Obtain new samples 
    
    # m_sample, mu_sample = get_new_sample()

    # with open('recording.txt','a') as file:

    #     for i, ii in zip(m_sample,mu_sample):
    #         # Call your function
    #         result = HNL_siren(i,ii)
    #         file.write(', '.join(map(str, result)))


    # # Save results to a file
    # output_filename = "./ND280UPGRD.txt"
    # save_results_to_file(results, output_filename)

    # print(f"Results have been saved to {output_filename}")
    
    

# with open('recording.txt', 'a') as file:



          
#         # if not isinstance(result, np.ndarray):
#         #     result = np.array(result)
        
#         # # Convert the array to a string with scientific notation and 16 decimal places
#         # array_string = ', '.join([f'{x:.16e}' for x in result])
        
#         # Write to file without iteration number
#         file.write(f"{result}\n")
