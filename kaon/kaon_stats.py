import awkward as awk
import numpy as np
import os

pot = POT['ND280_nu_2019']
enhance = 'nu'

n_m, n_U = 30, 20
m_sample = np.geomspace(2e-2, 4e-1, n_m)
U_sample = np.geomspace(1e-7, 1e-2, n_U)
m_sample, U_sample = np.meshgrid(m_sample, U_sample)
m_sample = m_sample.flatten()
U_sample = U_sample.flatten()

def rho(u, m):
    a = (0.51e-3 / 0.494)**2 # Electron mass and Kaon mass ratio in GeV
    b = (m / 0.494)**2 # HNL mass and Kaon mass ratio in GeV
    sqkallen = np.sqrt(1 + a**2 + b**2 -2*a - 2*b - 2*a*b) # Kallen function λ(1,a,b)
    return u**4 * ( a + b - ( a - b )**2 ) * sqkallen

with open('./counts.txt', 'w') as f, with open('./OGTPC1.txt', 'w') as w:
    f.write(f'mass U counts')
    for m, u in zip(m_sample, U_sample):
        path = "OGTPC1_Dipole_M%2.2e_Umu%2.2e_example.parquet" % (m, u)
        file_path = f'./output_{enhance}/{path}'
        if os.path.isfile(file_path):
            try:
                data = awk.from_parquet(file_path)
                weight = data['event_weight']
                mask = np.logical_not(np.isnan(weight))
                scaled = data['event_weight'] * rho(u, m)
                f.write(f'{m} {u} {np.sum(scaled[mask])}')
            except Exception as e:
                # If there's any error reading the file, write the parameters to the file
                print(f"Error reading file {file_path}: {str(e)}")
                w.write(f"{m} {u}\n")
        else:
            w.write(f"{m} {u}\n")

