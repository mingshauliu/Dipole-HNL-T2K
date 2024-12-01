import numpy as np
import os
import awkward as awk
from tqdm import tqdm

POT = 2.6098758621e22
weighted_med = {}

n_m = 10
n_mu = 10
m_sample = np.geomspace(1e-2,2,n_m)
mu_sample = np.geomspace(1e-7,1e-5,n_mu)
m_sample, mu_sample = np.meshgrid(m_sample, mu_sample)
m_sample = np.reshape(m_sample,[n_m*n_mu])
mu_sample = np.reshape(mu_sample,[n_m*n_mu])

def weighted_median(data, weights):
    """Calculate the weighted median of a list of values."""
    data, weights = np.array(data), np.array(weights)
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median

# Wrap the zip object with tqdm for the progress bar
for m4, tr4 in tqdm(zip(m_sample, mu_sample), total=len(m_sample), desc="Processing samples"):
    path = "ND280UPGRD_Dipole_M%2.2e_mu%2.2e_example.parquet" % (m4, tr4)
    if os.path.isfile('./output/' + path):
        data = awk.from_parquet("output/" + path)
        weight = data['event_weight']
        dist = []
        for pts in data['vertex']:
            [a, b] = pts
            dist.append(np.linalg.norm(a - b))
        weighted_med[(m4, tr4)] = weighted_median(dist, weight)

def extract_xyz(data_dict):
    # Extract the keys and values
    keys = np.array(list(data_dict.keys()))
    values = np.array(list(data_dict.values()))
    # Split the keys into x and y
    x = keys[:, 0]
    y = keys[:, 1]
    # The values are already z
    z = values
    return x, y, z

def save_xyz_to_txt_python(x, y, z, filename='xyz_data.txt'):
    with open(filename, 'w') as f:
        f.write('x\ty\tz\n')  # Header
        for xi, yi, zi in zip(x, y, z):
            f.write(f'{xi}\t{yi}\t{zi}\n')
    
    print(f"Data saved to {filename}")

x, y, z = extract_xyz(weighted_med)

save_xyz_to_txt_python(x, y, z, 'my_data.txt')

