__author__ = "Martin Widmer"
__email__ = "martin.widmer2@unibe.ch"

"""
Helper functions for monte carlo simulation
"""

# install cuda on computer and try to run pip install cupy-cuda12x in terminal
try:
    import cupy as np
    #print("CuPy available, calculating on gpu ...")
    _USE_GPU = True
except ModuleNotFoundError:
    import numpy as np
    #print("CuPy not available, falling back to cpu ...")
    _USE_GPU = False

def get_sim_endpoints(num_endpoints, data_cov, aimpoint_x):
    """
    Returns num_endpoints simulated endpoints distributed according to data_cov aimed at aimpoint_x. 
    It is not possible to set a y- aimpoint.
    """

    random = np.random.randn(num_endpoints, 2)
    # Calculate mean and covariance of random numbers
    random_mean = random.mean(axis=0)
    random_cov_matrix = np.cov(random, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    random_eigenvalues, random_eigenvectors = np.linalg.eigh(random_cov_matrix)
    data_eigenvalues, data_eigenvectors = np.linalg.eigh(data_cov)

    # Whitening transform (zero means correction) (@ is matrix multiplication!)
    #white_noise = (random_eigenvectors @ np.linalg.inv(np.sqrt(np.diag(random_eigenvalues))).T @ random.T - random_mean[:, np.newaxis])

    whitened_random = random_eigenvectors @ np.diag(1.0 / np.sqrt(random_eigenvalues)) @ random_eigenvectors.T @ (random - random_mean).T
    
    # Inverse whitening to center the distribution to means
    #sim_endpoints = (np.linalg.inv(data_eigenvectors @ np.linalg.inv(np.sqrt(np.diag(data_eigenvalues))).T) @ white_noise + np.array([aimpoint_x, 0])[:, np.newaxis]).T
    sim_endpoints = (data_eigenvectors @ np.diag(np.sqrt(data_eigenvalues)) @ data_eigenvectors.T @ whitened_random + np.array([aimpoint_x, 0])[:, np.newaxis]).T

    return sim_endpoints



def monte_carlo(data_cov,
                num_endpoints=100000,
                radius=30,
                gain_penalty=-500,
                gain_target=100,
                penalty_circle_location_x=-60,
                target_circle_location_x=0):
    """
    Performs a Monte-Carlo-Simulation by simulating endpoints for the provided aim-point covariance

    Returns proportion_red, proportion_green, expected_gain, optimal_aimpoint_x 
    """

    width = 60
    proportion_red = np.empty(width)
    proportion_green = np.empty(width)
    expected_gain = np.empty(width)
    for aimpoint_x in range(0,width):
        sim_endpoints = get_sim_endpoints(num_endpoints, data_cov, aimpoint_x)

        radius_squared = radius**2
        # Amount of endpoints in penalty circle
        # Using pythagoras and fact that sqrt(x^2) < r  <=>  x^2 < r^2 for all positive r
        prop_red = np.sum((sim_endpoints[:, 0] - penalty_circle_location_x)**2 +
                        (sim_endpoints[:, 1] - 0)**2 < radius_squared)

        # Amount of endpoints in target circle
        prop_green = np.sum((sim_endpoints[:, 0] - target_circle_location_x)**2 +
                        (sim_endpoints[:, 1] - 0)**2 < radius_squared)

        # Calculate proportions of endpoints within each target
        proportion_red[aimpoint_x] = prop_red / num_endpoints
        proportion_green[aimpoint_x] = prop_green / num_endpoints

    expected_gain = proportion_red * gain_penalty + proportion_green * gain_target
    optimal_aimpoint_x = np.argmax(expected_gain)

    return proportion_red, proportion_green, expected_gain, optimal_aimpoint_x
