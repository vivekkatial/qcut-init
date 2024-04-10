import networkx as nx
import numpy as np

def calculate_optimal_angles_for_triangle_free_graph(G):
    """
    Calculate the optimal angles γ*, β* for an arbitrary triangle-free graph within the range [0, π/2],
    with β* potentially fixed at π/8 depending on the regularity of the graph.

    Parameters:
    G : networkx.Graph
        The graph for which to calculate the angles.

    Returns:
    optimal_angles : list of tuples
        A list containing tuples of optimal pairs of angles for the triangle-free graph.
    """
    
    # Check if the graph is triangle-free
    if sum(nx.triangles(G).values()) != 0:
        return None

    # Get the maximum and minimum degree of the graph
    degrees = [degree for node, degree in G.degree()]
    D_max = max(degrees)
    D_min = min(degrees)
    
    optimal_angles = []

    # Case for D-regular graph
    if len(set(degrees)) == 1:  # Check if the graph is regular
        D = D_max  # Since the graph is regular, all degrees are the same
        gamma_star = np.arctan(1 / np.sqrt(D - 1))
        beta_star = -np.pi / 8
    
    # Case for arbitrary triangle-free graph
    else:
        # Lower bound for γ* based on D_max
        if D_max == 1:
            gamma_star_lb = 0
        else:
            gamma_star_lb = np.arctan(1 / np.sqrt(D_max - 1))

        # Upper bound for γ* based on D_min
        if D_min == 1:
            gamma_star_ub = np.pi / 2
        else:
            gamma_star_ub = np.arctan(1 / np.sqrt(D_min - 1))
            
        
        # Sample gamma_star from the range
        gamma_star = np.random.uniform(gamma_star_lb, gamma_star_ub)
        beta_star = -np.pi / 8
        
    return beta_star, gamma_star


def find_min_gamma_beta(data):
    """
    Finds the gamma and beta values that minimize the objective values (obj_vals)
    in the provided data dictionary.
    
    Parameters:
    - data: A dictionary containing 'gamma', 'beta', and 'obj_vals' keys.
    
    Returns:
    A tuple containing the minimum value and the corresponding gamma and beta values.
    """
    # Convert lists to numpy arrays for efficient computation
    gamma = np.array(data['gamma'])
    beta = np.array(data['beta'])
    obj_vals = np.array(data['obj_vals'])

    # Find the index of the minimum value in obj_vals
    min_val_index = np.argmin(obj_vals)
    # Convert the flat index back to a tuple of (i, j)
    min_beta_index, min_gamma_index  = np.unravel_index(min_val_index, obj_vals.shape)
    
    # Retrieve the minimum value, gamma, and beta
    min_val = obj_vals[min_gamma_index, min_beta_index]
    min_gamma = gamma[min_gamma_index]
    min_beta = beta[min_beta_index]

    print(
        f"Minimum value: {min_val}\n"
        f"Gamma: {min_gamma}\n"
        f"Beta: {min_beta}"
    )

    
    
    return min_beta, min_gamma