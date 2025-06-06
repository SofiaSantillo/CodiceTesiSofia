import pandas as pd
import numpy as np
import scipy.stats as stats
import pickle
import logging

# Logging Configuration
logging.basicConfig(filename="_Logs/bayesian_regression.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

#  Function to remove NaN from data 
def clean_data(df):
    return df.dropna()


df = pd.read_csv("Data_DAG/Nodi_DAG2.csv") 
logging.info(f"Dataset loaded with columns: {df.columns.tolist()}")

# Load Prior Distributions
try:
    with open("_Plot/fitted_distributions.pkl", "rb") as f:
        prior_dict = pickle.load(f)
    logging.info("Prior distributions loaded successfully.")
except FileNotFoundError:
    logging.error("The prior distributions file was not found.")
    raise

# Function to create prior from name + parameters 
def create_prior(name, dist_name, params):
    if dist_name == "uniform":
        lower, upper = params
        return stats.uniform(lower, upper - lower)
    elif dist_name == "expon":
        loc, scale = params
        return stats.expon(loc, scale)
    elif dist_name == "lognorm":
        s, loc, scale = params
        return stats.lognorm(s, loc, scale)
    else:
        raise ValueError(f"Distribution {dist_name} not supported.")

# Function to calculate joint probability P(X, Y)
def joint_probability(x, y, prior_dict):
    if x not in prior_dict or y not in prior_dict:
        logging.error(f"Variable {x} or {y} not found in prior distributions.")
        return None
    
    # Extract distribution and parameters for variable x
    dist_x_name = prior_dict[x]["best_distribution"]
    params_x = prior_dict[x]["parameters"]
    
    # Extract distribution and parameters for variable y
    dist_y_name = prior_dict[y]["best_distribution"]
    params_y = prior_dict[y]["parameters"]
    
    # Create the distribution object based on the name
    try:
        dist_x = create_prior(x, dist_x_name, params_x)
        dist_y = create_prior(y, dist_y_name, params_y)
    except ValueError as e:
        logging.error(f"Error in creating the distribution: {e}")
        return None

    # Calculate joint probability for a specific value
    p_x = dist_x.pdf(params_x[0])  # Probability of x (for the first parameter)
    p_y = dist_y.pdf(params_y[0])  # Probability of y (for the first parameter)
    return p_x * p_y

# Function to calculate conditional probability 
def conditional_probability(x, y, prior_dict):
    joint_prob = joint_probability(x, y, prior_dict)
    if joint_prob is None:
        return None
    
    # Calculate P(Y) as the sum of PDF of Y over all its values
    dist_y_name = prior_dict[y]["best_distribution"]
    params_y = prior_dict[y]["parameters"]
    dist_y = create_prior(y, dist_y_name, params_y)
    
    # P(Y) = Sum of PDF of Y over all possible values of y (numerical approximation)
    p_y = np.sum(dist_y.pdf(np.linspace(min(params_y), max(params_y), 100)))  
    
    if p_y == 0:
        logging.error(f"Probability P({y}) is zero, cannot calculate P({x} | {y})")
        return None
    
    return joint_prob / p_y

# === Main Execution ===
def main():
    logging.info(f"\n\n\nSTART\n\n")
    # Perform the calculation of joint and conditional probability for a pair of variables
    for xi in df.columns:
        for xj in df.columns:
            if xi == xj:
                continue
            logging.info(f"\nCalculating joint and conditional probability for {xi} and {xj}")
            # Joint probability
            p_joint = joint_probability(xi, xj, prior_dict)
            if p_joint is not None:
                logging.info(f"Joint probability P({xi}, {xj}): {p_joint}")
            else:
                logging.error(f"Error in calculating joint probability P({xi}, {xj})")

            # Conditional probability
            p_conditional = conditional_probability(xi, xj, prior_dict)
            if p_conditional is not None:
                logging.info(f"Conditional probability P({xi} | {xj}): {p_conditional}")
            else:
                logging.error(f"Error in calculating conditional probability P({xi} | {xj})")

if __name__ == '__main__':
    main()
