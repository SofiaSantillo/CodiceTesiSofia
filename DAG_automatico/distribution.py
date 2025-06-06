import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
from scipy.stats import kstest

# Function to fit distributions
def fit_distribution(data):
    """
    Fit multiple distributions to the data and return the best distribution and its parameters.
    Args:
        data (array-like): The data to fit.
        distributions (list, optional): List of distribution functions to fit. Defaults to None.
    Returns:
        tuple: Best distribution name and its parameters.
    """
    distributions = [stats.norm, stats.expon, stats.lognorm, stats.gamma, stats.beta, stats.uniform]
    best_distribution = None
    best_params = None
    best_sse = float('inf')
    best_dist_name = ""

    # Loop through distributions and compute the fitting
    for dist in distributions:
        try:
            # Fit the distribution
            params = dist.fit(data)
            
            # Calculate the log-likelihood value for the fitting
            sse = np.sum((data - dist.pdf(data, *params)) ** 2)
            
            # If the fit is better, save the distribution and parameters
            if sse < best_sse:
                best_sse = sse
                best_distribution = dist
                best_params = params
                best_dist_name = dist.name
        except Exception as e:
            continue
    
    return best_dist_name, best_params

# Function to calculate the fitting for all variables in the dataframe
def fit_distributions_for_dataframe(df):
    """
    Fit distributions to all numeric columns in the dataframe.
    Args:
        df (DataFrame): The dataframe to process.
        distributions (list, optional): List of distribution functions to fit. Defaults to None.
    Returns:
        dict: Results with best distribution and parameters for each numeric column.
    """
    results = {}
    
    for column in df.select_dtypes(include=[np.number]).columns:
        data = df[column].dropna()
        best_dist_name, best_params = fit_distribution(data)
        results[column] = {
            'best_distribution': best_dist_name,
            'parameters': best_params
        }
        
    return results

# Function to plot the distribution graph
def plot_distribution(data, dist_name, params, column_name):
    """
    Plot the distribution of the data along with the best fit distribution.
    Args:
        data (array-like): The data to plot.
        dist_name (str): The name of the distribution to plot.
        params (tuple): Parameters for the distribution.
        column_name (str): The column name for labeling.
        output_dir (str, optional): Directory to save the plot. Defaults to "_Plot".
    """
    # Create the x array for the distribution
    x = np.linspace(min(data), max(data), 100)
    dist = getattr(stats, dist_name)
    pdf_fitted = dist.pdf(x, *params)

    # Create the figure with aesthetic improvements
    plt.figure(figsize=(8, 6))
    
    # Plot the histogram with opacity for observed data
    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', label='Observed data')
    
    # Plot the fitted distribution curve
    plt.plot(x, pdf_fitted, 'r-', label=f'{dist_name} fit', linewidth=2)
    
    # Add the "area under the curve" (fill under the line)
    plt.fill_between(x, pdf_fitted, alpha=0.2, color='red')
    
    # Customize the plot
    plt.title(f"Distribution for {column_name}", fontsize=14, fontweight='bold')
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot in PNG format
    plt.savefig(f"_Plot/{column_name}_distribution.png")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv('Data_DAG/Nodi_DAG2.csv')  # Replace with your dataset
    
    # Calculate the best distribution for each continuous variable
    fitted_distributions = fit_distributions_for_dataframe(df)
    
    # Save the results to a .pkl file
    with open('_Plot/fitted_distributions.pkl', 'wb') as f:
        pickle.dump(fitted_distributions, f)
    
    # Save the results to a .log file
    with open('_Logs/fitted_distributions.log', 'w') as f:
        for column, result in fitted_distributions.items():
            f.write(f"Column: {column}\n")
            f.write(f"Best Distribution: {result['best_distribution']}\n")
            f.write(f"Parameters: {result['parameters']}\n")
            f.write('-' * 50 + '\n')
    
    # Plot and save the graphs for each variable
    for column, result in fitted_distributions.items():
        dist_name = result['best_distribution']
        params = result['parameters']
        plot_distribution(df[column].dropna(), dist_name, params, column)

