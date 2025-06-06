import logging
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from tabulate import tabulate

# Directory to save the plots
PLOT_DIR = "_Plot"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs("_Logs", exist_ok=True)

# Set up logging to file
logging.basicConfig(
    filename="_Logs/correlation_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function to save the plot
def save_plot(fig, filename):
    """
    Saves a figure as a PNG file to the specified directory.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to save.
        filename (str): The name of the file to save the figure as.
    """
    fig_path = os.path.join(PLOT_DIR, f"{filename}.png")
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

# Function for the correlation heatmap 
def correlation_heatmap(data, filename):
    """
    Generates and saves a correlation heatmap from the dataset, and logs the correlation matrix.

    Parameters:
        data (pandas.DataFrame): The dataset containing the features.
        filename (str): The name of the file to save the correlation heatmap as.

    Returns:
        pandas.DataFrame: The correlation matrix.
    """
    logging.info("(----------------CORRELATION HEATMAP------------------)")
    numeric_data = data.select_dtypes(include=['number']).dropna()
    corr_matrix = numeric_data.corr()

    # Save the matrix to a CSV
    corr_csv_path = f"Data_DAG/correlation_matrix.csv"
    corr_matrix.to_csv(corr_csv_path)

    # Write the matrix to the log file as a table
    corr_table = tabulate(corr_matrix, headers='keys', tablefmt='grid', floatfmt=".2f")
    logging.info("Correlation Matrix:\n%s", corr_table)

    # Generate and save the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title(f"Feature Correlation Heatmap for {filename}")
    save_plot(fig, f"{filename}_correlation_heatmap")

    logging.info("---> Heatmap saved to: %s", os.path.join(PLOT_DIR, f"{filename}_correlation_heatmap.png"))
    logging.info("---> Matrix CSV saved to: %s", corr_csv_path)

    return corr_matrix

# Function to create the graph based on the correlation matrix
def plot_correlation_graph(df, threshold, output_path_pickle="_Plot/correlation_graph.pkl", output_path_png="_Plot/correlation_graph.png"):
    """
    Generates a graph based on the correlation matrix, where edges represent correlations
    above a specified threshold. The graph is saved as both a pickle and PNG file.

    Parameters:
        df (pandas.DataFrame): The dataset containing the features.
        threshold (float): The correlation threshold above which an edge is added.
        output_path_pickle (str): Path to save the graph as a pickle file.
        output_path_png (str): Path to save the graph as a PNG image.
    """
    os.makedirs(os.path.dirname(output_path_pickle), exist_ok=True)

    # Calculate the correlation matrix
    corr = df.corr().abs()

    # Create an undirected graph
    G = nx.Graph()
    for col in corr.columns:
        G.add_node(col)

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] >= threshold:
                G.add_edge(corr.columns[i], corr.columns[j], weight=corr.iloc[i, j])

    # Save the graph as a pickle
    with open(output_path_pickle, 'wb') as f:
        pickle.dump(G, f)

    # Draw the graph with simple styling
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=1500)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    ax.set_title(f"Correlation Graph (Threshold ≥ {threshold:.2f})")
    ax.axis('off')
    plt.tight_layout()

    # Save the PNG image
    fig.savefig(output_path_png, bbox_inches='tight')
    plt.close(fig)

    logging.info("---> Graph saved as image to: %s", output_path_png)
    logging.info("---> Graph saved as pickle to: %s", output_path_pickle)


# ---------- MAIN SCRIPT ----------

if __name__ == "__main__":
    dataset_path = "Data_DAG/Nodi_DAG2.csv"

    try:
        logging.info("Loading the dataset: %s", dataset_path)
        df = pd.read_csv(dataset_path)
        logging.info("Dataset loaded successfully. Rows: %d, Columns: %d", df.shape[0], df.shape[1])

        output_filename = "correlation_output"
        correlation_matrix = correlation_heatmap(df, output_filename)

        # Create the correlation graph
        plot_correlation_graph(df, threshold=0.24)

    except Exception as e:
        logging.error("Error during the process: %s", str(e))

