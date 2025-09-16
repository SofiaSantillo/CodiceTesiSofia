import logging


def setup_logging(root, path):
    """Set up logging configuration."""
    logging.basicConfig(
        filename= root + "/" + path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


logs_path = "_Logs"
data_path = "_Data"
files_path = "_File"
plots_path = "Plot"
models_path = "_Model"