from pathlib import Path

# in notebooks or scripts:
# notebook.ipynb
# from projectname.config import data_path

data_dir = Path('/data')
data_raw = data_dir / 'raw'
data_processed = data_dir / 'processed'
