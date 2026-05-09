

# --------------- #
# !-- Imports --! #
# --------------- #

from oakley import *
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import os
import requests
from astropy.io import fits


folder_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
os.makedirs(folder_path, exist_ok=True)



class MirixGrid:
    
    _default_folder_path = folder_path
    _data_url = "https://api.github.com/repos/ProfesseurShadoko/mirix_grids/contents/data"
    
    
    # ---------------------- #
    # !-- Initialization --! #
    # ---------------------- #
    
    def __init__(self, filepath:str):
        """
        Initializes a MirixGrid object by loading the grid from a FITS file. If the file doesn't exist
        in the working directory, the function will look for it in the default folder.
        
        Files shoul be downlaoded from Github to the default folder using :meth:`MirixGrid.download()`.
        
        Parameters
        ----------
        filepath : str
            The path to the FITS file containing the grid. If the file doesn't exist in the working directory,
            the function will look for it in the default folder.
        """
        
        # 1. Load the file
        if os.path.exists(filepath):
            self.filepath = filepath
        else:
            filepath = os.path.join(MirixGrid._default_folder_path, filepath)
            if os.path.exists(filepath):
                self.filepath = filepath
            else:
                raise FileNotFoundError(f"File not found: {filepath}. Please make sure the file exists in the working directory or in the default folder, or download it from Github using MirixGrid.download().")
        self.hdul = fits.open(self.filepath)
    
        # 2. Extract information
        with Task("Loading grid"):
            self.components = self.hdul["COMPONENTS"].data
            self.coefficients = self.hdul["COEFFICIENTS"].data
            self.singular_values = self.hdul["SINGULAR_VALUES"].data
            self.xgrid = self.hdul["XGRID"].data
            self.ygrid = self.hdul["YGRID"].data
        
        #
        
    
    
    
    
    
    # ------------------- #
    # !-- Data loader --! #
    # ------------------- #
    
    def download(reset:bool = False):
        """
        Downloads all available grids from Github, and saves them in the default folder.
        """
        if reset:
            with Task("Deleting existing grids..."):
                for f in os.listdir(MirixGrid._default_folder_path):
                    if f.endswith(".fits") and "pca_k-" in f:
                        os.remove(os.path.join(MirixGrid._default_folder_path, f))
        with Task("Downloading available grids from Github..."):
            filepaths = MirixGrid._list_available_grids("github")
            for filename in ProgressBar(filepaths):
                if not os.path.exists(os.path.join(MirixGrid._default_folder_path, filename)):
                    with Task(f"Downloading {filename}..."):
                        response = requests.get(f"{MirixGrid._data_url}/{filename}")
                        response.raise_for_status()
                        file_content = response.json()
                        download_url = file_content["download_url"]
                        file_response = requests.get(download_url)
                        file_response.raise_for_status()
                        with open(os.path.join(MirixGrid._default_folder_path, filename), "wb") as f:
                            f.write(file_response.content)
                else:
                    Message(f"File already exists: {filename}.", "#")
    
    @staticmethod
    def extract_metadata_from_filename(filename:str) -> dict:
        """
        Extracts the metadata from the filename of a grid. The routine is the following:
        - split on "_", each part being used to extract a key-value pair
        - for each part, split on "-", the first part being the key, and the second part (if it exists) being the value, otherwise None.
        
        This function is intended to be used to choose which grid to use.
        
        Parameters
        ----------
        filename : str
            The filename of the grid, from which the metadata will be extracted. Can be a full 
            path as well, in which case only the basename will be used to extract the metadata.
            
        Returns
        -------
        dict
            A dictionary containing the metadata extracted from the filename, with keys being the part before the "-"
            and values being the part after the "-", or None if there is no "-".
        """
        filename = os.path.basename(filename)
        metadata = {}
        parts = filename.split("_")
        for part in parts:
            if "-" in part:
                key, value = part.split("-", 1)
                metadata[key] = value
            else:
                metadata[part] = None
        return metadata
    
    @staticmethod
    def _list_available_grids(folder_path:str = None) -> list:
        """
        Lists all available grids in the default folder, or in a provided folder.
        If "github" is provided as `folder_path`, the function will list all available grids on Github.
        
        Parameters
        ----------
        folder_path : str, optional
            The path to the folder containing the grids. If None, the default folder will be used.
            If "github" is provided, the function will list all available grids on Github.
        
        Returns
        -------
        list
            A list of filenames of the available grids in the specified folder or on Github.
        """
        if folder_path is None:
            folder_path = MirixGrid._default_folder_path
            
        if folder_path.lower() == "github":
            with Task("Looking for available grids on Github..."):
                response = requests.get(MirixGrid._data_url)
                response.raise_for_status()
                
                items = response.json()
                fits_files = [
                    f["name"] for f in items
                    if f["name"].endswith(".fits")
                    and "pca_k-" in f["name"]
                ]
                
                Message("Available grids:").list(fits_files)
                return fits_files
        else:
            with Task(f"Looking for available grids in folder {os.path.basename(folder_path)}..."):
                if not os.path.exists(folder_path):
                    Message(f"No folder found at {folder_path}.", "!")
                    return []
                
                fits_files = [
                    f for f in os.listdir(folder_path)
                    if f.endswith(".fits")
                    and "pca_k-" in f
                ]
                
                Message("Available grids:").list(fits_files)
                return fits_files
    
    
 

if __name__ == "__main__":
    
    # MirixGrid.download(reset=False)
    
    mxgrid = MirixGrid("grid_filter-F1140C_nlambda-1_oversample-4_date-2023-09-04_psfshape-183_gridshape-110_psfoversample-2_pca_k-200.fits")
    
    