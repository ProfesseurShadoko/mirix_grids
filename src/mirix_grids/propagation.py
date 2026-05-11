


# todo:
# make every function take no positional arguments
# only kwargs
# i,j means indeces
# x,y means separation in oversampled pixels, with (0,0) the center of the grid
# and in arcsec is x_arcsec, y_arcsec, with (0,0) the center of the grid as well

# todo: for torch precompute the components in fourier space
# reimplement convolution myself
# precompute the correct padding (see conversation with chatgpt)
# check about kernel sizes and all, make sure how fftconvolve will work with that
# sum the k components in fourier space, as the fourier transform is linear!


# --------------- #
# !-- Imports --! #
# --------------- #

from oakley import *
import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len
import matplotlib.pyplot as plt
import os
import requests
from astropy.io import fits
from xrdi_toolkit import *


folder_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
os.makedirs(folder_path, exist_ok=True)



class MirixGrid:
    
    _default_folder_path = folder_path
    _data_url = "https://api.github.com/repos/ProfesseurShadoko/mirix_grids/contents/data"
    
    
    # ---------------------- #
    # !-- Initialization --! #
    # ---------------------- #
    
    def __init__(
        self,
        filepath:str,
        pca_k:int = None,
        # device: str | torch.device | None = None,
        # dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes a MirixGrid object by loading the grid from a FITS file. If the file doesn't exist
        in the working directory, the function will look for it in the default folder.
        
        Files should be downlaoded from Github to the default folder using :meth:`download()`.
        
        Parameters
        ----------
        filepath : str
            The path to the FITS file containing the grid. If the file doesn't exist in the working directory,
            the function will look for it in the default folder.
            
        Notes
        -----
        Both device and dtype parameters need to match the ones used for the input image of the :meth:`forward()` method.
        The use of `torch` instead of `numpy` or `scipy` is motivated by the fact that only `torch` can do
        a convolution (or correlation) on a batch of images / components at once. Others would necessitate
        a python loop, which would be slower. Also, everything here keeps the differentiability of a disk model
        through the porpagation, which could be useful.
        """
        self.pca_k = pca_k
        # self.psf_size = psf_size
        
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
            self.components:np.ndarray = self.hdul["COMPONENTS"].data
            self.coefficients:np.ndarray = self.hdul["COEFFICIENTS"].data
            self.singular_values:np.ndarray = self.hdul["SINGULAR_VALUES"].data
            self.xgrid:np.ndarray = self.hdul["X_GRID"].data
            self.ygrid:np.ndarray = self.hdul["Y_GRID"].data
        
        # 3. Extract metadata from hdul
        header = self.hdul[0].header
        self.metadata = {
            "filter": header["FILTER"],
            "nlambda": header["NLAMBDA"],
            "oversample": header["OVERSAMP"],
            "date": header["DATE"],
            "psf_shape": header["PSFFOV"],
            "gridsize": header["GRIDSIZE"],
            "psf_oversample": header["PSFOVERS"],
            "pca_k": header["PCA_K"],
        }
        if self.pca_k is None:
            self.pca_k = self.metadata["pca_k"]
        #if self.grid_size is None:
        #    self.grid_size = self.metadata["gridsize"]
        #if self.psf_size is None:
        #    self.psf_size = self.metadata["psf_shape"]
            
        assert self.pca_k <= self.metadata["pca_k"], f"pca_k should be less than or equal to the number of PCA components in the grid ({self.metadata['pca_k']})."
        #assert self.grid_size <= self.metadata["gridsize"], f"grid_size should be less than or equal to the grid size in the grid ({self.metadata['gridsize']})."
        #assert self.grid_size % 2 == 0, f"grid_size should be even, but got {self.grid_size}."
        #assert self.psf_size <= self.metadata["psf_shape"], f"psf_size should be less than or equal to the PSF shape in the grid ({self.metadata['psf_shape']})."
        #assert self.psf_size % 2 == 1, f"psf_size should be odd, but got {self.psf_size}."
        
        Message("Grid metadata:").list(self.metadata)
        Message("Data shapes:").list({
            "components": self.components.shape,
            "coefficients": self.coefficients.shape,
            "singular_values": self.singular_values.shape,
            "xgrid": self.xgrid.shape,
            "ygrid": self.ygrid.shape,
        })
        
        # 4. Reshape coefficients and grids
        self.coefficients = self.coefficients.reshape((self.metadata["gridsize"], self.metadata["gridsize"], self.metadata["pca_k"]))
        self.xgrid = self.xgrid.reshape((self.metadata["gridsize"], self.metadata["gridsize"]))
        self.ygrid = self.ygrid.reshape((self.metadata["gridsize"], self.metadata["gridsize"]))
        
        # crop coeffs and components based on pca_k
        self.coefficients = self.coefficients[:, :, :self.pca_k]
        self.components = self.components[:self.pca_k, :, :]
        
        # 5. Precompute FFTs
        self.fft_shape = np.array(self.shape) + np.array(self.components.shape[1:]) - 1
        # use next_fast_len to speed up FFTs
        self.fast_fft_shape = [next_fast_len(s) for s in self.fft_shape]
        self.components_fft = np.fft.fft2(self.components, s=self.fast_fft_shape, axes=(1, 2)) # shape (pca_k, ny, nx)
        self.components_fft = self.components_fft.transpose((1, 2, 0)) # shape (ny, nx, pca_k)
        
        Message("Data shapes after reshape:").list({
            "components": self.components.shape,
            "coefficients": self.coefficients.shape,
            "xgrid": self.xgrid.shape,
            "ygrid": self.ygrid.shape,
            "components_fft": self.components_fft.shape,
        })
        
    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the grid (ny, nx). The grid is always square anyway.
        """
        return self.xgrid.shape    
    
    # ------------------- #
    # !-- Propagation --! #
    # ------------------- #
    
    def forward(
        self,
        image: np.ndarray
    ):
        """
        Propagates a 2D image through the PSF grid, by convolving (or rather correlating) the image with
        the PSF at each position of the grid.
        """
        
        # 1. Check input
        assert image.ndim == 2, f"Input image should be 2D, but got shape {image.shape}."
        assert image.shape == self.shape, f"Input image shape {image.shape} does not match the grid shape {self.shape}."
        
        # 2. Get the coefficients and components
        coeffs = self.coefficients # shape (ny, nx, pca_k)
        components = self.components # shape (pca_k, psf_my, psf_mx)
        
        # 3. Propagate in coefficient space
        coeffs = coeffs * image[:, :, np.newaxis] # shape (ny, nx, pca_k)
        
        # 4. Correlate with PSFs using fftconvolve from scipy
        out = np.zeros_like(image) # shape (ny, nx)
        for k in range(self.pca_k):
            out += fftconvolve(
                coeffs[:, :, k], # shape (ny, nx)
                components[k], # shape (psf_ny, psf_nx)
                mode="same" # shape of output = shape of input
            )
        return out        
        
        
    
    def forward_opt(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        This function is an improvement over the `forward()` function in terms of speed and
        parallelization. The core idea is that one won't do propagation once but several times,
        which means that we can precompute the Fourier transform of the components.
        
        Moreover, we can take advantage of the linearity of the Fourier transform, to sum the components
        in Fourier space first, and then only do one inverse Fourier transform at the end th get the final
        propagated image. This should be faster.
        """
        
        assert image.ndim == 2, f"Input image should be 2D, but got shape {image.shape}."
        assert image.shape == self.shape, f"Input image shape {image.shape} does not match the grid shape {self.shape}."
        
        # shape of the result of the convolution
        
        # 0. Compute FFTs of the components
        # => will be precomputed and stored in init later
        
        # 1. Get the coefficients
        coeffs = self.coefficients # shape (ny, nx, pca_k)
        coeffs = coeffs * image[:, :, np.newaxis] # shape (ny, nx, pca_k)
        
        # 2. Go to Fourier space
        coeffs_fft = np.fft.fft2(coeffs, s=self.fast_fft_shape, axes=(0, 1)) # shape (ny, nx, pca_k)
        
        # 3. Apply convolution in Fourier space (which is just a multiplication)
        propagated_ffts_per_components = coeffs_fft * self.components_fft # shape (ny, nx, pca_k)
        propagated_fft = np.sum(propagated_ffts_per_components, axis=2) # shape (ny, nx)
        
        # 4. Go back to real space
        propagate = np.fft.ifft2(propagated_fft, s=self.fast_fft_shape).real # shape (ny, nx)
        
        # 5. Center the result and crop to match the input shape
        current_shape = np.array(self.fft_shape)
        initial_shape = np.array(image.shape)
        start_index = (current_shape - initial_shape) // 2
        end_index = start_index + initial_shape
        return propagate[start_index[0]:end_index[0], start_index[1]:end_index[1]]
        
        
    # ------------- #
    # !-- Utils --! #
    # ------------- #
    
    def get_indices(
        self, *args,
        i:int|np.ndarray = None, j:int|np.ndarray = None,
        x:float|np.ndarray = None, y:float|np.ndarray = None,
        x_arcsec:float|np.ndarray = None, y_arcsec:float|np.ndarray = None,
    ) -> tuple:
        """
        Translates a position in the grid, given in either indeces (in which case they are directly returned),
        separation in (oversampled) pixels, relative to the center of the grid, or separation in arcseconds,
        relative to the center of the grid.
        
        Parameters
        ----------
        i, j: int or np.ndarray, optional
            The indices of the position(s) in the grid. Should be between 0 and `self.shape[0] - 1`.
        x, y: float or np.ndarray, optional
            The separation(s) from the center of the grid in the x and y directions, in oversampled pixels.
            These must be half integers (`x%1=0.5`) as the grid is even, hence the center lies between four pixels,
            and therefore the closest pixels to the center are at a separation of 0.5 oversampled pixels in each direction.
        x_arcsec, y_arcsec: float or np.ndarray, optional
            The separation(s) from the center of the grid in the x and y directions, in arcseconds.
            These will be converted to oversampled pixels using the pixel scale of MIRI (0.11" / oversample).
        
        
        Notes
        -----
        This function only accepts keyword arguments, and will raise an error if any positional argument is given.
        
        If numpy arrays are passed, numpy arrays of the same shape are returned.      
        
        Separations in arcseconds will be converted to the closest matching pixel position on the grid. Other arguments
        need to be spot on on a pixel. 
        """
        assert len(args) == 0, f"All arguments should be passed as keyword arguments, but got {len(args)} positional arguments."
        
        # 1. Check that only one type of argument is given
        num_args = sum(arg is not None for arg in [i, j, x, y, x_arcsec, y_arcsec])
        assert num_args > 0, "At least one argument should be given."
        assert num_args == 2, "Exactly two arguments should be given, one for x and one for y. For instance, if i is given, j should also be given, and if x is given, y should also be given, and if x_arcsec is given, y_arcsec should also be given."
        assert (i is not None and j is not None) or (x is not None and y is not None) or (x_arcsec is not None and y_arcsec is not None), "Arguments should be given in pairs, either (i, j), or (x, y), or (x_arcsec, y_arcsec)."
        
        # 2. If arguments are given in arcseconds, convert to oversampled pixels
        if x_arcsec is not None and y_arcsec is not None:
            pixel_scale = 0.11 / self.metadata["oversample"] # arcsec / oversampled pixel
            x = x_arcsec / pixel_scale
            y = y_arcsec / pixel_scale
            
            # round to the closest half integer
            x = np.round(x * 2) / 2
            y = np.round(y * 2) / 2
        
        # 3. If arguments are given in oversampled pixels, convert to indices
        if x is not None and y is not None:
            # check that x and y are half integers
            assert np.all(np.isclose(x % 1, 0.5, rtol=1e-5)), f"x should be half integers, but got {x}."
            assert np.all(np.isclose(y % 1, 0.5, rtol=1e-5)), f"y should be half integers, but got {y}."
            i = np.round(x + (self.shape[1] - 1) / 2).astype(int)
            j = np.round(y + (self.shape[0] - 1) / 2).astype(int)
        
        # 4. Check that indices are integers, and within the grid
        assert isinstance(i, int) or np.issubdtype(i.dtype, np.integer), f"i should be integers, but got {i}."
        assert isinstance(j, int) or np.issubdtype(j.dtype, np.integer), f"j should be integers, but got {j}."
        assert np.all((0 <= i) & (i < self.shape[1])), f"i should be between 0 and {self.shape[1] - 1}, but got {i}."
        assert np.all((0 <= j) & (j < self.shape[0])), f"j should be between 0 and {self.shape[0] - 1}, but got {j}."
        
        return i, j
        
    def grid(
        self, *args,
        i:int = None, j:int = None,
        x:float = None, y:float = None,
        x_arcsec:float = None, y_arcsec:float = None
    ) -> np.ndarray:
        """
        Reconstruct the PSF at a given position on the grid, by simply multiplying
        the PCA components by the coefficients at the given position, and summing them up.
        
        Parameters
        ----------
        i, j, x, y, x_arcsec, y_arcsec
            See `get_indices()`. However, numpy arrays cannot be used here.
        """
        
        # 1. Get the indices corresponding to the given position
        i, j = self.get_indices(
            i=i, j=j,
            x=x, y=y,
            x_arcsec=x_arcsec, y_arcsec=y_arcsec,
        )
        assert isinstance(i, int) and isinstance(j, int) or np.issubdtype(i.dtype, np.integer) and np.issubdtype(j.dtype, np.integer), f"i and j should be integers, but got {i} and {j}."
        
        # 2. Get coefficients
        coeffs = self.coefficients[j, i] # shape (pca_k,)
        
        # 3. Get the components, and multiply by coefficients
        psf = np.sum(coeffs[:, np.newaxis, np.newaxis] * self.components, axis=0) # shape (psf_ny, psf_nx)
        return psf
    
   
    def dirac(
        self, i:int | np.ndarray, j:int | np.ndarray
    ) -> np.ndarray:
        """
        Returns a dirac image (i.e. an image with all pixels equal to 0 except one pixel equal to 1)
        at a given position on the grid, given in oversampled pixels.
        
        Parameters
        ----------
        i : int or np.ndarray
            The x coordinate(s) of the position(s) where the dirac should be centered,
            in indices (not relative to the center of the grid). Should be between 0 and `self.shape[1] - 1`.
        j : int or np.ndarray
            The y coordinate(s) of the position(s) where the dirac should be centered,
            in indices (not relative to the center of the grid). Should be between 0 and `self.shape[0] - 1`.
        
        Returns
        -------
        np.ndarray
            A dirac image of same shape as the grid, with a single pixel equal to 1 at the given position,
            and all other pixels equal to 0. If i and j are arrays, the output will be a stack of dirac images,
            one for each position given by the corresponding elements of i and j.
            
        Notes
        -----
        If you need to create a dirac from a given separation, in oversampled pixels or in arcseconds,
        you can use the `get_indices()` method to convert the separation to indices, and then use this `dirac()`
        method to create the dirac image.
        """
        if np.isscalar(i) and np.isscalar(j):
            dirac = np.zeros(self.shape)
            dirac[int(j), int(i)] = 1
            return dirac
        else:
            if i.shape != j.shape:
                raise ValueError(f"i and j should be both scalars or 1D arrays of the same shape, but got shapes {i.shape} and {j.shape}.")
            N = len(i)
            diracs = np.zeros((N, *self.shape))
            diracs[np.arange(N), j.astype(int), i.astype(int)] = 1
            return diracs
        
      
    
   
        
    
    
    
    
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
    
    mxgrid = MirixGrid(
        "mirix_grids/src/mirix_grids/data/grid_filter-F1140C_nlambda-1_oversample-4_date-2023-09-04_psfshape-183_gridshape-272_psfoversample-2_pca_k-200.fits",
    )
    
    # ---------------------------- #
    # !-- Test convert indices --! #
    # ---------------------------- #
    
    sep_x, sep_y = 20.5, 5.5
    sep_x_np, sep_y_np = np.array([sep_x, sep_x+10]), np.array([sep_y, sep_y+10])
    
    grid_center_y, grid_center_x = (mxgrid.shape[0] - 1) / 2, (mxgrid.shape[1] - 1) / 2
    sep_x_arcsec = sep_x * 0.11 / mxgrid.metadata["oversample"]
    sep_y_arcsec = sep_y * 0.11 / mxgrid.metadata["oversample"]
    sep_x_arcsec_np = sep_x_np * 0.11 / mxgrid.metadata["oversample"]
    sep_y_arcsec_np = sep_y_np * 0.11 / mxgrid.metadata["oversample"]
    i = int(np.round(sep_x + grid_center_x))
    j = int(np.round(sep_y + grid_center_y))
    i_np = np.round(sep_x_np + grid_center_x).astype(int)
    j_np = np.round(sep_y_np + grid_center_y).astype(int)
    
    Message.title("Test: get_indices()")
    Message("Inputs and outputs for get_indices():").list({
        "Separation in pixels": f"{sep_x, sep_y} -> {mxgrid.get_indices(x=sep_x, y=sep_y)}",
        "Separation in pixels (numpy)": f"{sep_x_np, sep_y_np} -> {mxgrid.get_indices(x=sep_x_np, y=sep_y_np)}",
        "Separation in arcseconds": f"{sep_x_arcsec:.2f}\", {sep_y_arcsec:.2f}\" -> {mxgrid.get_indices(x_arcsec=sep_x_arcsec, y_arcsec=sep_y_arcsec)}",
        "Separation in arcseconds (numpy)": f"{sep_x_arcsec_np}, {sep_y_arcsec_np} -> {mxgrid.get_indices(x_arcsec=sep_x_arcsec_np, y_arcsec=sep_y_arcsec_np)}",
        "Separation in indices": f"{i, j} -> {mxgrid.get_indices(i=i, j=j)}",
        "Separation in indices (numpy)": f"{i_np, j_np} -> {mxgrid.get_indices(i=i_np, j=j_np)}"
    })
    
    
    # ------------------- #
    # !-- Test diracs --! #
    # ------------------- #
    
    Message.title("Test: dirac()")
    i, j = mxgrid.get_indices(x=sep_x_np, y=sep_y_np)
    diracs = mxgrid.dirac(i=i, j=j) # here i and j are arrays, so we should get a stack of dirac images
    Message(f"Shape of diracs: {cstr(diracs.shape):cb} (should be (2, {mxgrid.shape[0]}, {mxgrid.shape[1]}))")
    
    # let's make a plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_image2d(
        diracs[0],
        qmin=0,
        qmax=1,
        log_scale=False,
        colormap="plasma",
        colorbar_shrink=0.8,
        colorbar_label="Pixel value",
        pixel_scale_arcsec=0.11 / mxgrid.metadata["oversample"],
    )
    plt.title(f"Dirac at x={sep_x_arcsec_np[0]}\", y={sep_y_arcsec_np[0]}\"")
    plt.scatter(sep_x_arcsec_np[0], sep_y_arcsec_np[0], color="cyan", marker="o", facecolor="none", s=20, label="Dirac position")
    plt.legend()
    plt.subplot(1, 2, 2)
    plot_image2d(
        diracs[1],
        qmin=0,
        qmax=1,
        log_scale=False,
        colormap="plasma",
        colorbar_shrink=0.8,
        colorbar_label="Pixel value",
        pixel_scale_arcsec=0.11 / mxgrid.metadata["oversample"],
    )
    plt.title(f"Dirac at x={sep_x_arcsec_np[1]}\", y={sep_y_arcsec_np[1]}\"")
    plt.scatter(sep_x_arcsec_np[1], sep_y_arcsec_np[1], color="cyan", marker="o", facecolor="none", s=20, label="Dirac position")
    plt.legend()
    plt.tight_layout()
    plt.savefig("diracs.png")
    plt.show()
    
    
    # ----------------- #
    # !-- Test Grid --! #
    # ----------------- #
    
    Message.title("Test: grid()")
    x1, y1 = 0.5, 0.5 # a centered PSF
    x2, y2 = 10.5, 10.5 # a PSF far from the center
    psf1 = mxgrid.grid(x=x1, y=y1)
    psf2 = mxgrid.grid(x=x2, y=y2)
    
    # ------------------------ #
    # !-- Test propagation --! #
    # ------------------------ #
    
    dirac1 = mxgrid.dirac(*mxgrid.get_indices(x=x1, y=y1))
    dirac2 = mxgrid.dirac(*mxgrid.get_indices(x=x2, y=y2))
    propagated1 = mxgrid.forward(dirac1)
    propagated2 = mxgrid.forward(dirac2)
    
    # zoom on propagated1 and 2 to match the shape of dirac1 and dirac2
    propagated1 = crop_to(propagated1, psf1.shape[0] - 1)
    propagated2 = crop_to(propagated2, psf2.shape[0] - 1)
    
    
    
    plt.figure(figsize=(10, 15))
    plt.subplot(3, 2, 1)
    plot_image2d(
        psf1,
        qmin=0,
        qmax=1,
        log_scale=False,
        colormap="plasma",
        colorbar_shrink=0.8,
        colorbar_label="Normalized flux",
        pixel_scale_arcsec=0.11 / mxgrid.metadata["oversample"],
    )
    plt.title(f"PSF at x={x1}, y={y1}")
    plt.scatter(0, 0, color="cyan", marker="+", s=100, label="Center of the grid")
    #plt.legend()
    plt.subplot(3, 2, 2)
    plot_image2d(   
        psf2,
        qmin=0,
        qmax=1,
        log_scale=False,
        colormap="plasma",
        colorbar_shrink=0.8,
        colorbar_label="Normalized flux",
        pixel_scale_arcsec=0.11 / mxgrid.metadata["oversample"],
    )
    plt.title(f"PSF at x={x2}, y={y2}")
    plt.scatter(0, 0, color="cyan", marker="+", s=100, label="Center of the grid")
    #plt.legend()
    
    plt.subplot(3, 2, 3)
    plot_image2d(
        propagated1,
        qmin=0,
        qmax=1,
        log_scale=False,
        colormap="plasma",
        colorbar_shrink=0.8,
        colorbar_label="Normalized flux",
        pixel_scale_arcsec=1,
    )
    plt.title(f"Propagated dirac at x={x1}, y={y1}")
    plt.scatter(x1, y1, color="red", marker="+", s=100, label="Dirac position")
    plt.scatter(0, 0, color="cyan", marker="+", s=100, label="Center of the grid")
    #plt.legend()
    plt.subplot(3, 2, 4)
    plot_image2d(
        propagated2,    
        qmin=0,
        qmax=1,
        log_scale=False,
        colormap="plasma",
        colorbar_shrink=0.8,
        colorbar_label="Normalized flux",
        pixel_scale_arcsec=1,
    )
    plt.title(f"Propagated dirac at x={x2}, y={y2}")   
    plt.scatter(0, 0, color="cyan", marker="+", s=100, label="Center of the grid")
    plt.scatter(x2, y2, color="red", marker="+", s=100, label="Dirac position")
    plt.legend()
    
    # let's check the difference => we need to crop one row and one column again to match the shapes
    psf1 = psf1[1:, 1:]
    psf2 = psf2[1:, 1:]
    # we also need to roll the porpagated images by the number of pixels
    propagated1 = np.roll(propagated1, shift=(-int(y1)-1, -int(x1)-1), axis=(0, 1))
    propagated2 = np.roll(propagated2, shift=(-int(y2)-1, -int(x2)-1), axis=(0, 1))
    
    diff1 = (propagated1 - psf1) / np.max(psf1)
    diff2 = (propagated2 - psf2) / np.max(psf2)
    
    # let's further crop to exclude the boerders where numpy rolls
    crop = diff1.shape[0] - int(max(y1, y2, x1, x2)+1)*2
    diff1 = crop_to(diff1, crop)
    diff2 = crop_to(diff2, crop)
    
    plt.subplot(3, 2, 5)
    plot_image2d(
        diff1,  
        qmin=0,
        qmax=1,
        log_scale=False,
        colormap="bwr",
        colorbar_shrink=0.8,
        colorbar_label="Difference in flux",
        pixel_scale_arcsec=1,
    )
    plt.title(f"Relative difference at x={x1}, y={y1}")
    
    plt.subplot(3, 2, 6)
    plot_image2d(
        diff2,  
        qmin=0,
        qmax=1,
        log_scale=False,
        colormap="bwr",
        colorbar_shrink=0.8,
        colorbar_label="Difference in flux",
        pixel_scale_arcsec=1,
    )
    plt.title(f"Relative difference at x={x2}, y={y2}")
    
    
    plt.tight_layout()
    plt.savefig("psfs.png")
    plt.show()
    
    