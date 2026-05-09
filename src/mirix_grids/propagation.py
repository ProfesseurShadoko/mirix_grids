


# todo:
# make every function take no positional arguments
# only kwargs
# i,j means indeces
# x,y means separation in oversampled pixels, with (0,0) the center of the grid
# and in arcsec is x_arcsec, y_arcsec, with (0,0) the center of the grid as well


# --------------- #
# !-- Imports --! #
# --------------- #

from oakley import *
import numpy as np
from scipy.signal import fftconvolve
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
        # crop components based on psf_size
        #psf_center = self.metadata["psf_shape"] // 2
        #psf_half_size = self.psf_size // 2
        #self.components = self.components[:, psf_center - psf_half_size : psf_center + psf_half_size + 1, psf_center - psf_half_size : psf_center + psf_half_size + 1]
        # crop coefficients and grids based on grid_size
        #grid_center = self.metadata["gridsize"] // 2
        #grid_half_size = self.grid_size // 2
        #self.coefficients = self.coefficients[:, grid_center - grid_half_size : grid_center + grid_half_size, grid_center - grid_half_size : grid_center + grid_half_size]
        #self.xgrid = self.xgrid[grid_center - grid_half_size : grid_center + grid_half_size, grid_center - grid_half_size : grid_center + grid_half_size]
        #self.ygrid = self.ygrid[grid_center - grid_half_size : grid_center + grid_half_size, grid_center - grid_half_size : grid_center + grid_half_size]
        
        Message("Data shapes after reshape:").list({
            "components": self.components.shape,
            "coefficients": self.coefficients.shape,
            "xgrid": self.xgrid.shape,
            "ygrid": self.ygrid.shape,
        })
        
        # 5. Everyone becomes a torch tensor
        # self.components = torch.tensor(self.components, device=device, dtype=dtype)
        # self.coefficients = torch.tensor(self.coefficients, device=device, dtype=dtype)
    
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
        #image: torch.Tensor
        image: np.ndarray
    ) -> np.ndarray:
        """
        Propagates an image through the PSF grid, by convolving (or rather correlating) the image with
        the PSF at each position of the grid. The output is an image of same shape as the input image.
        
        Parameters
        ----------
        image : np.ndarray
            The input image to be propagated through the grid. Can be of any shape, but the last two dimensions
            should correspond respectively to the y and x dimensions of the image, and match the grid shape.
        pca_k : int, optional
            The number of PCA components to use for the propagation. If None, all components will be used.
        
        Returns
        -------
        np.ndarray
            The propagated image, of same shape as the input image.
        """
        #return image
        
        # 1. Check the shape of the input image
        if image.shape[-2:] != self.shape:
            raise ValueError(f"Input image shape {image.shape[-2:]} does not match the grid shape {self.shape}.")
        
        # 3. Flatten the image to a batch of images
        image_flat = image.reshape(-1, 1, *self.shape) # shape (batch_size, 1, ny, nx)
        
        # 4. Multiply coefficients
        coeffs = self.coefficients
        coeffs = image_flat * coeffs.reshape(1, self.pca_k, *self.shape) # shape (batch_size, pca_k, ny, nx)
        
        # 5. Correlate (yes conv2d correlates without flipping kernels)
        #out = F.conv2d(
        #    coeffs, # shape (batch_size, pca_k, ny, nx)
        #    self.components.reshape(self.pca_k, 1, *self.components.shape[-2:]), # shape (pca_k, 1, psf_ny, psf_nx)
        #    padding="same", # shape of output = shape of input
        #    groups=self.pca_k, # each component is convolved with its own PSF component
        #)
        
        # 5. Correlate with scipy
        out = np.zeros_like(image).reshape(-1, self.shape[0], self.shape[1]) # shape (batch_size, ny, nx)
        
        with Task("Propagating image..."):
            for b in range(out.shape[0]):
                for k in range(self.pca_k):
                    out[b] += fftconvolve(
                        coeffs[b, k], # shape (ny, nx)
                        self.components[k][::-1, ::-1], # shape (psf_ny, psf_nx) # flip the kernel to get correlation instead of convolution
                        mode="same" # shape of output = shape of input
                    )        
        
        # 6. Reshape the output to match the input image shape
        out = out.reshape(*image.shape)
        return out

    
    # ------------- #
    # !-- Utils --! #
    # ------------- #
    
    def grid(
        self, *args,
        i:int = None, j:int = None,
        x:float = None, y:float = None,
        x_arcsec:float = None, y_arcsec:float = None
    ) -> np.ndarray:
        """
        Reconstruct the PSF at a given position on the grid, by simply multiplying
        the PCA components by the coefficients at the given position
        RESUME HERE
        
        
        Returns the PSF at a given position on the grid, given by its indices x_index and y_index.
        The separation from the center of the grid can be obtained by looking at the xgrid
        and ygrid attributes, which give the separation in oversampled pixels.
        
        Parameters
        ----------
        x_index : int
            The index of the grid in the x direction. Should be between 0 and the number of pixels in x direction - 1.
        y_index : int
            The index of the grid in the y direction. Should be between 0 and the number of pixels in y direction - 1.
        """
        assert len(args) == 0, f"All arguments should be passed as keyword arguments, but got {len(args)} positional arguments."
        assert 0 <= x_index < self.shape[1], f"x_index should be between 0 and {self.shape[1]-1}, but got {x_index}."
        assert 0 <= y_index < self.shape[0], f"y_index should be between 0 and {self.shape[0]-1}, but got {y_index}."
        
        # 1. Get coefficients at the given position
        coeffs = self.coefficients[y_index, x_index] # shape (pca_k,)
        
        # 2. Get the components, and multiply by coefficients
        psf = np.sum(coeffs[:, np.newaxis, np.newaxis] * self.components, axis=0) # shape (psf_ny, psf_nx)
        return psf
    
    def grid_from_separation(
        self, x_sep:float, y_sep:float,
        pixel2unit:float = 1
    ) -> np.ndarray:
        """
        Returns the PSF at a given separation from the center of the grid, in oversampled pixels. The separation can be converted to arcseconds by multiplying by the pixel scale of MIRI (0.11" / oversample).
        
        Parameters
        ----------
        x_sep : float
            The separation from the center of the grid in the x direction, in oversampled pixels.
        y_sep : float
            The separation from the center of the grid in the y direction, in oversampled pixels.
        pixel2unit : float, optional
            The conversion factor from (oversampled) pixels to the desired unit for x_sep and y_sep.
            For instance, if x_sep and y_sep are given in arcseconds, and the pixel scale is 0.11" / oversample,
            then pixel2unit should be 0.11 / oversample.
        """
        # 1. Get indices corresponding to the given separation
        x_index = int(np.round(x_sep / pixel2unit + (self.shape[1] - 1) / 2))
        y_index = int(np.round(y_sep / pixel2unit + (self.shape[0] - 1) / 2))
        
        return self.grid(x_index, y_index)
    
    def dirac(
        self, i:int | np.ndarray, j:int | np.ndarray
    ) -> np.ndarray:
        """
        Returns a dirac image (i.e. an image with all pixels equal to 0 except one pixel equal to 1)
        at a given position on the grid, given in oversampled pixels.
        
        Parameters
        ----------
        x : int or np.ndarray
            The x coordinate(s) of the position(s) where the dirac should be centered,
            in oversampled pixels, in indeces (not relative to the center of the grid).
        y : int or np.ndarray
            The y coordinate(s) of the position(s) where the dirac should be centered,
            in oversampled pixels, in indeces (not relative to the center of the grid).    
        
        Returns
        -------
        np.ndarray
            A dirac image of same shape as the grid, with a single pixel equal to 1 at the given position,
            and all other pixels equal to 0. If x and y are arrays, the output will be a stack of dirac images,
            one for each position given by the corresponding elements of x and y.
        """
        if np.isscalar(x) and np.isscalar(y):
            dirac = np.zeros(self.shape)
            dirac[int(y), int(x)] = 1
            return dirac
        else:
            if x.shape != y.shape:
                raise ValueError(f"x and y should be both scalars or 1D arrays of the same shape, but got shapes {x.shape} and {y.shape}.")
            N = len(x)
            diracs = np.zeros((N, *self.shape))
            diracs[np.arange(N), y.astype(int), x.astype(int)] = 1
            return diracs
        
   
    def sgd_plot(self) -> None:
        """
        A small debug function. Plots the four PSFs closest to the center of the grid.
        """
        top_left = [-0.5, 0.5]
        top_right = [0.5, 0.5]
        bottom_left = [-0.5, -0.5]
        bottom_right = [0.5, -0.5]
        #psfs = self.psf(
        #    x = np.array([top_left[0], top_right[0], bottom_left[0], bottom_right[0]]),
        #    y = np.array([top_left[1], top_right[1], bottom_left[1], bottom_right[1]]),
        #)
        diracs = self.dirac(
            x = np.array([top_left[0], top_right[0], bottom_left[0], bottom_right[0]]),
            y = np.array([top_left[1], top_right[1], bottom_left[1], bottom_right[1]]),
        )
        psfs = self.forward(diracs)
        Message(f"Shape of PSFs: {psfs.shape}")
        return
        
        plt.figure(figsize=(15, 10))
        for i, coords in enumerate([top_left, top_right, bottom_left, bottom_right]):
            plt.subplot(2, 2, i + 1)
            plot_image2d(
                psfs[i],
                qmin=0,
                qmax=1,
                log_scale=False,
                colormap="plasma",
                colorbar_shrink=0.8,
                colorbar_label="Normalized flux",
                pixel_scale_arcsec=0.11 / self.metadata["oversample"],
            )
            plt.title(f"PSF at x={coords[0]}, y={coords[1]}")
            
            plt.scatter(0, 0, color="cyan", marker="+", s=100, label="Center of the grid")
        plt.tight_layout()
        plt.savefig("sgd_plot.png")
        plt.show()
        
        
        
    
   
        
    
    
    
    
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
    
    separations = [0.5, 0.5]
    # find the i, j indices corresponding to these separations
    radial_distances = np.sqrt((mxgrid.xgrid - separations[0])**2 + (mxgrid.ygrid - separations[1])**2)
    i,j = np.unravel_index(np.argmin(radial_distances), radial_distances.shape)
    
    psf = mxgrid.grid(i, j)
    
    plt.figure(figsize=(5, 5))
    plot_image2d(   
        psf,
        qmin=0,
        qmax=1,
        log_scale=False,
        colormap="plasma",
        colorbar_shrink=0.8,
        colorbar_label="Normalized flux",
        pixel_scale_arcsec=0.11 / mxgrid.metadata["oversample"],
    )
    #plt.scatter(0, 0, color="cyan", marker="+", s=100, label="Center of the grid")
    #plt.legend()
    plt.tight_layout()
    plt.savefig("test_psf.png")
    plt.show()
    
    # create a dirac image at the position of the psf
    image = mxgrid.dirac(x=j, y=i)
    
    psf = mxgrid.forward(
        im
    )
    