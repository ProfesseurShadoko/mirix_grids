

# --------------- #
# !-- Imports --! #
# --------------- #

from oakley import *
import numpy as np
from .propagation import MirixGrid
import torch

class MirixGridGPU(MirixGrid):
    """
    An implementation of the `MirixGrid` class optimized for GPU computation
    and autodifferentiability using `torch`.
    """
    
    def __init__(
        self,
        filepath:str,
        pca_k:int = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        complex_dtype: torch.dtype = torch.complex64
    ):
        """
        Initializes a `MirixGridGPU` instance by loading the grid from a FITS file,
        and putting it on the specified device (a few tens of hundreds of MBs of GPU memory,
        this implementation is intended to be as memory efficient as possible).
        
        Parameters
        ----------
        filepath : str
            The path to thz FITS file containing the grid. If the file does not exist in the
            working directory, the function will look for it in the default folder, where 
            `mirix_grids` is installed (where grids are downloaded from `Github`).
        pca_k : int, optional
            The number of PCA components to use to reconstruct PSFs. If `None`, all components present
            in the file are used. Defaults to `None`. A higher number of components will lead to a more
            accurate reconstruction of the PSFs, but will also lead to a higher memory usage and
            computational cost. 200 components is already more than enough, even at the center of the grid
            (where the reconstruction is most difficult).
        device : str or torch.device, optional
            The device on which the grid should be loaded. Can be a string (e.g., "cpu", "cuda:0", "cuda:1", etc.)
            or a `torch.device` object. Defaults to "cpu". This much match the image(s) passed to the `forward()` method.
        dtype : torch.dtype, optional
            The data type of the grid. Defaults to `torch.float32`. This much match the image(s) passed to the `forward()` method.
            `torch.float32` is optimal for GPUs, and any lower precision would be detrimental to the accuracy of the results,
            which rely heavily on the precision of the FFT.
        complex_dtype : torch.dtype, optional
            The complex data type to use for the Fourier space representation of the grid. Defaults to `torch.complex64`,
            which is the complex counterpart of `torch.float32`.
        """
        super().__init__(filepath, pca_k)
        
        # 1. Convert everything necessary to tensors
        # => we need components and coefficients
        self.components_fft = torch.from_numpy(self.components_fft).to(device=device, dtype=complex_dtype) # shape (pca_k, ny, nx)
        self._batch_first_coeffs = torch.from_numpy(self._batch_first_coeffs).to(device=device, dtype=dtype) # shape (pca_k, ny, nx)
    
    @property
    def device(self):
        """
        The device on which the grid is loaded.
        """
        return self.components_fft.device
    
    @property
    def dtype(self):
        """
        The data type of the grid.
        """
        return self.components_fft.dtype
    
    @property
    def complex_dtype(self):
        """
        The complex data type of the grid.
        """
        return self.components_fft.dtype
    
    # --------------- #
    # !-- Forward --! #
    # --------------- #
    
    def forward(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagates an image (or a batch of images) through the grid, by convolving it with the
        components of the PSF grid.
        
        Parameters
        ----------
        image : torch.Tensor
            The image (or batch of images, at most 3 dimensions, with shape (batch_size, ny, nx) or (ny, nx)) to propagate through the grid.
            The image(s) should be on the same device and have the same dtype as the grid.
    
        Returns
        -------
        torch.Tensor
            The propagated image(s), with the same shape as the input image(s).
        """
        
        # 1. Check inputs
        assert isinstance(image, torch.Tensor), f"Input image should be a torch.Tensor, but got {type(image)}."
        assert image.device == self.components_fft.device, f"Input image should be on the same device as the grid (got {image.device} and {self.components_fft.device})."
        assert image.dtype == self.components_fft.dtype, f"Input image should have the same dtype as the grid (got {image.dtype} and {self.components_fft.dtype})."
        assert image.shape[-2:] == self.shape, f"Input image should have the same spatial dimensions as the grid (got {image.shape[-2:]} and {self.shape})."
        
        initial_ndim = image.ndim
        assert initial_ndim in [2, 3], f"Input image should have 2 or 3 dimensions, but got {initial_ndim}."
        if initial_ndim == 2:
            image = image.unsqueeze(0) # add batch dimension
        
        # 2. Propagate to coefficient space (simple multiplication by coefficients)
        # (b, ny, nx) * (pca_k, ny, nx) -> (b, pca_k, ny, nx)
        coeffs = image[:, None, :, :] * self._batch_first_coeffs[None, :, :, :]
        
        # 3. Propagate from coefficient space to Fourier space
        coeffs_fft:torch.Tensor = torch.fft.fft2(coeffs, dim=(-2, -1), s=self.fast_fft_shape)
        
        # 4. Convolve with components in Fourier space (simple multiplication)
        # (b, pca_k, ny, nx) * (pca_k, ny, nx) -> (b, pca_k, ny, nx)
        convolved_fft = coeffs_fft * self.components_fft[None, :, :, :]
        
        # 5. Combine components (sum over pca_k dimension)
        psf_fft = convolved_fft.sum(dim=1) # shape (b, ny, nx)
        
        # 6. Inverse FFT to get the propagated image
        propagated = torch.fft.ifft2(psf_fft, dim=(-2, -1), s=self.fast_fft_shape).real # shape (b, ny, nx)
        
        # 7. Crop to original size
        current_shape = np.array(self.fft_shape) # don't ask why its not fast_fft_shape, idk
        original_shape = np.array(self.shape)
        start = (current_shape - original_shape) // 2
        end = start + original_shape
        propagated = propagated[:, start[0]:end[0], start[1]:end[1]]
        
        # 7. Remove batch dimension if the input was 2D
        if initial_ndim == 2:
            propagated = propagated.squeeze(0)
        return propagated
    
    # ------------- #
    # !-- Utils --! #
    # ------------- #
    
    def forward_np(
        self,
        image:np.ndarray
    ):
        """
        Propagates an image (or a batch of images) with the `forward()` method,
        by creating the appripriate `torch.Tensor` object.
        
        Parameters
        ----------
        image : np.ndarray
            The image (or set of images). Shape is expected to be (..., ny, nx).
        """
        
        # 1. Check inputs
        input_shape = image.shape
        assert input_shape[-2:] == self.shape, f"Input image should have the same spatial dimensions as the grid components (got {input_shape[-2:]} and {self.shape})."

        # 2. Convert to tensor
        image_tensor = torch.from_numpy(image).to(self.components_fft.device).to(self.components_fft.dtype)
        # handle the shape, flatten all dimensions except the last two into a batch dimension
        initial_shape = image_tensor.shape
        spatial_shape = initial_shape[-2:]
        image_tensor = image_tensor.reshape(-1, *spatial_shape) # shape (batch_size, ny, nx)
        
        # 3. Propagate
        propagated_tensor = self.forward(image_tensor) # shape (batch_size, ny, nx)
        
        # 4. Reshape to original shape
        propagated = propagated_tensor.reshape(initial_shape) # shape (..., ny, nx)
        return propagated.cpu().numpy()