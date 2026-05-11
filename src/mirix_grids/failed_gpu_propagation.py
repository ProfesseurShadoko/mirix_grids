

class GpuNotAvailableError(ImportError):
    def __init__(self):
        return super().__init__("GPU propagation is not available. Please install PyTorch to use this feature, and reload the package.")


class _GPUBlocked(type):
    """
    A Metaclass that raises an error when used.
    This is used to block any attempt to use MirixGridGPU when
    PyTorch is not installed.
    """
    def __getattribute__(cls, name):
        raise GpuNotAvailableError()
    
    def __getattr__(self, name):
        raise GpuNotAvailableError()

class BlockedMirixGridGPU(metaclass=_GPUBlocked):
    
    def __init__(self, *args, **kwargs):
        raise GpuNotAvailableError()

    
    