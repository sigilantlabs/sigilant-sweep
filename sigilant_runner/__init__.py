from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sigilant-sweep")
except PackageNotFoundError:
    # Local source tree before package install.
    __version__ = "0.0.0+local"
