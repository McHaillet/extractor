from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("extractor")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Marten Chaillet"
__email__ = "martenchaillet@gmail.com"
