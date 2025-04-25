"""Top-level package for fastgan_comfyui."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """fastgan"""
__email__ = "berard.hugo@gmail.com"
__version__ = "0.0.1"

from .src.fastgan_comfyui.nodes import NODE_CLASS_MAPPINGS
from .src.fastgan_comfyui.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
