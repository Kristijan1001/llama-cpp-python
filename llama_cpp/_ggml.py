"""Internal module use at your own risk

This module provides a minimal interface for working with ggml tensors from llama-cpp-python
"""
import enum
import os
import pathlib

import llama_cpp._ctypes_extensions as ctypes_ext

libggml_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
libggml = ctypes_ext.load_shared_library("ggml", libggml_base_path)

# enum ggml_log_level {
#     GGML_LOG_LEVEL_NONE  = 0,
#     GGML_LOG_LEVEL_DEBUG = 1,
#     GGML_LOG_LEVEL_INFO  = 2,
#     GGML_LOG_LEVEL_WARN  = 3,
#     GGML_LOG_LEVEL_ERROR = 4,
#     GGML_LOG_LEVEL_CONT  = 5, // continue previous log
# };

class GGMLLogLevel(enum.IntEnum):
    GGML_LOG_LEVEL_NONE = 0
    GGML_LOG_LEVEL_DEBUG = 1
    GGML_LOG_LEVEL_INFO = 2
    GGML_LOG_LEVEL_WARN = 3
    GGML_LOG_LEVEL_ERROR = 4
    GGML_LOG_LEVEL_CONT = 5 # continue previous log
