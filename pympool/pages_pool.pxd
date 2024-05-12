# cython: c_string_type=unicode, c_string_encoding=utf8
from libcpp.string cimport string

cdef extern from "pages_pool.h":
    cdef struct PagesPoolConf:
        size_t page_nbytes
        size_t pool_nbytes
        string shm_name
        string log_prefix
        size_t shm_nbytes
