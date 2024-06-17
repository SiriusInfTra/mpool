
import ctypes
from cuda.cuda import CUdeviceptr
from numba import config, cuda
from numba.cuda import HostOnlyCUDAMemoryManager, MemoryPointer
import mpool
from mpool import C_CachingAllocator, C_RAIIMemBlock
import numba.cuda


class MPoolAllocator(HostOnlyCUDAMemoryManager):
    
    def __init__(self, caching_allocator: C_CachingAllocator, context):
        super().__init__(context=context)
        self._caching_allocator = caching_allocator
        self._context = context


    def initialize(self):
        pass

    def memalloc(self, size):
        """
        Allocate an on-device array from the RMM pool.
        """
        stream = numba.cuda.default_stream()
        mem = self._caching_allocator.Alloc(size, stream, True)
        addr = self._caching_allocator.base_ptr + mem.addr_offset
        numba.cuda.IpcHandle()

        if config.CUDA_USE_NVIDIA_BINDING:
            ptr = CUdeviceptr(addr)
        else:
            # expect ctypes bindings in numba
            ptr = ctypes.c_uint64(addr)

        self.allocations[addr] = mem
        
        def finalizer():
            self._caching_allocator.Free(mem)
            del self.allocations[addr]

        return MemoryPointer(self._context, ptr, size, finalizer=finalizer)

    def get_ipc_handle(self, memory):
        raise NotImplementedError()
    
    def get_memory_info(self):
        raise NotImplementedError()

    @property
    def interface_version(self):
        return 1
    

def override_numba_allocator(caching_allocator: C_CachingAllocator):
    def plugin(context):
        return MPoolAllocator(caching_allocator, context)
    numba.cuda.set_memory_manager(plugin)