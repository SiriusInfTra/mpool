import mpool

def override_cupy_allocator(caching_allocator):
    import cupy
    def cupy_allocator(nbytes):
        mem_block = caching_allocator.Alloc(nbytes, cupy.cuda.get_current_stream().ptr, True)
        addr = caching_allocator.base_ptr + mem_block.addr_offset
        mem = cupy.cuda.UnownedMemory(
            ptr=addr, 
            size=mem_block.nbytes, 
            owner=mpool.C_RAIIMemBlock(caching_allocator, mem_block), 
            device_id=cupy.cuda.device.get_device_id()
        )
        ptr = cupy.cuda.memory.MemoryPointer(mem, 0)
        return ptr
    cupy.cuda.set_allocator(cupy_allocator)
        
    