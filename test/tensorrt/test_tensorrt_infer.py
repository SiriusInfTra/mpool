import os
from typing import Optional
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # must import
import numpy as np
import mpool
from argparse import ArgumentParser

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

mpool_allocator: Optional[mpool.C_CachingAllocator] = None


def allocate_memblock(nbytes, cuda_stream) -> int:
    if mpool_allocator is not None:
        mem_block = mpool_allocator.alloc(nbytes, cuda_stream.handle, True)
        addr = mpool_allocator.base_ptr + mem_block.addr_offset
        return addr
    else:
        return cuda.mem_alloc(nbytes)

def create_builder():
    builder = trt.Builder(TRT_LOGGER)
    if mpool_allocator is not None:
        mpool.set_igpu_allocator(builder, [mpool_allocator])
    return builder

def build_engine(onnx_file_path):
    builder = create_builder()
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 * 1024 * 1024 * 1024)
    
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    
    engine_bytes = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        shape = engine.get_tensor_shape(binding)
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(binding)
        dtype = np.dtype(trt.nptype(trt_type))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = allocate_memblock(host_mem.nbytes, stream)
        
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        
        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    return inputs, outputs, bindings, stream

def do_inference(context, engine, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    # Run inference.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    context.execute_async_v3(stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    # Synchronize the stream
    stream.synchronize()
    return outputs[0]['host']

def main():
    parser = ArgumentParser()
    parser.add_argument('--use-mpool', action='store_true')
    args = parser.parse_args()
    if args.use_mpool:
        global mpool_allocator
        mpool_allocator = mpool.create_caching_allocator('test_tensorrt', 0, 12 * 1024 * 1024 * 1024)
        print('Use mpool.')
    else:
        print('Not use mpool.')
    onnx_file_path = os.path.join(os.path.dirname(__file__), "resnet152.onnx")
    engine = build_engine(onnx_file_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Assume input is a numpy array of the right shape
    np.random.seed(0)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    print(input_data)
    np.copyto(inputs[0]['host'], input_data.ravel())

    result = do_inference(context, engine, bindings, inputs, outputs, stream)
    print(f'Result {result[:10]}')

if __name__ == '__main__':
    main()
