# cuMemPool

## Buld & Install
```bash
git clone git@github.com:SiriusInfTra/mpool.git
cd mpool
pip install -r requirements.txt
mkdir build
cd build
cmake ..
cmake --build .
cd ..
pip install -e ./python
```

## Get Started
### PyTorch API
Example
```bash
python example/pytorch/finetune_swin.py
```
### Numba / CuPy
```
python/mpool/numba_integrate.py
python/mpool/cupy_integrate.py
```

### C++ API
```cpp
#include <mpool/pages_pool.h>
#include <mpool/caching_allocator.h>
#include <mpool/direct_allocator.h>
struct PagesPoolConf {
  int device_id;
  size_t page_nbytes;
  size_t pool_nbytes;
  std::string shm_name;
  std::string log_prefix;
  size_t shm_nbytes;
};
struct VMMAllocatorConfig {
  std::string log_prefix;
  std::string shm_name;
  size_t shm_nbytes;
  size_t va_range_scale;
  std::string belong_name;
  size_t small_block_nbytes;
  size_t align_nbytes;
};
// Create PagesPool / CachingAllocator in shared memory
auto pages_pool = new SharableObject<PagesPool>{
                 conf.shm_name, conf.shm_nbytes, conf};
auto *caching_allocator = new SharableObject<CachingAllocator>{
                 conf.shm_name, conf.shm_nbytes, pages_pool, conf};
```
