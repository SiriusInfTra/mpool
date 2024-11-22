#include <mpool/shm.h>
#include <mpool/util.h>
#include <mpool/pages.h>

#include <sys/socket.h>
#include <sys/un.h>

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include <cuda.h>
#include <glog/logging.h>

namespace mpool {
}