#include "shm.h"
#include <atomic>
#include <thread>
#include <util.h>
#include <pages.h>


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

void HandleTransfer::SendHandles(int fd_list[], size_t len, bip::scoped_lock<bip_mutex> &lock) {
    int socket_fd;
    struct sockaddr_un server_addr;

    // create socket and bind
    CHECK_GE(socket_fd = socket(AF_UNIX, SOCK_DGRAM, 0), 0) << "[mempool] Socket creat fail.";

    bzero(&server_addr, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;

    unlink(master_name_.c_str());
    strncpy(server_addr.sun_path, master_name_.c_str(), master_name_.size());

    CHECK_EQ(bind(socket_fd, (struct sockaddr *)&server_addr, SUN_LEN(&server_addr)), 0) 
        << "[mempool] Bind error.";

    // send to client
    ready_cond_->wait(lock);

    struct msghdr msg;
    struct iovec iov[1];


    std::vector<std::byte> control_un(CMSG_SPACE(len * sizeof(int)));

    struct cmsghdr *cmptr;
    struct sockaddr_un client_addr;
    bzero(&client_addr, sizeof(client_addr));
    client_addr.sun_family = AF_UNIX;
    strncpy(client_addr.sun_path, slave_name_.c_str(), slave_name_.size());

    msg.msg_control = control_un.data();
    msg.msg_controllen = control_un.size();

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(len * sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;


    memcpy(CMSG_DATA(cmptr), fd_list, len * sizeof(int));

    msg.msg_name = (void *)&client_addr;
    msg.msg_namelen = sizeof(struct sockaddr_un);
    iov[0].iov_base = (void *)"";
    iov[0].iov_len = len;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t send_result = sendmsg(socket_fd, &msg, 0);
    CHECK_GE(send_result, 0) << "[mempool] Send msg fail.";

    // close socket
    unlink(master_name_.c_str());
    close(socket_fd);
}

void HandleTransfer::ReceiveHandle(int fd_list[], size_t len) {
    int socket_fd;
    struct sockaddr_un client_addr;
    CHECK_GE(socket_fd = socket(AF_UNIX, SOCK_DGRAM, 0), 0) << "[mempool] Socket creat fail.";

    bzero(&client_addr, sizeof(client_addr));
    client_addr.sun_family = AF_UNIX;

    unlink(slave_name_.c_str());
    strncpy(client_addr.sun_path, slave_name_.c_str(), slave_name_.size());
    CHECK_EQ(bind(socket_fd, (struct sockaddr *)&client_addr, SUN_LEN(&client_addr)), 0) 
                    << "[mempool] Bind fail.";

    // recv from server
    {
        bip::scoped_lock lock{*ready_mutex_};
        ready_cond_->notify_all();
    }

    struct msghdr msg = {0};
    struct iovec iov[1];

    std::vector<std::byte> control_un(CMSG_SPACE(len * sizeof(int)));

    struct cmsghdr *cmptr;
    ssize_t n;
    char dummy_buf[1];

    msg.msg_control = control_un.data();
    msg.msg_controllen = control_un.size();

    iov[0].iov_base = (void *)dummy_buf;
    iov[0].iov_len = 1;

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    CHECK_GE(n = recvmsg(socket_fd, &msg, 0), 0)  << "[mempool] Recv msg fail.";

    CHECK((cmptr = CMSG_FIRSTHDR(&msg)) != nullptr) << "[mempool] Bad cmsg received.";
    CHECK_EQ(cmptr->cmsg_len, CMSG_LEN(sizeof(int) * len))        << "[mempool] Bad cmsg received.";

    memcpy(fd_list, CMSG_DATA(cmptr), sizeof(int) * len);
    CHECK_EQ(cmptr->cmsg_level, SOL_SOCKET) << "[mempool] Bad cmsg received.";
    CHECK_EQ(cmptr->cmsg_type, SCM_RIGHTS) << "[mempool] Bad cmsg received.";

    // close socket
    unlink(slave_name_.c_str());
    close(socket_fd);
}

void HandleTransfer::ExportWorker() {
    LOG(INFO) << "[mempool] Master is now waitting for request vmm handles.";
    bip::scoped_lock request_lock{*request_mutex_};
    bip::scoped_lock ready_lock{*ready_mutex_};
    vmm_export_running_.store(true, std::memory_order_relaxed);    
    while (true) {
        request_cond_->wait(request_lock);
        if (!vmm_export_running_.load(std::memory_order_relaxed)) {
            break;
        }
        LOG(INFO) << "[mempool] Master received request and began to send.";
        std::vector<int> fd_list(TRANSFER_CHUNK_SIZE);
        size_t chunk_base = 0;
        while (chunk_base < phy_mem_list_.size()) {
            size_t chunk_size = std::min(TRANSFER_CHUNK_SIZE, phy_mem_list_.size() - chunk_base);
                for (size_t k = 0; k < chunk_size; ++k) {
                    CU_CALL(cuMemExportToShareableHandle(
                        &fd_list[k], phy_mem_list_[chunk_base + k].cu_handle,
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
                }
            LOG(INFO) << "[mempool] Master is sending handles: " << chunk_base << "/" << phy_mem_list_.size() << ".";
            SendHandles(fd_list.data(), chunk_size, ready_lock);
            for (size_t k = 0; k < chunk_size; ++k) {
                    close(fd_list[k]);
            }
            chunk_base += chunk_size;
        }
    }
    LOG(INFO) << "[mempool] Master exit watting for request.";
}

HandleTransfer::HandleTransfer(SharedMemory &shared_memory,
                const PagesPoolConf &conf,
                std::vector<PhyPage> &ref_phy_pages
) : shared_memory_(shared_memory),
    phy_mem_list_(ref_phy_pages),
    phy_pages_num_(conf.pool_nbytes / conf.page_nbytes), 
    phy_pages_nbytes_(conf.page_nbytes)
{
    CHECK_EQ(phy_mem_list_.size(), 0);
    phy_mem_list_.reserve(phy_pages_num_);
    master_name_ = conf.shm_name + "__sock_ipc_master";
    slave_name_ = conf.shm_name + "__sock_ipc_slave";
}

void HandleTransfer::InitMaster(Belong kFree) {
    InitShm(kFree);
    CUmemAllocationProp prop = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = 0
        }
    };
    CUmemGenericAllocationHandle cu_handle;
    auto start = std::chrono::steady_clock::now();
    for (index_t index = 0; index < phy_pages_num_; ++ index) {
            CU_CALL(cuMemCreate(&cu_handle, phy_pages_nbytes_, &prop, 0));
            phy_mem_list_.push_back(PhyPage{index, cu_handle, &shm_belong_list_[index]});
    }
    auto end = std::chrono::steady_clock::now();
    LOG(INFO) << "[mempool] Alloc " 
        << phy_mem_list_.size() << " x " << ByteDisplay(phy_pages_nbytes_) 
        << " block(s) costs " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms.";
    vmm_export_thread_.reset(new std::thread([&] { ExportWorker(); }));
    while (!vmm_export_running_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void HandleTransfer::InitSlave(Belong kFree) {
    InitShm(kFree);
    std::vector<int> fd_list(TRANSFER_CHUNK_SIZE);
    size_t chunk_base = 0;
    {
        bip::scoped_lock ready_lock(*request_mutex_);
        request_cond_->notify_all();
    }
    auto begin = std::chrono::steady_clock::now();
    while(chunk_base < phy_pages_num_) {
        size_t chunk_size = std::min(TRANSFER_CHUNK_SIZE, phy_pages_num_ - chunk_base);
        LOG(INFO) << "[mempool] Slave is receving handles: " 
                                                        << chunk_base << "/" << phy_pages_num_ << ".";
        ReceiveHandle(fd_list.data(), chunk_size);
        for (size_t k = 0; k < chunk_size; ++k) {
            CUmemGenericAllocationHandle cu_handle;
            CU_CALL(cuMemImportFromShareableHandle(
                            &cu_handle, 
                            reinterpret_cast<void *>(fd_list[k]), 
                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
            close(fd_list[k]);
            index_t index = chunk_base + k;
            phy_mem_list_.push_back(PhyPage{index, cu_handle, &shm_belong_list_[index]});
        }
        chunk_base += chunk_size;
    }
    auto end = std::chrono::steady_clock::now();
    LOG(INFO) << "[mempool] Transfer CUDA IPC handles costs " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms.";
}

void HandleTransfer::ReleaseMaster() {
    vmm_export_running_.store(false, std::memory_order_relaxed);
    {
        boost::interprocess::scoped_lock ipc_lock(*request_mutex_);
        request_cond_->notify_all();
    }
    vmm_export_thread_->join();
}
}