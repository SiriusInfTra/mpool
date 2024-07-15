#include <mpool/pages.h>
#include <mpool/cuda_handle.h>
#include <glog/logging.h>

namespace mpool {
void CUDAIpcTransfer::BindServer(int &socket_fd) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 2) << log_prefix_ << "BIND SERVER.";
  struct sockaddr_un server_addr;

  // create socket and bind
  CHECK_GE(socket_fd = socket(AF_UNIX, SOCK_DGRAM, 0), 0)
      << log_prefix_ << "Create socket failed.";

  bzero(&server_addr, sizeof(server_addr));
  server_addr.sun_family = AF_UNIX;

  unlink(("/dev/shm/" + server_sock_name_).c_str());
  // strncpy(server_addr.sun_path, server_sock_name_.c_str(),
  //         server_sock_name_.size());
  sprintf(server_addr.sun_path, "/dev/shm/%s", server_sock_name_.c_str());

  CHECK_EQ(
      bind(socket_fd, (struct sockaddr *)&server_addr, SUN_LEN(&server_addr)),
      0)
      << log_prefix_ << "Bind error.";
}
void CUDAIpcTransfer::UnlinkServer(int socket_fd) {
  unlink(("/dev/shm/" + server_sock_name_).c_str());
  close(socket_fd);
}
void CUDAIpcTransfer::BindClient(int &socket_fd) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 2) << log_prefix_ << "BIND CLIENT.";
  struct sockaddr_un client_addr;
  CHECK_GE(socket_fd = socket(AF_UNIX, SOCK_DGRAM, 0), 0)
      << log_prefix_ << "Create socket fail.";

  bzero(&client_addr, sizeof(client_addr));
  client_addr.sun_family = AF_UNIX;

  unlink(("/dev/shm/" + client_sock_name_).c_str());
  // strncpy(client_addr.sun_path, client_sock_name_.c_str(),
  //         client_sock_name_.size());
  sprintf(client_addr.sun_path, "/dev/shm/%s", client_sock_name_.c_str());
  CHECK_EQ(
      bind(socket_fd, (struct sockaddr *)&client_addr, SUN_LEN(&client_addr)),
      0)
      << log_prefix_ << "Bind fail.";
}
void CUDAIpcTransfer::UnlinkClient(int socket_fd) {
  unlink(("/dev/shm" + client_sock_name_).c_str());
  close(socket_fd);
}
void CUDAIpcTransfer::Receive(int fd_list[], size_t len, int socket_fd) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1) << log_prefix_ <<  "RECEIVE.";
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

  CHECK_GE(n = recvmsg(socket_fd, &msg, 0), 0) << log_prefix_ << "Recv msg fail.";

  CHECK((cmptr = CMSG_FIRSTHDR(&msg)) != nullptr)
      << log_prefix_ << "Bad cmsg received.";
  CHECK_EQ(cmptr->cmsg_len, CMSG_LEN(sizeof(int) * len))
      << log_prefix_ << "cmsg received.";

  memcpy(fd_list, CMSG_DATA(cmptr), sizeof(int) * len);
  CHECK_EQ(cmptr->cmsg_level, SOL_SOCKET) << log_prefix_ << "Bad cmsg received.";
  CHECK_EQ(cmptr->cmsg_type, SCM_RIGHTS) << log_prefix_ << "Bad cmsg received.";
}
void CUDAIpcTransfer::Send(int fd_list[], size_t len, int socket_fd) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 2) << log_prefix_ << "SEND";
  struct msghdr msg;
  struct iovec iov[1];

  std::vector<std::byte> control_un(CMSG_SPACE(len * sizeof(int)));

  struct cmsghdr *cmptr;
  struct sockaddr_un client_addr;
  bzero(&client_addr, sizeof(client_addr));
  client_addr.sun_family = AF_UNIX;
  // strncpy(client_addr.sun_path, client_sock_name_.c_str(),
  //         client_sock_name_.size());
  sprintf(client_addr.sun_path, "/dev/shm/%s", client_sock_name_.c_str());

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
  CHECK_GE(send_result, 0) << log_prefix_ << "Send msg fail.";
}
CUDAIpcTransfer::CUDAIpcTransfer(SharedMemory &shared_memory,
                                 std::vector<PhyPage> &phy_pages_ref,
                                 const PagesPoolConf &conf)
    : server_sock_name_(conf.shm_name + "__sock_ipc_server.sock"),
      client_sock_name_(conf.shm_name + "__sock_ipc_client.sock"),
      message_queue_(shared_memory), phy_pages_ref_(phy_pages_ref),
      device_id_(conf.device_id),
      pages_num_(conf.pool_nbytes / conf.page_nbytes),
      page_nbytes_(conf.page_nbytes) {
  shm_belong_list_ =
      shared_memory->find_or_construct<shm_ptr<BelongImpl>>("HT_belong_list")[pages_num_]();
  phy_pages_ref_.reserve(pages_num_);
}

CUDAIpcTransfer::~CUDAIpcTransfer() {
  message_queue_.Close();
  export_handle_thread_.join();
}

void CUDAIpcTransfer::InitMaster(Belong kFree) {
  message_queue_.RecordEvent(Event::kMasterChance);
  for (size_t k = 0; k < pages_num_; ++k) {
    shm_belong_list_[k] = kFree;
  }
  CUmemAllocationProp prop = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = device_id_}};
  DLOG(INFO) << log_prefix_ << "Allocating " << pages_num_ << " x "
             << ByteDisplay(page_nbytes_) << " block(s) on device " << device_id_;
  CUmemGenericAllocationHandle cu_handle;
  auto start = std::chrono::steady_clock::now();
  for (index_t index = 0; index < pages_num_; ++index) {
    CU_CALL(cuMemCreate(&cu_handle, page_nbytes_, &prop, 0));
    phy_pages_ref_.push_back(
        PhyPage{index, cu_handle, &shm_belong_list_[index]});
  }
  auto end = std::chrono::steady_clock::now();
  LOG_IF(INFO, VERBOSE_LEVEL >= 1) << log_prefix_ 
      << "[CUDAIpcTransfer] Alloc " << pages_num_ << " x "
      << ByteDisplay(page_nbytes_) << " block(s) costs "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
      << " ms.";
  export_handle_thread_ = std::thread{[&] { Run(); }};
  while (!cuda_ipc_work_ready) {}
  LOG_IF(INFO, VERBOSE_LEVEL >= 1) << log_prefix_ << "[CUDAIpcTransfer] Init master ok!";
}

void CUDAIpcTransfer::InitMirror() {
  message_queue_.WaitEvent(Event::kServerReady);
  int socket_fd;
  BindClient(socket_fd);
  message_queue_.RecordEvent(Event::kClientBound);
  for (size_t chunk_begin = 0; chunk_begin < pages_num_;
       chunk_begin += TRANSFER_CHUNK_SIZE) {
    size_t chunk_size =
        std::min(pages_num_ - chunk_begin, TRANSFER_CHUNK_SIZE);
    std::vector<int> fd_list(chunk_size);
    LOG_IF(INFO, VERBOSE_LEVEL >= 2) << log_prefix_ 
        << "[CUDAIpcTransfer] Mirror is receving handles: " << chunk_begin << "/"
        << pages_num_ << ".";
    message_queue_.WaitEvent(Event::kClientReceived);
    Receive(fd_list.data(), fd_list.size(), socket_fd);
    for (size_t k = 0; k < chunk_size; ++k) {
      CUmemGenericAllocationHandle cu_handle;
      CU_CALL(cuMemImportFromShareableHandle(
          &cu_handle, reinterpret_cast<void *>(fd_list[k]),
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      close(fd_list[k]);
      index_t index = chunk_begin + k;
      phy_pages_ref_.push_back(
          PhyPage{index, cu_handle, &shm_belong_list_[index]});
    }

  }
  UnlinkClient(socket_fd);
  message_queue_.RecordEvent(Event::kClientExit);
  export_handle_thread_ = std::thread{[&] { Run(); }};
  while (!cuda_ipc_work_ready) {}
  LOG_IF(INFO, VERBOSE_LEVEL >= 1) << log_prefix_ << "[CUDAIpcTransfer] Init mirror ok!";
}

void CUDAIpcTransfer::Run() {
  if (message_queue_.WaitEvent(Event::kMasterChance, true) ==
      false) {
    return;
  }
  LOG_IF(INFO, VERBOSE_LEVEL >= 1) << log_prefix_ << "[CUDAIpcTransfer] Worker become master!";
  int socket_fd;
  BindServer(socket_fd);
  while (true) {
    message_queue_.RecordEvent(Event::kServerReady);
    if (message_queue_.WaitEvent(Event::kClientBound, true) ==
        false) {
      LOG_IF(INFO, VERBOSE_LEVEL >= 2) << log_prefix_ << "[CUDAIpcTransfer] Worker exit due to close.";
      break;
    }
    for (size_t chunk_begin = 0; chunk_begin < pages_num_;
         chunk_begin += TRANSFER_CHUNK_SIZE) {
      size_t chunk_size =
          std::min(pages_num_ - chunk_begin, TRANSFER_CHUNK_SIZE);
      std::vector<int> fd_list(chunk_size);
      for (size_t k = 0; k < chunk_size; ++k) {
        CU_CALL(cuMemExportToShareableHandle(
            &fd_list[k], phy_pages_ref_[chunk_begin + k].cu_handle,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
      }
      LOG_IF(INFO, VERBOSE_LEVEL >= 2) << log_prefix_ << "[CUDAIpcTransfer] Master is sending handles: " << chunk_begin << "/"
                << pages_num_ << ".";

      Send(fd_list.data(), fd_list.size(), socket_fd);
      message_queue_.RecordEvent(Event::kClientReceived);
      for (size_t k = 0; k < chunk_size; ++k) {
        close(fd_list[k]);
      }
      // message_queue_.WaitEvent(Event::kClientReceived);
    }
    message_queue_.WaitEvent(Event::kClientExit);
  }
  UnlinkServer(socket_fd);
  message_queue_.RecordEvent(Event::kMasterChance);
  LOG(INFO) << log_prefix_ << "EXIT!";
}

MessageQueue::MessageQueue(SharedMemory &shared_memory) : is_close(false) {
  mutex = shared_memory->find_or_construct<bip_mutex>("MQ_mutex")();
  cond = shared_memory->find_or_construct<bip_cond>("MQ_cond")();
  message_queue_ = shared_memory->find_or_construct<bip_list<Event>>("MQ_mq")(
      shared_memory->get_segment_manager());
}
bool MessageQueue::WaitEvent(Event event, bool interruptable) {
  bip::scoped_lock lock{*mutex};
  LOG_IF(INFO, VERBOSE_LEVEL >= 3) << "WaitEvent " << event << ".";
  LOG_IF(INFO, VERBOSE_LEVEL >= 3) << "CURRENT LIST: " << *message_queue_; 
  cond->wait(lock, [&] {
    return (interruptable && is_close) ||
           (!message_queue_->empty() && message_queue_->front() == event);
  });
  LOG_IF(INFO, VERBOSE_LEVEL >= 3) << "WaitEvent OK " << event << "." ;
  if (message_queue_->front() == event) {
    message_queue_->pop_front();
    cond->notify_all();
    LOG_IF(INFO, VERBOSE_LEVEL >= 3) << "CURRENT LIST: " << *message_queue_;
    return true;
  } 
  LOG_IF(INFO, VERBOSE_LEVEL >= 3) << "CURRENT LIST: " << *message_queue_;
  return false;

}
void MessageQueue::RecordEvent(Event event) {
  bip::scoped_lock lock{*mutex};
  LOG_IF(INFO, VERBOSE_LEVEL >= 3) << "RecordEvent " << event << "."; 
  message_queue_->push_back(event);
  cond->notify_all();
  LOG_IF(INFO, VERBOSE_LEVEL >= 3) << "CURRENT LIST: " << *message_queue_;
}
void MessageQueue::Close() {
  is_close = true;
  cond->notify_all();
}

} // namespace mpool