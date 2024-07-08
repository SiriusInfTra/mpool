#pragma once

#include <atomic>
#include <ostream>

#include <boost/container/list.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <mpool/pages.h>
#include <mpool/shm.h>
#include <mpool/belong.h>

#include <sys/socket.h>
#include <sys/un.h>

namespace mpool {
  enum class Event {
    kMasterChance,
    kClientRequested,
    kClientBound,
    kClientReceived,
    kClientExit,
    kServerReady,
  };

inline std::ostream &operator<<(std::ostream &out, const Event &event) {
  switch (event) {
  case Event::kMasterChance:
    out << "kMasterChance";
    break;
  case Event::kClientRequested:
    out << "kClientRequested";
    break;
  case Event::kClientBound:
    out << "kClientBound";
    break;
  case Event::kClientReceived:
    out << "kClientReceived";
    break;
  case Event::kClientExit:
    out << "kClientExit";
    break;
  case Event::kServerReady:
    out << "kServerReady";
    break;
  }
  return out;
}

inline std::ostream &operator<<(std::ostream &out, const bip_list<Event> &events) {
  bool first = true;
  for (auto &&event : events) {
    out << event;
    if (first) {
      first = false;
    } else {
      out << " ";
    }
  }
  return out;
}
class MessageQueue {
private:
  bip_mutex *mutex;
  bip_cond *cond;
  std::atomic<bool> is_close;
  bip_list<Event> *message_queue_;

public:
  MessageQueue(SharedMemory &shared_memory);

  bool WaitEvent(Event event, bool interruptable = false);

  void RecordEvent(Event event);

  void Close();
};

// NOTE: concurrency construction or deconstruction is NOT allowed.
class CUDAIpcTransfer {
private:
  // typically, there is a limit on the maximum number of transferred FD
  // include/net/scm.h SCM_MAX_FD 253
  static const constexpr size_t TRANSFER_CHUNK_SIZE = 128;

  std::atomic<bool> cuda_ipc_work_ready = true;

  std::string server_sock_name_;
  std::string client_sock_name_;
  MessageQueue message_queue_;
  std::vector<PhyPage> &phy_pages_ref_;
  shm_ptr<BelongImpl> *shm_belong_list_;
  std::thread export_handle_thread_;
  
  const int device_id_;
  const size_t pages_num_;
  const size_t page_nbytes_;
  const std::string log_prefix_;

public:
  CUDAIpcTransfer(SharedMemory &shared_memory,
                    std::vector<PhyPage> &phy_pages_ref,
                    const PagesPoolConf &conf);

  ~CUDAIpcTransfer();

  void InitMaster(Belong kFree);

  void InitMirror();

  void BindServer(int &socket_fd);

  void Send(int fd_list[], size_t len, int socket_fd);

  void UnlinkServer(int socket_fd);

  void BindClient(int &socket_fd);

  void Receive(int fd_list[], size_t len, int socket_fd);

  void UnlinkClient(int socket_fd);

  void Run();
};
} // namespace mpool