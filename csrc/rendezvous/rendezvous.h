#pragma once

#include "common/logging.h"

#include <zmq.h>
#include <string>

namespace hetu {

class TCPRendezvous
{
private:
  int rank_, nrank_;
  void *socket_ = nullptr;
  void *context_ = nullptr;
  std::string addr_;
  void bind();
  void connect();
public:
  TCPRendezvous(int rank, int nrank, std::string ip, int port);
  void broadcast(void *data, size_t len);
  ~TCPRendezvous();
};

} // namespace hetu
