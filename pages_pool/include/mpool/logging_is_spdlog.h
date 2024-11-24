#pragma once

#include "spdlog/common.h"
#include <ostream>
#include <spdlog/spdlog.h>
#include <sstream>
enum LogSeverity {
    OFF = spdlog::level::off,
    INFO = spdlog::level::info,
    WARNING = spdlog::level::warn,
    ERROR = spdlog::level::err,
    FATAL = spdlog::level::critical
};

class LogMessage: public std::stringstream {
 private:
  const spdlog::level::level_enum level_;
  const spdlog::source_loc source_loc_;
//   std::stringstream ss_;

 public:
  LogMessage(spdlog::level::level_enum severity, spdlog::source_loc source_loc) : level_(severity), source_loc_(source_loc) {}

//   std::ostream &stream() { return ss_; }

  ~LogMessage() {
    if (level_ == spdlog::level::off) {
      return;
    }
    *this << "\n";
    spdlog::default_logger_raw()->log(source_loc_, level_, this->str());
  }
};


#define SEVERITY_OFF spdlog::level::off
#define SEVERITY_INFO spdlog::level::info   
#define SEVERITY_WARNING spdlog::level::warn
#define SEVERITY_ERROR spdlog::level::err
#define SEVERITY_FATAL spdlog::level::critical

#define LOG(severity) LogMessage(SEVERITY_##severity, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#define LOG_IF(severity, condition) \
  LogMessage((condition) ? SEVERITY_##severity : SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#define CHECK(condition) \
 LogMessage((condition) ? SEVERITY_OFF : SEVERITY_FATAL, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}) << "Check failed: " #condition " "
#define CHECK_EQ(val1, val2) CHECK((val1) == (val2))
#define CHECK_NE(val1, val2) CHECK((val1) != (val2))
#define CHECK_LE(val1, val2) CHECK((val1) <= (val2))
#define CHECK_LT(val1, val2) CHECK((val1) < (val2))
#define CHECK_GE(val1, val2) CHECK((val1) >= (val2))
#define CHECK_GT(val1, val2) CHECK((val1) > (val2))

#ifdef _DEBUG

#define DLOG LOG
#define DLOG_IF LOG_IF

#define DCHECK CHECK
#define DCHECK_EQ CHECK_EQ
#define DCHECK_NE CHECK_NE
#define DCHECK_LE CHECK_LE
#define DCHECK_LT CHECK_LT
#define DCHECK_GE CHECK_GE
#define DCHECK_GT CHECK_GT

#else

#define DLOG(severity) LogMessage(SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#define DLOG_IF(severity, condition) LogMessage(SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})

#define DCHECK(condition) LogMessage(SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#define DCHECK_EQ(val1, val2) LogMessage(SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#define DCHECK_NE(val1, val2) LogMessage(SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#define DCHECK_LE(val1, val2) LogMessage(SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#define DCHECK_LT(val1, val2) LogMessage(SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#define DCHECK_GE(val1, val2) LogMessage(SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#define DCHECK_GT(val1, val2) LogMessage(SEVERITY_OFF, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})
#endif

