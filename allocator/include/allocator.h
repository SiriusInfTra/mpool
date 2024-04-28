#pragma once

#include <string>
#include <belong.h>

namespace mpool {

struct AllocatorConfig {
    const std::string log_prefix;
    const Belong belong;
};

}