/*
 * CUB stub: DeviceScan for CPU-only builds.
 */
#ifndef CUB_DEVICE_SCAN_STUB_H
#define CUB_DEVICE_SCAN_STUB_H

#include <cstddef>

namespace cub {

struct DeviceScan {
    template <typename InputIt, typename OutputIt>
    static int InclusiveSum(void* temp, size_t& temp_bytes, InputIt in, OutputIt out, int n, void* stream = nullptr) {
        (void)temp; (void)temp_bytes; (void)in; (void)out; (void)n; (void)stream;
        return 0;
    }

    template <typename InputIt, typename OutputIt>
    static int ExclusiveSum(void* temp, size_t& temp_bytes, InputIt in, OutputIt out, int n, void* stream = nullptr) {
        (void)temp; (void)temp_bytes; (void)in; (void)out; (void)n; (void)stream;
        return 0;
    }
};

}  // namespace cub

#endif
