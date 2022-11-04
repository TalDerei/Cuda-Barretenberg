#include <cooperative_groups.h>

namespace gpu_barretenberg {
    struct fixnum {
        static constexpr unsigned WIDTH = 4;

        __device__
        static cooperative_groups::thread_block_tile<WIDTH>
        layout() {
            return cooperative_groups::tiled_partition<WIDTH>(
                cooperative_groups::this_thread_block());
        }
    };
}