#include "group.cu"
#include <cstdint>
#include <string>

namespace gpu_barretenberg {
namespace gpu_waffle {

template < typename T >
class SRS {
    public:
        SRS(size_t num_points, std::string const& path);
        
        ~SRS();

        T* point_table_alloc(size_t num_points);

        T* get_point_table();

    private:
        T* monomials_;
        size_t num_points_;
};

}
}