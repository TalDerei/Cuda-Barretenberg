#include "./io.cu"
using namespace std;

namespace gpu_barretenberg {
namespace gpu_waffle {

template <class T>
SRS<T>::SRS(size_t num_points, std::string const& srs_path) {
    num_points_ = num_points;

    monomials_ = point_table_alloc(num_points);

    read_transcript_g1(monomials_, num_points, srs_path);
}

template <class T>
SRS<T>::~SRS() {
    free(monomials_);
}

template <class T> 
T* SRS<T>::point_table_alloc(size_t num_points) {
    return (T*)aligned_alloc(64, 2 * num_points * sizeof(T));
}

template <class T> 
T* SRS<T>::get_point_table() {
    return monomials_;
}

}
}