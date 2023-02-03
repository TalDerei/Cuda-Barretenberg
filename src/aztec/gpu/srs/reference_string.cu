#include "./reference_string_methods.cu"

namespace gpu_barretenberg {
namespace gpu_waffle {

class FileReferenceString {
    public:
        FileReferenceString(const size_t num_points, std::string const& srs_path)
            : n(num_points)
            , srs(num_points, srs_path)
        {}

        g1::affine_element* get_monomials() { return srs.get_point_table(); }

        size_t get_size() { return n; }
    
    private:
        size_t n;
        gpu_waffle::SRS<g1::affine_element> srs; 
};

} 
}