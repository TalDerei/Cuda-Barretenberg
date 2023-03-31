#include "./reference_string_methods.cuh"
#include <common/net.hpp>
#include <common/throw_or_abort.hpp>
#include <fstream>
#include <sys/stat.h>

namespace gpu_barretenberg {
namespace gpu_waffle {

struct Manifest {
    uint32_t transcript_number;
    uint32_t total_transcripts;
    uint32_t total_g1_points;
    uint32_t total_g2_points;
    uint32_t num_g1_points;
    uint32_t num_g2_points;
    uint32_t start_from;
};

/**
 * Manifest structure of the SRS 
*/
void read_manifest(std::string const& filename, Manifest& manifest) {
    std::ifstream file;
    file.open(filename, std::ifstream::binary);
    file.read((char*)&manifest, sizeof(Manifest));
    file.close();

    manifest.transcript_number = ntohl(manifest.transcript_number);
    manifest.total_transcripts = ntohl(manifest.total_transcripts);
    manifest.total_g1_points = ntohl(manifest.total_g1_points);
    manifest.total_g2_points = ntohl(manifest.total_g2_points);
    manifest.num_g1_points = ntohl(manifest.num_g1_points);
    manifest.num_g2_points = ntohl(manifest.num_g2_points);
    manifest.start_from = ntohl(manifest.start_from);
}

void byteswap(g1::affine_element* elements, size_t elements_size) {
    constexpr size_t bytes_per_element = sizeof(g1::affine_element);
    size_t num_elements = elements_size / bytes_per_element;

    if (is_little_endian()) {
        for (size_t i = 0; i < num_elements; ++i) {
            elements[i].x.data[0] = __builtin_bswap64(elements[i].x.data[0]);
            elements[i].x.data[1] = __builtin_bswap64(elements[i].x.data[1]);
            elements[i].x.data[2] = __builtin_bswap64(elements[i].x.data[2]);
            elements[i].x.data[3] = __builtin_bswap64(elements[i].x.data[3]);
            elements[i].y.data[0] = __builtin_bswap64(elements[i].y.data[0]);
            elements[i].y.data[1] = __builtin_bswap64(elements[i].y.data[1]);
            elements[i].y.data[2] = __builtin_bswap64(elements[i].y.data[2]);
            elements[i].y.data[3] = __builtin_bswap64(elements[i].y.data[3]);
        }
    }
}

size_t get_file_size(std::string const& filename) {
    struct stat st;
    if (stat(filename.c_str(), &st) != 0) {
        return 0;
    }
    return (size_t)st.st_size;
}

void read_file_into_buffer(char* buffer, size_t& size, std::string const& filename, size_t offset = 0, size_t amount = 0) {
    size = amount ? amount : get_file_size(filename);

    std::ifstream file;
    file.open(filename, std::ifstream::binary);
    file.seekg((int)offset);

    file.read(buffer, (int)size);
    if (!file) {
        ptrdiff_t read = file.gcount();
        throw_or_abort(format("Only read ", read, " bytes from file but expected ", size, "."));
    }

    file.close();
}

std::string get_transcript_path(std::string const& dir, size_t num) {
    return dir + "/transcript" + (num < 10 ? "0" : "") + std::to_string(num) + ".dat";
};

bool is_file_exist(std::string const& fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

void read_transcript_g1(g1::affine_element* monomials, size_t degree, std::string const& dir) {
    // g1 basic generator
    monomials[0] = {
        {0xd35d438dc58f0d9dUL, 0xa78eb28f5c70b3dUL, 0x666ea36f7879462cUL, 0xe0a77c19a07df2fUL} , 
        {0xa6ba871b8b1e1b3aUL, 0x14f1d651eb8e167bUL, 0xccdd46def0f28c58UL, 0x1c14ef83340fbe5eUL }
    };

    size_t num = 0;
    size_t num_read = 1;
    std::string path = get_transcript_path(dir, num);

    while (is_file_exist(path) && num_read < degree) {
        Manifest manifest;
        read_manifest(path, manifest);

        auto offset = sizeof(Manifest);
        const size_t num_to_read = std::min((size_t)manifest.num_g1_points, degree - num_read);
        const size_t g1_buffer_size = sizeof(g1::affine_element) * 2 * num_to_read;

        char* buffer = (char*)&monomials[num_read];
        size_t size = 0;

        read_file_into_buffer(buffer, size, path, offset, g1_buffer_size);
        byteswap(&monomials[num_read], size);

        num_read += num_to_read;
        path = get_transcript_path(dir, ++num);
    }

    if (num_read < degree) {
        throw_or_abort(format("Only read ", num_read, " points but require ", degree, ". Is your srs large enough?"));
    }
}

}
} 
