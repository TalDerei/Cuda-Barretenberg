#include "fq.cuh"
#include <ecc/curves/grumpkin/grumpkin.hpp>
#include <gtest/gtest.h>

using namespace barretenberg;
using namespace gpu_barretenberg;
using namespace std;


TEST(fr, mul)
{
    // field_gpu a{ 0x192f9ddc938ea63, 0x1db93d61007ec4fe, 0xc89284ec31fa49c0, 0x2478d0ff12b04f0f };
    // field_gpu b{ 0x7aade4892631231c, 0x8e7515681fe70144, 0x98edb76e689b6fd8, 0x5d0886b15fc835fa };
    // field_gpu expected{ 0xab961ef46b4756b6, 0xbc6b636fc29678c8, 0xd247391ed6b5bd16, 0x12e8538b3bde6784 };
    // field_gpu result;
    // result = a * b;
    // EXPECT_EQ((result == expected), true);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}