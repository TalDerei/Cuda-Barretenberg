#include <iostream>
#include "group.cu"
#include <fstream>

using namespace std;

/**
 * Cooley-Tuckey NTT
 */
namespace ntt_common {

/**
 * Nth roots of unity
 */
static var omega_1[LIMBS] = { 0x974bc177a0000006, 0xf13771b2da58a367, 0x51e1a2470908122e, 0x2259d6b14729c0fa };
static var omega_2[LIMBS] = { 0x7f753d979edcef8b, 0x5f3f172cb1479120, 0x8db1627974096c6e, 0x2b377b3539c108cf };
static var omega_3[LIMBS] = { 0xf8ad4ae2b5ad7c3f, 0x2d60c6ff83cb85be, 0xd976fd2b5d9429f7, 0x24407ce73f9ad9a9 };
static var omega_4[LIMBS] = { 0xf4b67d7c742f8f03, 0x681b2ddc63d068cc, 0x8ce5bcef1bfb576a, 0x167c2951d2b63cfe };
static var omega_5[LIMBS] = { 0x2214f7b1bf574c64, 0x28f9232ff7317df2, 0xe57584a8b0ad75cd, 0x2b81fb59dc176d03 };
static var omega_6[LIMBS] = { 0x8e9003e5818f61bf, 0x8c9bbf349bf8fec2, 0x53dceecd3f01534e, 0x2690966be529aa3c };
static var omega_7[LIMBS] = { 0x4e5bb9c416817bb2, 0xa3e6b4c5130cee81, 0x6f1a6d8b83c33f3b, 0x30df01b217649076 };
static var omega_8[LIMBS] = { 0xa2afb83f83a24acc, 0x92f255d9525d536e, 0x5e7566080286dd19, 0x187bb1a6c52d2549 };
static var omega_9[LIMBS] = { 0xe035317b12a423df, 0x8c5e855af698de9e, 0xc60cbafcc21c1862, 0x31549b00936ef0ab };
static var omega_10[LIMBS] = { 0x1ef4b3b4280c1184, 0xcf7ad4c2ae5e2a2c, 0xb8063b6cc5a36518, 0x2348c4b965dfc08c };
static var omega_11[LIMBS] = { 0x48e72189ae4fcfb2, 0xa03fb3c8df85a07, 0xff4d8a35ea9b2e0a, 0x28a98c2ecd9c1d77 };
static var omega_12[LIMBS] = { 0x1dd4522fb3ecdbd3, 0x68222a93d055f3ad, 0xbe9c7d66b3d555e8, 0x1b92f6b86194f846 };
static var omega_13[LIMBS] = { 0x894cdcbe79a04ed6, 0x956cde6a44d30787, 0x59a1b62bd7dbc15f, 0x12ebe4109a806f4e };
static var omega_14[LIMBS] = { 0xba13a0c74d05eab8, 0x2e015d6311ab3116, 0xb503922c8ca5a05a, 0x6be15d7fee394da };
static var omega_15[LIMBS] = { 0x804ef7055bfdb854, 0x7aa76b7140ceeaf2, 0xfcc95a68b2fe89cb, 0x1d461c35f1c406c7 };
static var omega_16[LIMBS] = { 0x47b3e75939397433, 0x6d3a3a920d1c24d1, 0xa134125174f75f43, 0x1b821f01ee6ad556 };
static var omega_17[LIMBS] = { 0x1de7ae6c0204dc7d, 0x7066132e7fe9fdd3, 0xed85d3de061fe189, 0x3474196eb75f75bb };
static var omega_18[LIMBS] = { 0x6948bbc2f9700b85, 0x5b4c22bf1a78f6f1, 0x9189d50fd8d1b14e, 0x35d17d411b036d5e };
static var omega_19[LIMBS] = { 0x49ea3b7e6a27a36, 0xefebe603053edbbc, 0x8424b45a3ace9ed4, 0x287c8390a688795e };
static var omega_20[LIMBS] = { 0xda32d465aa8d931a, 0x2669f68561808f9c, 0x247bab46e4c8b085, 0xd3b668781d6021a };
static var omega_21[LIMBS] = { 0x27ea2192f5322f3c, 0xb11884e9658fe9a7, 0x3a8623bca053c069, 0x128ff3f025e139a6 };
static var omega_22[LIMBS] = { 0x9b8e226ebcef1af2, 0x9e45f1abdf406b60, 0x538dd257d5a7bb3d, 0xa3893037882a3bc };
static var omega_23[LIMBS] = { 0xb22a1998b09e9101, 0x35a97a0fd2b99643, 0x5c526d2fbbe0166e, 0x31c8f536a2daecd2 };
static var omega_24[LIMBS] = { 0xc722bd69d84fd030, 0x600e4a26cf52162c, 0xfb727ed745f3a7e9, 0x1652a7b269fb275c };
static var omega_25[LIMBS] = { 0x400efaff575c07e2, 0x552373494b8f9ac5, 0xaa79abed81e7ad37, 0x1ac6e5b8084d2e39 };
static var omega_26[LIMBS] = { 0xa0a29422c98a20fe, 0x73d462ca65935c9d, 0xe1ba4a6ed44582f7, 0x28fc14c00c3a82b6 };

/**
 * Nth inverse roots of unity
 */
static var omega_inverse_1[LIMBS] = { 0x974bc177a0000006, 0xf13771b2da58a367, 0x51e1a2470908122e, 0x2259d6b14729c0fa };
static var omega_inverse_2[LIMBS] = { 0x84ead9041231077, 0xf128b964422b5002, 0xe2ef28f38ef9444b, 0x359121b088a23783 };
static var omega_inverse_3[LIMBS] = { 0x3207cbdddda29e9b, 0x77ca35a6262fbefc, 0x3d7e07dfb059de38, 0x12a91d5f32bd1157 };
static var omega_inverse_4[LIMBS] = { 0x302fa787a5d2b60e, 0x80cc914fb54036b7, 0x4f65c0b3d235eb3b, 0xadb12085108a15e };
static var omega_inverse_5[LIMBS] = { 0x283ae711001fcfb4, 0x5ead253feb4f32a, 0xcbc338abfa1144ca, 0x300e4c10db78fc8e };
static var omega_inverse_6[LIMBS] = { 0x91c0d6fac5757884, 0xa782859be6bf4559, 0xc3ce14edb987f6cd, 0xbe9c8c56d591f67 };
static var omega_inverse_7[LIMBS] = { 0x8ee513b2d38119f0, 0x87ee89f46bab8280, 0x7fab75cbc4cc2cf4, 0x1bef2752d8704b2c };
static var omega_inverse_8[LIMBS] = { 0x9bf52a88a5a90b53, 0x39ccb835523899e2, 0x6a3cc4193be1b610, 0x134c2780ccac2b38 };
static var omega_inverse_9[LIMBS] = { 0x7e1bb7c18adb9263, 0x82782771c1e4a45, 0xc3b2b36da915deb6, 0xb9db5e6ee822f9a };
static var omega_inverse_10[LIMBS] = { 0x9a42b93a4388ee4f, 0x16c835b156abb91e, 0x5fcd5d77a3b49b70, 0x1d24a254e5e81852 };
static var omega_inverse_11[LIMBS] = { 0xa2be1e2a36fa814f, 0xe25ec7610dd79739, 0xd0725623d6cafca5, 0x68d1538c117d364 };
static var omega_inverse_12[LIMBS] = { 0x422d9ca46c9fc2d9, 0xec33febb4b73a08d, 0x69366b9ee4da54b6, 0x2b22e3d430dc6b9f };
static var omega_inverse_13[LIMBS] = { 0xf2fe10e92ba9466a, 0x77a9e3c167af44ce, 0x723cad05841b37c9, 0x1768636eb0a2cbb1 };
static var omega_inverse_14[LIMBS] = { 0xea9351a4708c94ca, 0xac912ec7a5f17c35, 0x920b769623d9760a, 0x2c3adf7c6cf0bc39 };
static var omega_inverse_15[LIMBS] = { 0x39635a33503045fa, 0x4f4b5b8d30d9f86e, 0xefba9fb03979eba3, 0x2d4d2d2da7cd1545 };
static var omega_inverse_16[LIMBS] = { 0x489362b650573894, 0xa19b79043d35763, 0xda6a5ff6bbc04fa8, 0x2f6a7c2881855e62 };
static var omega_inverse_17[LIMBS] = { 0xcffb35357da73c34, 0x6fc2578de6bb924b, 0x20682d59b41d06df, 0x34842516ccd94688 };
static var omega_inverse_18[LIMBS] = { 0xa83de39403cbc9c4, 0xd04ead4997044238, 0xc85f3ba436d001d2, 0x353809febc022d06 };
static var omega_inverse_19[LIMBS] = { 0x1f296f022886e2a5, 0x550e4e0b1e34eb1c, 0xdf87a2c1b4688775, 0x2707e80c3ba99c65 };
static var omega_inverse_20[LIMBS] = { 0x29a730286ed9e349, 0x1ff90914d9689b7f, 0x24bbb6ac251e133c, 0x22d75746283dfd6 };
static var omega_inverse_21[LIMBS] = { 0x169b4cbac88809a8, 0x457d00f867e4fbe3, 0x6801ddc2b96932b3, 0x48e0c9a914e975b };
static var omega_inverse_22[LIMBS] = { 0xd8bb95832fb90b14, 0xb02515b9a0c38601, 0x5e77f76e182c7fb7, 0x2d000c05d8c1deb4 };
static var omega_inverse_23[LIMBS] = { 0xb095acdf50e3d58, 0x52ef5716a99eae65, 0x9b8665251b9421f0, 0x23c1ca181e965c85 };
static var omega_inverse_24[LIMBS] = { 0x667f0e72d8e96c48, 0x94109e0f4948a4fa, 0xf8504de3afe7d786, 0x2cd33fae85a4f030 };
static var omega_inverse_25[LIMBS] = { 0x5fde22a0a60bd4de, 0xc2222a37273daee6, 0x577b5a278e437749, 0xe8a14abae1d702f };
static var omega_inverse_26[LIMBS] = { 0xf90ad83d70784d69, 0xc98f396490089c33, 0x538bf7a46a63be44, 0x1f2fab5ccf5c3c58 };

static fr_gpu omega(var size) {
    switch (size) {
        case 1: return fr_gpu { omega_1[0], omega_1[1], omega_1[2], omega_1[3] };
        case 2: return fr_gpu { omega_2[0], omega_2[1], omega_2[2], omega_2[3] };
        case 3: return fr_gpu { omega_3[0], omega_3[1], omega_3[2], omega_3[3] };
        case 4: return fr_gpu { omega_4[0], omega_4[1], omega_4[2], omega_4[3] };
        case 5: return fr_gpu { omega_5[0], omega_5[1], omega_5[2], omega_5[3] };
        case 6: return fr_gpu { omega_6[0], omega_6[1], omega_6[2], omega_6[3] };
        case 7: return fr_gpu { omega_7[0], omega_7[1], omega_7[2], omega_7[3] };
        case 8: return fr_gpu { omega_8[0], omega_8[1], omega_8[2], omega_8[3] };
        case 9: return fr_gpu { omega_9[0], omega_9[1], omega_9[2], omega_9[3] };
        case 10: return fr_gpu { omega_10[0], omega_10[1], omega_10[2], omega_10[3] };
        case 11: return fr_gpu { omega_11[0], omega_11[1], omega_11[2], omega_11[3] };
        case 12: return fr_gpu { omega_12[0], omega_12[1], omega_12[2], omega_12[3] };
        case 13: return fr_gpu { omega_13[0], omega_13[1], omega_13[2], omega_13[3] };
        case 14: return fr_gpu { omega_14[0], omega_14[1], omega_14[2], omega_14[3] };
        case 15: return fr_gpu { omega_15[0], omega_15[1], omega_15[2], omega_15[3] };
        case 16: return fr_gpu { omega_16[0], omega_16[1], omega_16[2], omega_16[3] };
        case 17: return fr_gpu { omega_17[0], omega_17[1], omega_17[2], omega_17[3] };
        case 18: return fr_gpu { omega_18[0], omega_18[1], omega_18[2], omega_18[3] };
        case 19: return fr_gpu { omega_19[0], omega_19[1], omega_19[2], omega_19[3] };
        case 20: return fr_gpu { omega_20[0], omega_20[1], omega_20[2], omega_20[3] };
        case 21: return fr_gpu { omega_21[0], omega_21[1], omega_21[2], omega_21[3] };
        case 22: return fr_gpu { omega_22[0], omega_22[1], omega_22[2], omega_22[3] };
        case 23: return fr_gpu { omega_23[0], omega_23[1], omega_23[2], omega_23[3] };
        case 24: return fr_gpu { omega_24[0], omega_24[1], omega_24[2], omega_24[3] };
        case 25: return fr_gpu { omega_25[0], omega_25[1], omega_25[2], omega_25[3] };
        case 26: return fr_gpu { omega_26[0], omega_26[1], omega_26[2], omega_26[3] };
    }
}

static fr_gpu omega_inverse(var size) {
    switch (size) {
        case 1: return fr_gpu { omega_inverse_1[0], omega_inverse_1[1], omega_inverse_1[2], omega_inverse_1[3] };
        case 2: return fr_gpu { omega_inverse_2[0], omega_inverse_2[1], omega_inverse_2[2], omega_inverse_2[3] };
        case 3: return fr_gpu { omega_inverse_3[0], omega_inverse_3[1], omega_inverse_3[2], omega_inverse_3[3] };
        case 4: return fr_gpu { omega_inverse_4[0], omega_inverse_4[1], omega_inverse_4[2], omega_inverse_4[3] };
        case 5: return fr_gpu { omega_inverse_5[0], omega_inverse_5[1], omega_inverse_5[2], omega_inverse_5[3] };
        case 6: return fr_gpu { omega_inverse_6[0], omega_inverse_6[1], omega_inverse_6[2], omega_inverse_6[3] };
        case 7: return fr_gpu { omega_inverse_7[0], omega_inverse_7[1], omega_inverse_7[2], omega_inverse_7[3] };
        case 8: return fr_gpu { omega_inverse_8[0], omega_inverse_8[1], omega_inverse_8[2], omega_inverse_8[3] };
        case 9: return fr_gpu { omega_inverse_9[0], omega_inverse_9[1], omega_inverse_9[2], omega_inverse_9[3] };
        case 10: return fr_gpu { omega_inverse_10[0], omega_inverse_10[1], omega_inverse_10[2], omega_inverse_10[3] };
        case 11: return fr_gpu { omega_inverse_11[0], omega_inverse_11[1], omega_inverse_11[2], omega_inverse_11[3] };
        case 12: return fr_gpu { omega_inverse_12[0], omega_inverse_12[1], omega_inverse_12[2], omega_inverse_12[3] };
        case 13: return fr_gpu { omega_inverse_13[0], omega_inverse_13[1], omega_inverse_13[2], omega_inverse_13[3] };
        case 14: return fr_gpu { omega_inverse_14[0], omega_inverse_14[1], omega_inverse_14[2], omega_inverse_14[3] };
        case 15: return fr_gpu { omega_inverse_15[0], omega_inverse_15[1], omega_inverse_15[2], omega_inverse_15[3] };
        case 16: return fr_gpu { omega_inverse_16[0], omega_inverse_16[1], omega_inverse_16[2], omega_inverse_16[3] };
        case 17: return fr_gpu { omega_inverse_17[0], omega_inverse_17[1], omega_inverse_17[2], omega_inverse_17[3] };
        case 18: return fr_gpu { omega_inverse_18[0], omega_inverse_18[1], omega_inverse_18[2], omega_inverse_18[3] };
        case 19: return fr_gpu { omega_inverse_19[0], omega_inverse_19[1], omega_inverse_19[2], omega_inverse_19[3] };
        case 20: return fr_gpu { omega_inverse_20[0], omega_inverse_20[1], omega_inverse_20[2], omega_inverse_20[3] };
        case 21: return fr_gpu { omega_inverse_21[0], omega_inverse_21[1], omega_inverse_21[2], omega_inverse_21[3] };
        case 22: return fr_gpu { omega_inverse_22[0], omega_inverse_22[1], omega_inverse_22[2], omega_inverse_22[3] };
        case 23: return fr_gpu { omega_inverse_23[0], omega_inverse_23[1], omega_inverse_23[2], omega_inverse_23[3] };
        case 24: return fr_gpu { omega_inverse_24[0], omega_inverse_24[1], omega_inverse_24[2], omega_inverse_24[3] };
        case 25: return fr_gpu { omega_inverse_25[0], omega_inverse_25[1], omega_inverse_25[2], omega_inverse_25[3] };
        case 26: return fr_gpu { omega_inverse_26[0], omega_inverse_26[1], omega_inverse_26[2], omega_inverse_26[3] };
    }
}

}