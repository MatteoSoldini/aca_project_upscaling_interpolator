#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <cassert>
#include <chrono>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define STB_IMAGE_IMPLEMENTATION 
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define INPUT_FILE "input.bmp"
#define SCALE_FACTOR 2.0f

std::vector<uint32_t> load_file(std::string file_path) {
    // Open file in binary mode
    std::ifstream file_stream(file_path, std::ios::binary);
    if (!file_stream.is_open()) {
        throw std::runtime_error("Unable to open file\n");
    }

    // Get the size of the file
    file_stream.seekg(0, std::ios::end);
    std::streamsize size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);

    // Check that the file size is a multiple of 4 bytes (size of uint32_t)
    if (size % 4 != 0) {
        throw std::runtime_error("File size is not a multiple of 4 bytes\n");
    }

    // Allocate vector and read the binary data
    std::vector<uint32_t> content(size / 4);
    if (!file_stream.read(reinterpret_cast<char *>(content.data()), size)) {
        throw std::runtime_error("Failed to read instruction file\n");
    }
    return content;
}

void neareast_neightbor(
    uint8_t *in_pixels, uint32_t in_w, uint32_t in_h,
    uint8_t *out_pixels, uint32_t out_w, uint32_t out_h
) {
    double ratio_x = (double)out_w / in_w;
    double ratio_y = (double)out_h / in_h;

    for (uint32_t i = 0; i < out_w; i++) {
        for (uint32_t j = 0; j < out_h; j++) {
            uint32_t in_x = i / ratio_x;
            uint32_t in_y = j / ratio_y;
            
            out_pixels[i + j * out_w] = in_pixels[in_x + in_y * in_w];
        }
    }
}

int main(void) {
    // Initialize device
    xrt::device device = xrt::device(0);

    auto xclbin = xrt::xclbin("build/final.xclbin");
    device.register_xclbin(xclbin);

    xrt::hw_context context(device, xclbin.get_uuid());
    xrt::kernel kernel = xrt::kernel(context, "MLIR_AIE");

    std::vector<uint32_t> instr_v = load_file("build/insts.bin");
    
    // Load image
    int32_t w, h, c;
    uint8_t *pixels = stbi_load(INPUT_FILE, &w, &h, &c, 1);
    assert(pixels != NULL && "failed to load the image");    

    uint32_t in_size = w * h;
    uint32_t o_w = w * SCALE_FACTOR;
    uint32_t o_h = h * SCALE_FACTOR;
    uint32_t out_size = o_w * o_h;

    printf("w: %u, h: %u, tot: %u\n", w, h, in_size);

    // set up the buffer objects
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto in_buf = xrt::bo(device, in_size * sizeof(uint8_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    //auto bo_inFactor = xrt::bo(device, 1 * sizeof(int32_t),
    //                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto out_buf = xrt::bo(device, out_size * sizeof(uint8_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

    // Copy instruction stream to xrt buffer object
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    uint8_t *bufInA = in_buf.map<uint8_t *>();
    memcpy(bufInA, pixels, in_size * sizeof(uint8_t));    

    // Initialize buffer bo_inFactor
    //uint8_t *bufInFactor = bo_inFactor.map<uint8_t *>();
    //*bufInFactor = 2; // TEMP

    // Zero out buffer out_buf
    uint8_t *bufOut = out_buf.map<uint8_t *>();
    memset(bufOut, 0, out_size * sizeof(uint8_t));

    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    //bo_inFactor.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    unsigned int opcode = 3; // ??

    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(opcode, bo_instr, instr_v.size(), in_buf, out_buf);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    
    out_buf.sync(XCL_BO_SYNC_BO_FROM_DEVICE);    
    
    uint8_t *ref = (uint8_t *)malloc(o_w * o_h * sizeof(uint8_t));
    neareast_neightbor(pixels, w, h, ref, o_w, o_h);
    
    stbi_write_bmp("ref_out.bmp", o_w, o_h, 1, ref);
    stbi_write_bmp("aie_out.bmp", o_w, o_h, 1, bufOut);

    uint64_t errors = 0;
    int32_t err_x = -1;
    int32_t err_y = -1;
    for (size_t y = 0; y < o_h; y++) {
        for (size_t x = 0; x < o_w; x++) {
            if (ref[x + y * o_w] != bufOut[x + y * o_w]) {
                err_x = x;
                err_y = y;
                errors++;
            }

            printf("(x: %4zu, y: %4zu)> %4u: %4u\n", x, y, ref[x + y * o_w], bufOut[x + y * o_w]);
        }
    }

    uint32_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    printf("time: %ums\n", ms);
    
    printf("errors: %lu\n", errors);
    printf("last error at x: %i, y: %i\n", err_x, err_y);

    return 0;
}
