#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <cassert>
#include <chrono>
#include <math.h>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define STB_IMAGE_IMPLEMENTATION 
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define INPUT_FILE "input.bmp"
#define SCALE_FACTOR 2.0f

#define INT_SCALE (1 << 12)

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

double lanczos_kernel(double x, int32_t a) {
    if (x == 0.0f) return 1.0f;
    return a * sin(M_PI * x) * sin(M_PI * x / a) / pow(x, 2) / pow(M_PI, 2);
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

int32_t clamp(int32_t in, int32_t low, int32_t high) {
    if (in < low) return low;
    if (in > high) return high;
    return in;
}

void lanczos(
    uint8_t *in_pixels,
    int32_t in_w,
    int32_t in_h,
    uint8_t *out_pixels,
    int32_t out_w,
    int32_t out_h,
    int32_t a
) {
    if (!(a > 0)) {
        printf("a=%i should be greater than 0\n", a);
        return;
    }

    double ratio_x = (double)in_w / out_w;
    double ratio_y = (double)in_h / out_h;
    
    for (int32_t i = 0; i < out_w; i++) {
        for (int32_t j = 0; j < out_h; j++) {
            int32_t in_x = i * ratio_x;
            int32_t in_y = j * ratio_y;

            // convolve
            int32_t sum = 0;
            int32_t pixel = 0;
            for (int32_t m = -a; m < a; m++) {
                for (int32_t n = -a; n < a; n++) {
                    double x = in_x - i * ratio_x + m;
                    double y = in_y - j * ratio_y + n;
                    
                    int32_t weight = lanczos_kernel(x, a) * lanczos_kernel(y, a) * INT_SCALE;
                    sum += weight;
                                        
                    int32_t real_in_x = clamp(in_x + m, 0, in_w - 1);
                    int32_t real_in_y = clamp(in_y + n, 0, in_h - 1);
                    
                    int32_t idx = real_in_x + real_in_y * in_w;
                    pixel += in_pixels[idx] * weight;
                }
            }

            int32_t idx = i + j * out_w;
            out_pixels[idx] = clamp((uint32_t)(pixel / sum), 0, 255);
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
    uint32_t kernel_size = 5;    

    printf("w: %u, h: %u, tot: %u\n", w, h, in_size);

    // set up the buffer objects
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto in_buf = xrt::bo(device, in_size * sizeof(uint8_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto out_buf = xrt::bo(device, out_size * sizeof(uint8_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    
    // 4 5x5 kernel 32bit aligned
    auto kernel_buf = xrt::bo(device, 128 * sizeof(int16_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

    // Copy instruction stream to xrt buffer object
    void *instr_map = bo_instr.map<void *>();
    memcpy(instr_map, instr_v.data(), instr_v.size() * sizeof(int));

    uint8_t *in_map = in_buf.map<uint8_t *>();
    memcpy(in_map, pixels, in_size * sizeof(uint8_t));    

    // Zero out buffer out_buf
    uint8_t *out_map = out_buf.map<uint8_t *>();
    memset(out_map, 0, out_size * sizeof(uint8_t));

    int16_t *kernel_map = kernel_buf.map<int16_t *>();
    memset(kernel_map, 0, 128 * sizeof(int16_t));
    
    for (uint32_t x = 0; x < 2; x++) {
        for (uint32_t y = 0; y < 2; y++) {
            for (int32_t m = 0; m < kernel_size; m++) {
                for (int32_t n = 0; n < kernel_size; n++) {
                    uint32_t in_x = x / 2;
                    uint32_t in_y = y / 2;

                    double i = in_x - x * 0.5f + (m - 2);
                    double j = in_y - y * 0.5f + (n - 2);
                    
                    double weight = lanczos_kernel(i, 2) * lanczos_kernel(j, 2); 
                    uint32_t idx = (x + y * 2) * 32 \
                        + m * kernel_size \
                        + n;
                    
                    kernel_map[idx] = weight * INT_SCALE;
                }
            } 
        }
    }

    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    kernel_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    unsigned int opcode = 3; // ??

    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(
        opcode,
        bo_instr, instr_v.size(),
        in_buf,
        kernel_buf,
        out_buf
    );
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    
    out_buf.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    uint8_t *ref = (uint8_t *)malloc(o_w * o_h * sizeof(uint8_t));
    lanczos(pixels, w, h, ref, o_w, o_h, 2);

    stbi_write_bmp("ref_out.bmp", o_w, o_h, 1, ref);
    stbi_write_bmp("aie_out.bmp", o_w, o_h, 1, out_map);

    uint64_t errors = 0;
    int32_t err_x = -1;
    int32_t err_y = -1;
    for (size_t y = 0; y < o_h; y++) {
        for (size_t x = 0; x < o_w; x++) {
            if (abs(ref[x + y * o_w] - out_map[x + y * o_w]) > 1) {
                err_x = x;
                err_y = y;
                errors++;
            }

            printf("(x: %4zu, y: %4zu)> %4u: %4u\n", x, y, ref[x + y * o_w], out_map[x + y * o_w]);
        }
    }

    uint32_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    printf("time: %ums\n", ms);
    
    printf("errors: %lu\n", errors);
    printf("last error at x: %i, y: %i\n", err_x, err_y);

    return 0;
}
