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

#include <opencv2/opencv.hpp>

#define INPUT_FILE "input.jpg"
#define SCALE_FACTOR 2.0f   // must be an integer value

#define INT_SCALE (1 << 12)
#define A 2

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

int32_t clamp(int32_t in, int32_t low, int32_t high) {
    if (in < low) return low;
    if (in > high) return high;
    return in;
}

double lanczos_kernel(double x, int32_t a) {
    if (x == 0.0f) return 1.0f;
    return a * sin(M_PI * x) * sin(M_PI * x / a) / pow(x, 2) / pow(M_PI, 2);
}

uint8_t *lanczos(
    uint8_t *in,
    int32_t in_w,
    int32_t in_h,
    double scale_factor,
    int32_t a
) {
    int32_t out_w = in_w * scale_factor;
    int32_t out_h = in_h * scale_factor;
    uint8_t *out = (uint8_t *)malloc(out_w * out_h * sizeof(uint8_t));
    
    if (!(a > 0)) {
        printf("a=%i should be greater than 0\n", a);
        return NULL;
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
            for (int32_t m = -a+1; m <= a; m++) {
                for (int32_t n = -a+1; n <= a; n++) {
                    double x = in_x - i * ratio_x + m;
                    double y = in_y - j * ratio_y + n;
                    
                    int32_t weight = lanczos_kernel(x, a) * lanczos_kernel(y, a) * INT_SCALE;
                    sum += weight;
                                        
                    int32_t real_in_x = clamp(in_x + m, 0, in_w - 1);
                    int32_t real_in_y = clamp(in_y + n, 0, in_h - 1);
                    
                    int32_t idx = real_in_x + real_in_y * in_w;
                    pixel += in[idx] * weight;
                }
            }

            int32_t idx = i + j * out_w;
            out[idx] = clamp((uint32_t)(pixel / sum), 0, 255);
        }
    }

    return out;
}

uint8_t *lanczos_opencv(uint8_t *in, int32_t in_w, int32_t in_h, double scale_factor) {
    int32_t out_w = in_w * scale_factor;
    int32_t out_h = in_h * scale_factor;
    uint8_t *out = (uint8_t *)malloc(out_w * out_h * sizeof(uint8_t));

    cv::Mat cv_in(in_h, in_w, CV_8UC1, in);
    cv::Mat cv_out;
    cv::resize(cv_in, cv_out, cv::Size(), scale_factor, scale_factor, cv::INTER_LANCZOS4);
    
    memcpy(out, cv_out.data, out_w * out_h);
    return out;
}

void build_aie(int32_t in_w, int32_t in_h, double scale_factor, bool scalar) {
    char command[1024];
    sprintf(command, 
        "cd build && ${PEANO_INSTALL_DIR}/bin/clang++ \
            -std=c++20 \
            -O2 \
            --target=aie2-none-unknown-elf \
            -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
            -DNDEBUG \
            %s \
            -I$VIRTUAL_ENV/lib/python3.12/site-packages/mlir_aie/include \
            -c \
            -o kernel.o \
            ../kernel.cpp",
        scalar ? "-DSCALAR" : ""
    );
    system(command);

    sprintf(command, "cd build && python ../aie2.py %i %i %f > aie.mlir", in_w, in_h, scale_factor);
    system(command);
    
    system(
        "cd build && aiecc.py \
            --aie-generate-xclbin \
            --no-compile-host \
            --xclbin-name=final.xclbin \
            --no-xchesscc \
            --no-xbridge \
            --aie-generate-npu-insts \
            --npu-insts-name=insts.bin \
            aie.mlir"
    );
}

uint8_t *lanczos_aie(uint8_t *in, int32_t in_w, int32_t in_h, double scale_factor) {
    // Initialize device
    xrt::device device = xrt::device(0);
    
    printf("device: %s\n", device.get_info<xrt::info::device::name>().c_str());

    auto xclbin = xrt::xclbin("build/final.xclbin");
    device.register_xclbin(xclbin);

    xrt::hw_context context(device, xclbin.get_uuid());
    xrt::kernel kernel = xrt::kernel(context, "MLIR_AIE");

    std::vector<uint32_t> instr_v = load_file("build/insts.bin");
    
    // set up the buffer objects
    int32_t in_size = in_w * in_h;
    int32_t out_w = in_w * scale_factor;
    int32_t out_h = in_h * scale_factor;
    uint8_t *out = (uint8_t *)malloc(out_w * out_h * sizeof(uint8_t));
    int32_t out_size = out_w * out_h; 

    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto in_buf = xrt::bo(device, in_size * sizeof(uint8_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto out_buf = xrt::bo(device, out_size * sizeof(uint8_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    
    // 4x4 convolution matrix
    int32_t n_c_mtx = scale_factor;
    int32_t c_mtx_size = 16 * n_c_mtx * n_c_mtx;
    auto c_mtx_buf = xrt::bo(device, c_mtx_size * sizeof(int16_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    // Copy instruction stream to xrt buffer object
    void *instr_map = bo_instr.map<void *>();
    memcpy(instr_map, instr_v.data(), instr_v.size() * sizeof(int));

    uint8_t *in_map = in_buf.map<uint8_t *>();
    memcpy(in_map, in, in_size * sizeof(uint8_t));    

    // Zero out buffer out_buf
    uint8_t *out_map = out_buf.map<uint8_t *>();
    memset(out_map, 0, out_size * sizeof(uint8_t));

    int16_t *c_mtx_map = c_mtx_buf.map<int16_t *>();
    memset(c_mtx_map, 0, c_mtx_size * sizeof(int16_t));
    
    for (uint32_t x = 0; x < n_c_mtx; x++) {
        for (uint32_t y = 0; y < n_c_mtx; y++) {
            for (int32_t m = -A+1; m <= A; m++) {
                for (int32_t n = -A+1; n <= A; n++) {
                    uint32_t in_x = x / scale_factor;
                    uint32_t in_y = y / scale_factor;

                    double i = in_x - x * (1 / scale_factor) + m;
                    double j = in_y - y * (1 / scale_factor) + n;
                    
                    double weight = lanczos_kernel(i, A) * lanczos_kernel(j, A); 
                    uint32_t idx = (x + y * n_c_mtx) * 16 \
                        + (n+1) * 2*A \
                        + (m+1);
                    
                    //printf("x=%lf, x=%lf> w=%lf\n", i, j, weight);                    
                    
                    c_mtx_map[idx] = weight * INT_SCALE;
                }
            } 
        }
    }

    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c_mtx_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    unsigned int opcode = 3; // ??

    auto run = kernel(
        opcode,
        bo_instr, instr_v.size(),
        in_buf,
        out_buf,
        c_mtx_buf
    );
    run.wait();
    
    out_buf.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    memcpy(out, out_map, out_size);
    return out;
}

int main(void) {
    // Load image
    int32_t w, h, c;
    uint8_t *pixels = stbi_load(INPUT_FILE, &w, &h, &c, 1);
    assert(pixels != NULL && "failed to load the image");    
    
    uint32_t in_size = w * h;
    uint32_t o_w = w * SCALE_FACTOR;
    uint32_t o_h = h * SCALE_FACTOR;
    uint32_t out_size = o_w * o_h;

    build_aie(w, h, SCALE_FACTOR, true);
    
    printf("w: %5i, h: %5i => o_w: %5i, o_h: %5i\n", w, h, o_w, o_h);

    // CPU
    auto start = std::chrono::high_resolution_clock::now();
    uint8_t *c_out = lanczos(pixels, w, h, SCALE_FACTOR, 2);
    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    printf("cpu time: %.0lf ms\n", ms);

    // OpenCV
    start = std::chrono::high_resolution_clock::now();
    uint8_t *cv_out = lanczos_opencv(pixels, w, h, SCALE_FACTOR); 
    stop = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    printf("opencv time: %.0lf ms\n", ms);

    build_aie(w, h, SCALE_FACTOR, true);
    
    // AIE Scalar
    start = std::chrono::high_resolution_clock::now();
    uint8_t *aie_sca_out = lanczos_aie(pixels, w, h, SCALE_FACTOR);    
    stop = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    printf("aie scalar time: %.0lf ms\n", ms);

    build_aie(w, h, SCALE_FACTOR, false);
    
    // AIE Vector
    start = std::chrono::high_resolution_clock::now();
    uint8_t *aie_vec_out = lanczos_aie(pixels, w, h, SCALE_FACTOR);    
    stop = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    printf("aie vector time: %.0lf ms\n", ms);

    //neareast_neightbor(pixels, w, h, ref, o_w, o_h);
    stbi_write_bmp("c_out.bmp", o_w, o_h, 1, c_out);
    stbi_write_bmp("aie_sca_out.bmp", o_w, o_h, 1, aie_sca_out);
    stbi_write_bmp("aie_vec_out.bmp", o_w, o_h, 1, aie_vec_out);
    stbi_write_bmp("cv_out.bmp", o_w, o_h, 1, cv_out);

    uint64_t errors = 0;
    int32_t err_x = -1;
    int32_t err_y = -1;
    for (size_t y = 0; y < o_h; y++) {
        for (size_t x = 0; x < o_w; x++) {
            if (c_out[x + y * o_w] != aie_vec_out[x + y * o_w]) {
                err_x = x;
                err_y = y;
                errors++;
            }

            //printf("(x: %4zu, y: %4zu)> %4u: %4u\n", x, y, c_out[x + y * o_w], aie_out[x + y * o_w]);
        }
    }
    
    printf("errors: %lu\n", errors);
    
    if (errors) {
        printf("last error at x: %i, y: %i\n", err_x, err_y);
    }

    return 0;
}
