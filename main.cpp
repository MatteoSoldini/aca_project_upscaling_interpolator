#include <stdio.h>
#include <fstream>
#include <string.h>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

void init_xrt_load_kernel(xrt::device &device, xrt::kernel &kernel,
                                      std::string xclbinFileName,
                                      std::string kernelNameInXclbin) {
    // Get a device handle
    unsigned int device_index = 0;
    device = xrt::device(device_index);

    auto xclbin = xrt::xclbin(xclbinFileName);

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                    [kernelNameInXclbin](xrt::xclbin::kernel &k) {
                      auto name = k.get_name();
                      return name.rfind(kernelNameInXclbin, 0) == 0;
                    });
    auto kernelName = xkernel.get_name();

    device.register_xclbin(xclbin);

    xrt::hw_context context(device, xclbin.get_uuid());

    kernel = xrt::kernel(context, kernelName);

    return;
}

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

int main(void) {
    std::vector<uint32_t> instr_v = load_file("build/insts.bin");

    xrt::device device;
    xrt::kernel kernel;    

    init_xrt_load_kernel(device, kernel, "build/final.xclbin", "MLIR_AIE"); 

    printf("kernel created\n");

    uint32_t IN_SIZE = 4096;
    uint32_t OUT_SIZE = IN_SIZE;
    int32_t scaleFactor = 2;    

    // set up the buffer objects
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_inFactor = xrt::bo(device, 1 * sizeof(int32_t),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_outC = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    // Copy instruction stream to xrt buffer object
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    // Initialize buffer bo_inA
    int32_t *bufInA = bo_inA.map<int32_t *>();
    for (int i = 0; i < IN_SIZE; i++)
        bufInA[i] = i + 1;

    // Initialize buffer bo_inFactor
    int32_t *bufInFactor = bo_inFactor.map<int32_t *>();
    *bufInFactor = scaleFactor;

    // Zero out buffer bo_outC
    int32_t *bufOut = bo_outC.map<int32_t *>();
    memset(bufOut, 0, OUT_SIZE * sizeof(int32_t));

    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inFactor.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    unsigned int opcode = 3;
    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inFactor, bo_outC);
    run.wait();
    
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);    

    for (size_t i = 0; i < OUT_SIZE; i++) {
        printf("%i:\t %i\n", bufInA[i], bufOut[i]);
    }

    return 0;
}
