#include <stdio.h>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

int main(void) {
    xrt::device device = xrt::device(0);
    printf("device: %s\n", device.get_info<xrt::info::device::name>().c_str());
	
	// buffers are 4k aligned
    auto a_buf = xrt::bo(device, 1024, kernel.group_id(0));
	auto c_buf = xrt::bo(device, 1024, kernel.group_id(1));
	auto a_buf = xrt::bo(device, 1024, kernel.group_id(0));
	auto a_buf = xrt::bo(device, 1024, kernel.group_id(0));

	auto xclbin_uuid = device.load_xclbin("kernel.xclbin");
    auto krnl = xrt::kernel(device, xclbin_uuid, "kernel");

    return 0;
}
