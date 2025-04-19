//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for further information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Main file containing the main function and routines for kernel execution,
// using the utilities defined in utils.hpp/ utils.cpp.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cassert>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <cfloat>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// Include XRT libraries
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Include the utilities header
#include "utils.hpp"

//--------------------------------------------------------------------------
// Buffer size macros (modify if necessary)
//--------------------------------------------------------------------------
#ifndef IN1_SIZE
#define IN1_SIZE 1024     // Example: change as needed
#endif

#ifndef IN2_SIZE
#define IN2_SIZE 4        // For scalar multiplication, one element is sufficient
#endif

#ifndef OUT_SIZE
#define OUT_SIZE 1024     // Example: change as needed
#endif

//--------------------------------------------------------------------------
// XRT Wrapper struct and functions (only those used)
//--------------------------------------------------------------------------

/**
 * @brief Structure to hold command-line arguments.
 *
 * This struct stores all the necessary parameters read from the command line.
 */
struct args {
  int verbosity;
  int do_verify;
  int n_iterations;
  int n_warmup_iterations;
  int trace_size;
  std::string instr;
  std::string xclbin;
  std::string kernel;
  std::string trace_file;
};

/**
 * @brief Parses the command-line arguments.
 *
 * This function defines the allowed options, parses the arguments using boost::program_options,
 * and fills in the args structure.
 *
 * @param argc The count of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @return A populated args structure.
 */
args parse_args(int argc, const char *argv[]) {
  po::options_description desc("Allowed options");
  po::variables_map vm;
  utils::add_default_options(desc);
  args myargs;
  utils::parse_options(argc, argv, desc, vm);
  myargs.verbosity = vm["verbosity"].as<int>();
  myargs.do_verify = vm["verify"].as<bool>();
  myargs.n_iterations = vm["iters"].as<int>();
  myargs.n_warmup_iterations = vm["warmup"].as<int>();
  myargs.trace_size = vm["trace_sz"].as<int>();
  myargs.instr = vm["instr"].as<std::string>();
  myargs.xclbin = vm["xclbin"].as<std::string>();
  myargs.kernel = vm["kernel"].as<std::string>();
  myargs.trace_file = vm["trace_file"].as<std::string>();
  return myargs;
}

/**
 * @brief XRT-based test wrapper for two inputs and one output.
 *
 * This templated function sets up the device, buffers, and kernel, then runs the kernel
 * for a specified number of iterations (including warmup iterations). It also verifies the result.
 *
 * Template parameters:
 * - T1, T2, T3: Data types for input1, input2, and output.
 * - init_bufIn1: Function pointer to initialize the first input buffer.
 * - init_bufIn2: Function pointer to initialize the second input buffer.
 * - init_bufOut: Function pointer to initialize the output buffer.
 * - verify_vector_scalar_mul: Function pointer to verify the result.
 *
 * @param IN1_VOLUME Volume of the first input buffer (number of elements).
 * @param IN2_VOLUME Volume of the second input buffer (number of elements).
 * @param OUT_VOLUME Volume of the output buffer (number of elements).
 * @param myargs Command-line arguments.
 * @return 0 if the test passes, 1 if there are errors.
 */
template <typename T1, typename T2, typename T3,
          void (*init_bufIn1)(T1 *, int),
          void (*init_bufIn2)(T2 *, int),
          void (*init_bufOut)(T3 *, int),
          int (*verify_vector_scalar_mul)(T1 *, T2 *, T3 *, int, int)>
int setup_and_run_aie(int IN1_VOLUME, int IN2_VOLUME, int OUT_VOLUME, args myargs) {
  srand(time(NULL));
  std::vector<uint32_t> instr_v = utils::load_instr_binary(myargs.instr);
  if (myargs.verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  xrt::device device;
  xrt::kernel kernel;
  utils::init_xrt_load_kernel(device, kernel, myargs.verbosity,
                              myargs.xclbin, myargs.kernel);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in1 = xrt::bo(device, IN1_VOLUME * sizeof(T1), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_in2 = xrt::bo(device, IN2_VOLUME * sizeof(T2), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_VOLUME * sizeof(T3), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  int tmp_trace_size = (myargs.trace_size > 0) ? myargs.trace_size : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  if (myargs.verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  T1 *bufIn1 = bo_in1.map<T1 *>();
  T2 *bufIn2 = bo_in2.map<T2 *>();
  T3 *bufOut = bo_out.map<T3 *>();
  char *bufTrace = bo_trace.map<char *>();

  init_bufIn1(bufIn1, IN1_VOLUME);
  init_bufIn2(bufIn2, IN2_VOLUME);
  init_bufOut(bufOut, OUT_VOLUME);

  // TRACE: To enable tracing, comment out the following line(s)
  // if (myargs.trace_size > 0)
  //   memset(bufTrace, 0, myargs.trace_size);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // TRACE: To enable tracing, comment out the following line(s)
  // if (myargs.trace_size > 0)
  //   bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = myargs.n_iterations + myargs.n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;
  int errors = 0;

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_in2, bo_out, bo_tmp1, bo_trace);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (myargs.trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < myargs.n_warmup_iterations)
      continue;

    if (myargs.do_verify) {
      if (myargs.verbosity >= 1)
        std::cout << "Verifying results ..." << std::endl;
      auto vstart = std::chrono::system_clock::now();
      errors += verify_vector_scalar_mul(bufIn1, bufIn2, bufOut, IN1_VOLUME, myargs.verbosity);
      auto vstop = std::chrono::system_clock::now();
      float vtime = std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart).count();
      if (myargs.verbosity >= 1)
        std::cout << "Verify time: " << vtime << " secs." << std::endl;
    } else {
      if (myargs.verbosity >= 1)
        std::cout << "WARNING: results not verified." << std::endl;
    }

    // TRACE: To enable tracing, comment out the following block
    // if (myargs.trace_size > 0 && iter == myargs.n_warmup_iterations) {
    //   utils::write_out_trace(bufTrace, myargs.trace_size, myargs.trace_file);
    // }

    float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  float macs = 0;
  std::cout << std::endl << "Avg NPU time: " << npu_time_total / myargs.n_iterations << " us." << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: " << macs / (1000 * npu_time_total / myargs.n_iterations) << std::endl;
  std::cout << std::endl << "Min NPU time: " << npu_time_min << " us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;
  std::cout << std::endl << "Max NPU time: " << npu_time_max << " us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}

/**
 * @brief Initializes the first input buffer.
 *
 * Fills bufIn1 with consecutive integer values starting from 1.
 *
 * @param bufIn1 Pointer to the first input buffer.
 * @param SIZE Number of elements in the buffer.
 */
void initialize_bufIn1(std::int32_t *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn1[i] = i + 1;
}

/**
 * @brief Initializes the second input buffer.
 *
 * Sets the first element of bufIn2 to a scaling factor (3).
 *
 * @param bufIn2 Pointer to the second input buffer.
 * @param SIZE Number of elements in the buffer.
 */
void initialize_bufIn2(std::int32_t *bufIn2, int SIZE) {
  bufIn2[0] = 3; // scaleFactor
}

/**
 * @brief Initializes the output buffer.
 *
 * Clears the output buffer by setting all its bytes to zero.
 *
 * @param bufOut Pointer to the output buffer.
 * @param SIZE Number of bytes in the buffer.
 */
void initialize_bufOut(std::int32_t *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

/**
 * @brief Verifies the vector-scalar multiplication result.
 *
 * Compares each element of the output buffer with the expected result computed as
 * bufIn1[i] multiplied by bufIn2[0]. Increases the error count if any discrepancy is found.
 *
 * @param bufIn1 Pointer to the first input buffer.
 * @param bufIn2 Pointer to the second input buffer.
 * @param bufOut Pointer to the output buffer.
 * @param SIZE Number of elements in the input/output buffers.
 * @param verbosity Verbosity level for logging.
 * @return The number of errors found.
 */
int verify_vector_scalar_mul(std::int32_t *bufIn1, std::int32_t *bufIn2,
                             std::int32_t *bufOut, int SIZE, int verbosity) {
  int errors = 0;
  for (int i = 0; i < SIZE; i++) {
    int32_t ref = bufIn1[i] * bufIn2[0];
    int32_t test = bufOut[i];
    if (test != ref) {
      if (verbosity >= 1)
        std::cout << "Error in output " << test << " != " << ref << std::endl;
      errors++;
    } else {
      if (verbosity >= 1)
        std::cout << "Correct output " << test << " == " << ref << std::endl;
    }
  }
  return errors;
}

/**
 * @brief Main function.
 *
 * Sets up the buffers, parses command-line arguments, launches the kernel execution,
 * and returns the test result.
 *
 * @param argc The count of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @return 0 on success, 1 on failure.
 */
int main(int argc, const char *argv[]) {
  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(std::int32_t);
  constexpr int IN2_VOLUME = IN2_SIZE / sizeof(std::int32_t);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(std::int32_t);

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<std::int32_t, std::int32_t, std::int32_t,
                              initialize_bufIn1, initialize_bufIn2,
                              initialize_bufOut, verify_vector_scalar_mul>(
      IN1_VOLUME, IN2_VOLUME, OUT_VOLUME, myargs);
  return res;
}
