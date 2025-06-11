set -x

rm -rf build
mkdir build
cd build

#python ../aie2.py > aie.mlir

#MLIR_AIE_DIR="$VIRTUAL_ENV/lib/python3.12/site-packages/mlir_aie"
#${PEANO_INSTALL_DIR}/bin/clang++ \
#    -std=c++20 \
#    -O2 \
#    --target=aie2-none-unknown-elf \
#    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
#    -DNDEBUG \
#    -I$MLIR_AIE_DIR/include \
#    -c \
#    -o kernel.o \
#    ../kernel.cpp

#aiecc.py \
#    --aie-generate-xclbin \
#    --no-compile-host \
#    --xclbin-name=final.xclbin \
#    --no-xchesscc \
#    --no-xbridge \
#    --aie-generate-npu-insts \
#    --npu-insts-name=insts.bin \
#    aie.mlir

clang++ \
    -g -O0 \
    -I$XILINX_XRT/include \
    -I../deps \
    -L$XILINX_XRT/lib \
    -lxrt_coreutil -pthread \
    `pkg-config --cflags --libs opencv4` \
    -o main \
    ../main.cpp
