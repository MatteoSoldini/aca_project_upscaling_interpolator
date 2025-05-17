mkdir -p build
cd build

python ../aie2.py > aie.mlir

MLIR_AIE_DIR="$VIRTUAL_ENV/lib/python3.12/site-packages/mlir_aie"
echo $MLIR_AIE_DIR

${PEANO_INSTALL_DIR}/bin/clang++ \
    -O2 \
    -std=c++20 \
    --target=aie2-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -DNDEBUG \
    -I $MLIR_AIE_DIR/include \
    -c \
    ../kernel.cc \
    -o kernel.o 

aiecc.py \
    --aie-generate-xclbin \
    --no-compile-host \
    --xclbin-name=kernel.xclbin \
    --no-xchesscc \
    --no-xbridge \
    --aie-generate-npu-insts \
    --npu-insts-name=insts.bin \
    aie.mlir

g++ \
    -std=c++17 \
    -I$XILINX_XRT/include \
    -L$XILINX_XRT/lib \
    -o main \
    ../main.cpp \
    -lxrt_coreutil -pthread
