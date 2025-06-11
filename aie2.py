import numpy as np
import sys
import math

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_

def convolution_module(dev, in_w, in_h, scale_factor):
    out_w = int(in_w * scale_factor)
    out_h = int(in_h * scale_factor)
    c_mtx_cols = int(scale_factor)

    out_t =       np.ndarray[(out_w * out_h,),np.dtype[np.uint8]]
    out_w_t =     np.int32
    out_h_t =     np.int32
    in_t =        np.ndarray[(in_w * in_h,), np.dtype[np.uint8]]
    in_w_t =      np.int32
    in_h_t =      np.int32
    in_tile_t =   np.ndarray[(in_w,), np.dtype[np.uint8]]
    out_tile_t =  np.ndarray[(out_w,), np.dtype[np.uint8]]
    c_mtx_t =     np.ndarray[(c_mtx_cols * c_mtx_cols * 16, ), np.dtype[np.uint16]]
    c_mtx_row_t = np.ndarray[(16 * c_mtx_cols, ), np.dtype[np.uint16]]
    
    conv2d4k_fn = Kernel(
        "conv2d4k",
        "kernel.o",
        [
            in_tile_t,      # in_row_0
            in_tile_t,      # in_row_1
            in_tile_t,      # in_row_2
            in_tile_t,      # in_row_3
            c_mtx_row_t,    # c_mtx_row
            np.int32,       # c_mtx_cols
            out_tile_t,     # out_row
            out_w_t         # out_w
        ],
    )
    
    # Data movement
    in_fifo = ObjectFifo(
        in_tile_t,
        name="in",
        default_depth=4
    )
    
    out_fifo = ObjectFifo(
        out_tile_t,
        name="out",
    )
    
    c_mtx_fifo = ObjectFifo(
        c_mtx_row_t,
        name="c_mtx",
        default_depth=c_mtx_cols
    )
    
    def core_fn(
        c_mtx_fifo,
        in_fifo,
        out_fifo,
        kernel
    ):
        k_row = c_mtx_fifo.acquire(c_mtx_cols)
    
        # First row
        in_row = in_fifo.acquire(3)
        
        for i in range(c_mtx_cols):
            out_row = out_fifo.acquire(1)
            kernel(
                in_row[0],
                in_row[0],
                in_row[1],
                in_row[2],
                k_row[i],
                c_mtx_cols,
                out_row,
                out_w
            )
            out_fifo.release(1)
        
        # Middle
        for _ in range_(in_h - 3):
            in_row = in_fifo.acquire(4)
     
            for i in range(c_mtx_cols):
                out_row = out_fifo.acquire(1)
                kernel(
                    in_row[0],
                    in_row[1],
                    in_row[2],
                    in_row[3],
                    k_row[i],
                    c_mtx_cols,
                    out_row,
                    out_w
                )
                out_fifo.release(1)
            
            in_fifo.release(1)
        
        # Second last row
        in_row = in_fifo.acquire(3)
        
        for i in range(c_mtx_cols):
            out_row = out_fifo.acquire(1)
            kernel(
                in_row[0],
                in_row[1],
                in_row[2],
                in_row[2],
                k_row[i],
                c_mtx_cols,
                out_row,
                out_w
            )
            out_fifo.release(1)
        
        in_fifo.release(1)
     
        # Last row 
        in_row = in_fifo.acquire(2)
        
        for i in range(c_mtx_cols):
            out_row = out_fifo.acquire(1)
            kernel(
                in_row[0],
                in_row[1],
                in_row[1],
                in_row[1],
                k_row[i],
                c_mtx_cols,
                out_row,
                out_w
            )
            out_fifo.release(1)
    
        in_fifo.release(2)
        
        c_mtx_fifo.release(c_mtx_cols)
    
    my_worker = Worker(
        core_fn,
        [
            c_mtx_fifo.cons(),
            in_fifo.cons(),
            out_fifo.prod(),
            conv2d4k_fn
        ],
        while_true=False    # If true, will wrap the core_fn in a while(true) loop to ensure it runs until reconfiguration. Defaults to True.
    )

    
    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(
        in_t,
        out_t,
        c_mtx_t,
    ) as (
        a_in,
        c_out,
        c_mtx_buf,
    ):
        rt.start(my_worker)
        rt.fill(in_fifo.prod(), a_in)
        rt.fill(c_mtx_fifo.prod(), c_mtx_buf)
        rt.drain(out_fifo.cons(), c_out, wait=True)
    
    my_program = Program(dev, rt)
    return my_program.resolve_program(SequentialPlacer())

assert len(sys.argv) == 4, "Expecting 3 arguments: input width, input height, scale factor"

in_w = int(sys.argv[1])
in_h = int(sys.argv[2])
scale_factor = float(sys.argv[3])

assert in_w % 32 == 0, "Expecting a 32bit aligned input width"

dev = NPU1Col1()
module = convolution_module(dev, in_w, in_h, scale_factor)

print(module)
