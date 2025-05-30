import numpy as np
import sys
import math

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_

dev = NPU1Col1()

w = 640
h = 427
scale_factor = 2.0  # FIXED
o_w = int(w * scale_factor)
o_h = int(h * scale_factor)
#tile_size = o_w  # must be multiple of 4
kernel_size = 5

# Define tensor types
#tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.uint8]]
#tile_ty = np.ndarray[(tile_size, tile_size,), np.dtype[np.uint8]]
#scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

out_t =         np.ndarray[(o_w * o_h,),np.dtype[np.uint8]]
out_w_t =       np.int32
out_h_t =       np.int32
in_t =          np.ndarray[(w * h,), np.dtype[np.uint8]]
in_w_t =        np.int32
in_h_t =        np.int32
in_tile_t =     np.ndarray[(w,), np.dtype[np.uint8]]
out_tile_t =    np.ndarray[(o_w,), np.dtype[np.uint8]]
kernel_t =      np.ndarray[(o_w * kernel_size * o_h, ), np.dtype[np.int32]]
kernel_tile_t = np.ndarray[(o_w * kernel_size, ), np.dtype[np.int32]]

#passthrough_fn = Kernel(
#    "passthrough",
#    "kernel.o",
#    [
#        tile_t,
#        tile_t,
#        np.int32,
#    ],
#)

write_fn = Kernel(
    "write",
    "kernel.o",
    [
        out_tile_t,
        np.int32,
    ],
)

nearest_neightbor_fn = Kernel(
    "nearest_neightbor_2x",
    "kernel.o",
    [
        in_tile_t,
        out_tile_t,
        out_w_t
    ],
)

conv2d3k2x_fn = Kernel(
    "conv2d3k2x",
    "kernel.o",
    [
        in_tile_t,  # in_row_0
        in_tile_t,  # in_row_1
        in_tile_t,  # in_row_2
        out_tile_t, # out_row
        out_w_t     # out_w
    ],
)

conv2d5k2x_fn = Kernel(
    "conv2d5k2x",
    "kernel.o",
    [
        in_tile_t,  # in_row_0
        in_tile_t,  # in_row_1
        in_tile_t,  # in_row_2
        in_tile_t,  # in_row_3
        in_tile_t,  # in_row_4
        out_tile_t, # out_row
        out_w_t,    # out_w
        #kernel_tile_t
    ],
)

# Data movement
in_fifo = ObjectFifo(
    in_tile_t,
    name="in",
    default_depth=3
)

out_fifo = ObjectFifo(
    out_tile_t,
    name="out",
)

kernel_fifo = ObjectFifo(
    kernel_tile_t,
    name="kernel",
)

def core_fn(
    in_fifo,
    out_fifo,
    #kernel_fifo,
    kernel
):
    #for _ in range_(h):
    # The for loop does not seem to be necessary with worker while_true=True
    # This function is executed h (input height) times
    
    # First row
    in_row_0 = in_fifo.acquire(1)
    in_row_1 = in_fifo.acquire(1)
    in_row_2 = in_fifo.acquire(1)

    for _ in range_(2): 
        out_row = out_fifo.acquire(1)
        #k_row = kernel_fifo.acquire(1)
        kernel(
            in_row_0,
            in_row_0,
            in_row_0,
            in_row_1,
            in_row_2,
            out_row,
            o_w,
           # k_row
        )
        out_fifo.release(1)
        #kernel_fifo.release(1)
    
    in_fifo.release(1)
    
    # Second row
    in_row_0 = in_fifo.acquire(1)
    in_row_1 = in_fifo.acquire(1)
    in_row_2 = in_fifo.acquire(1)
    in_row_3 = in_fifo.acquire(1)
 
    for _ in range_(2): 
        #k_row = kernel_fifo.acquire(1)
        out_row = out_fifo.acquire(1)
        kernel(
            in_row_0,
            in_row_0,
            in_row_1,
            in_row_2,
            in_row_3,
            out_row,
            o_w,
            #k_row
        )
        out_fifo.release(1)
        #kernel_fifo.release(1)
    
    in_fifo.release(1)

    # Middle
    for _ in range_(h - 4): 
        in_row_0 = in_fifo.acquire(1)
        in_row_1 = in_fifo.acquire(1)
        in_row_2 = in_fifo.acquire(1)
        in_row_3 = in_fifo.acquire(1)
        in_row_4 = in_fifo.acquire(1) 
        
        for _ in range_(2):
            #k_row = kernel_fifo.acquire(1)
            out_row = out_fifo.acquire(1)
            kernel(
                in_row_0,
                in_row_1,
                in_row_2,
                in_row_3,
                in_row_4,
                out_row,
                o_w,
               # k_row
            )
            out_fifo.release(1)
            #kernel_fifo.release(1)

        in_fifo.release(1)    # release in_row_0
    
    # Second last row
    in_row_0 = in_fifo.acquire(1)
    in_row_1 = in_fifo.acquire(1)
    in_row_2 = in_fifo.acquire(1)
    in_row_3 = in_fifo.acquire(1)
 
    for _ in range_(2): 
        #k_row = kernel_fifo.acquire(1)
        out_row = out_fifo.acquire(1)
        kernel(
            in_row_0,
            in_row_1,
            in_row_2,
            in_row_3,
            in_row_3,
            out_row,
            o_w,
            #k_row
        )
        out_fifo.release(1)
        #kernel_fifo.release(1)

    in_fifo.release(1)
    
    # Last row 
    in_row_0 = in_fifo.acquire(1)
    in_row_1 = in_fifo.acquire(1)    
    in_row_2 = in_fifo.acquire(1)    
    
    for _ in range_(2):
        #k_row = kernel_fifo.acquire(1)
        out_row = out_fifo.acquire(1)
        kernel(
            in_row_0,
            in_row_1,
            in_row_2,
            in_row_2,
            in_row_2,
            out_row,
            o_w,
            #k_row
        )
        out_fifo.release(1)
        #kernel_fifo.release(1)        

    in_fifo.release(3)

my_worker = Worker(
    core_fn,
    [
        in_fifo.cons(),
        out_fifo.prod(),
        #kernel_fifo.cons(),
        conv2d5k2x_fn
    ],
    while_true=False    # If true, will wrap the core_fn in a while(true) loop to ensure it runs until reconfiguration. Defaults to True.
)

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(
    in_t,
    out_t,
    #kernel_t
) as (
    a_in,
    c_out,
    #kernel_buf
):
    rt.start(my_worker)
    rt.fill(in_fifo.prod(), a_in)
    #rt.fill(kernel_fifo.prod(), kernel_buf)
    rt.drain(out_fifo.cons(), c_out, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())

print(module)
