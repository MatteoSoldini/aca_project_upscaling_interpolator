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
scale_factor = 2.0
o_w = int(w * scale_factor)
o_h = int(h * scale_factor)
#tile_size = o_w  # must be multiple of 4
kernel_size = 3

# Define tensor types
#tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.uint8]]
#tile_ty = np.ndarray[(tile_size, tile_size,), np.dtype[np.uint8]]
#scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

out_t = np.ndarray[(o_w * o_h,),np.dtype[np.uint8]]
out_w_t = np.int32
out_h_t = np.int32
in_t = np.ndarray[(w * h,), np.dtype[np.uint8]]
in_w_t = np.int32
in_h_t = np.int32
in_tile_t = np.ndarray[(w,), np.dtype[np.uint8]]
out_tile_t = np.ndarray[(o_w,), np.dtype[np.uint8]]

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

# Data movement
of_in = ObjectFifo(
    in_tile_t,
    name="in",
    default_depth=3
)

of_out = ObjectFifo(
    out_tile_t,
    name="out",
)

def core_fn(of_in, of_out, kernel):
    #for _ in range_(h):
    # The for loop does not seem to be necessary with worker while_true=True
    # This function is executed h (input height) times
    
    in_row = of_in.acquire(1)
#row1 = of_in.acquire(1)
    
    for _ in range(2):
        out_row = of_out.acquire(1)
        kernel(in_row, out_row, o_w)
        of_out.release(1)
    #kernel(row0, elem_out, tile_size)
    #kernel(row0, row0, w, h, elem_out, o_w, o_h, tile_size, 100)

    of_in.release(1)    # release row0

my_worker = Worker(
    core_fn,
    [of_in.cons(), of_out.prod(), nearest_neightbor_fn],
    while_true=True    # If true, will wrap the core_fn in a while(true) loop to ensure it runs until reconfiguration. Defaults to True.
)

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(in_t, out_t) as (a_in, c_out):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), c_out, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())

print(module)
