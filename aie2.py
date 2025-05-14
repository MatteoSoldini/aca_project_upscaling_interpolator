import numpy as np
import sys

from aie.iron import Program, Runtime, Worker, GlobalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col4, Tile
from aie.iron.controlflow import range_

DATA_SIZE = 48
data_ty = np.ndarray[(DATA_SIZE,), np.dtype[np.int32]]

buff = GlobalBuffer(data_ty, name="buff")

def core_fn(buff_in):
    for i in range_(DATA_SIZE):
        buff_in[i] = buff_in[i] + 1

my_worker = Worker(core_fn, [buff], placement=Tile(0, 2))

rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):
    rt.start(my_worker)

my_program = Program(NPU1Col4(), rt)

module = my_program.resolve_program(SequentialPlacer())

print(module)
