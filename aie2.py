from aie.iron import Program, Runtime, Worker, GlobalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col4, Tile

buff = GlobalBuffer(data_ty, name="buff")

# Task for the core to perform
def core_fn(buff_in):
    for i in range(data_size):
        buff_in[i] = buff_in[i] + 1

# Create a worker to perform the task
my_worker = Worker(core_fn, [buff], placement=Tile(0, 2))

rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):
    rt.start(my_worker)

my_program = Program(NPU1Col4(), rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)