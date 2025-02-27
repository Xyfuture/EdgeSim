from Desim.Core import SimSession
from Desim.module.FIFO import FIFO

from EdgeSim.Chip import EdgeChiplet
from EdgeSim.Commands import ComputeCommand




def gen_compute_command():
    compute_command_list = []

    for macro_id in range(16):
        compute_command_list.append(ComputeCommand(
            opcode = 'Compute',
            batch_size=1,
            chunk_size=128,
            macro_id=[macro_id],

            dst = 200 + macro_id,
            dst_chunk_num= 1 ,

            src_dict={macro_id:100},
            src_chunk_num_dict={macro_id:16},

        ))

    # 第二波指令
    for macro_id in range(16):
        compute_command_list.append(ComputeCommand(
            opcode='Compute',
            batch_size=1,
            chunk_size=128,
            macro_id=[macro_id],

            dst = 300 + macro_id,
            dst_chunk_num= 1 ,

            src_dict={macro_id:200},
            src_chunk_num_dict={macro_id:16},
        ))

    return compute_command_list




if __name__ == '__main__':
    SimSession.reset()
    SimSession.init()

    chiplet = EdgeChiplet()

    compute_command_list = gen_compute_command()

    chiplet.load_commands(compute_command_list,[])

    chiplet.config_connection({1:FIFO(10),2:FIFO(20),3:FIFO(10),4:FIFO(10)})

    for i in range(16):
        chiplet.l3_memory.direct_write(100+i,i,True,num_elements=128,num_batch_size=1,element_bytes=1)


    SimSession.scheduler.run()
    print(f"Simulation Finish at {SimSession.sim_time}")