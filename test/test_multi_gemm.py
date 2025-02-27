from Desim.Core import SimSession
from Desim.module.FIFO import FIFO

from EdgeSim.Chip import EdgeChiplet
from EdgeSim.Commands import ComputeCommand


# 应该是一个对称的指令, 生成一份,对所有的都适用
def gen_compute_command_list():
    compute_command_list = []

    for macro_id in range(8):
        compute_command_list.append(
            ComputeCommand(
                opcode='compute',
                batch_size=1,
                chunk_size=128,
                macro_id=[macro_id, macro_id+8],

                dst = 200 + macro_id,
                dst_chunk_num=1,

                src_dict={
                    macro_id: 100,
                    macro_id+8: 100+8,
                },
                src_chunk_num_dict={
                    macro_id: 8,
                    macro_id+8: 8,
                },
            )
        )


    for macro_id in range(8):
        compute_command_list.append(
            ComputeCommand(
                opcode='compute',
                batch_size=1,
                chunk_size=128,

                macro_id=[macro_id,macro_id+8],

                dst = 400 + macro_id,
                dst_chunk_num=1,

                src_dict={
                    macro_id: 300,
                    macro_id+8: 300+8,
                },
                src_chunk_num_dict={
                    macro_id: 8,
                    macro_id+8: 8,
                },
            )
        )


    return compute_command_list

def gen_up_forward_command_list():
    forward_command_list = []

    return forward_command_list


def gen_down_forward_command_list():
    forward_command_list = []

    return forward_command_list



if __name__ == "__main__":
    SimSession.reset()
    SimSession.init()

    chiplet_list = [
        EdgeChiplet() for _ in range(4)
    ]

    chiplet_list[0].load_commands(gen_compute_command_list(),gen_up_forward_command_list())
    chiplet_list[1].load_commands(gen_compute_command_list(),gen_up_forward_command_list())

    chiplet_list[2].load_commands(gen_compute_command_list(),gen_down_forward_command_list())
    chiplet_list[3].load_commands(gen_compute_command_list(),gen_down_forward_command_list())

    # 构建复杂的 fifo
    fifo_0_1 = FIFO(100)
    fifo_1_0 = FIFO(100)
    fifo_2_3 = FIFO(100)
    fifo_3_2 = FIFO(100)

    fifo_0_2 = FIFO(100)
    fifo_2_0 = FIFO(100)
    fifo_1_3 = FIFO(100)
    fifo_3_1 = FIFO(100)

    chiplet_list[0].config_connection(
        {

        }
    )


    for  i in range(8):
        for chiplet in chiplet_list:
            chiplet.l3_memory.direct_write(100+i,i,True,num_elements=128,num_batch_size=1,element_bytes=1)

    SimSession.scheduler.run()
    print(f"Simulation Finish at {SimSession.sim_time}")