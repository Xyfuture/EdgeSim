from Desim.Core import SimSession
from Desim.module.FIFO import FIFO

from EdgeSim.Chip import EdgeChiplet
from EdgeSim.Commands import ComputeCommand,ForwardCommand


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

                dst = 250 + macro_id,
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


    # 第二个矩阵的部分
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

    for macro_id in range(8):
        compute_command_list.append(
            ComputeCommand(
                opcode='compute',
                batch_size=1,
                chunk_size=128,

                macro_id=[macro_id,macro_id+8],

                dst = 450 + macro_id,
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

def gen_forward_command_list(compute_dst_base:int,next_src_base:int,top:bool,left:bool,):
    forward_command_list = []

    # 发送自己的 part 数据
    
    if top:
        part_src = compute_dst_base+4
    else:
        part_src = compute_dst_base

    forward_command_list.append(
        ForwardCommand(
            opcode= 'Forward',
            batch_size=1,
            chunk_size=128,
            chunk_num=4,

            memory_dst_flag=False,
            memory_dst=-1,
            fifo_dst_flag=True,
            fifo_dst=1,

            memory_src_flag=True,
            memory_src=part_src,

            fifo_src_flag=False,
            fifo_src=-1,

            reduce=False

        )
    )


    # 接收来自其他 chiplet 的完整数据
    
    if left:
        full_recv_dst = next_src_base + 8 
    else:
        full_recv_dst = next_src_base
    forward_command_list.append(
        ForwardCommand(
            opcode='Forward',
            batch_size=1,
            chunk_size=128,
            chunk_num=4,

            memory_dst_flag=True,
            memory_dst=full_recv_dst,
            fifo_dst_flag=False,
            fifo_dst=-1,

            memory_src_flag=False,
            memory_src=-1,
            fifo_src_flag=True,
            fifo_src=4,

            reduce=False
        )
    )



    # 进行reduce 并转发给其他 chiplet

    if top:
        reduce_memory_src = compute_dst_base
    else:
        reduce_memory_src = compute_dst_base + 4


    if left:
        reduce_memory_dst = next_src_base
    else:
        reduce_memory_dst = next_src_base + 8
    
    forward_command_list.append(
        ForwardCommand(
            opcode='Forward',
            batch_size=1,
            chunk_size=128,
            chunk_num=4,

            memory_dst_flag=True,
            memory_dst=reduce_memory_dst,
            fifo_dst_flag=True,
            fifo_dst=2,

            memory_src_flag=True,
            memory_src=reduce_memory_src,
            fifo_src_flag=True,
            fifo_src=3,

            reduce=True
        )
    )




    return forward_command_list



def gen_full_forward_command_list(top:bool,left:bool):
    first_round = gen_forward_command_list(200,300,top,left)
    second_round = gen_forward_command_list(250,304,top,left)

    return first_round + second_round



if __name__ == "__main__":
    SimSession.reset()
    SimSession.init()

    chiplet_list = [
        EdgeChiplet() for _ in range(4)
    ]

    chiplet_list[0].load_commands(gen_compute_command_list(),gen_full_forward_command_list(True,True))
    chiplet_list[1].load_commands(gen_compute_command_list(),gen_full_forward_command_list(True,False))

    chiplet_list[2].load_commands(gen_compute_command_list(),gen_full_forward_command_list(False,True))
    chiplet_list[3].load_commands(gen_compute_command_list(),gen_full_forward_command_list(False,False))

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
        { 1:fifo_0_2,2:fifo_0_1,3:fifo_2_0,4:fifo_1_0 }
    )

    chiplet_list[1].config_connection(
        { 1:fifo_1_3,2:fifo_1_0,3:fifo_3_1,4:fifo_0_1 }
    )

    chiplet_list[2].config_connection(
        { 1:fifo_2_0,2:fifo_2_3,3:fifo_0_2,4:fifo_3_2 }
    )

    chiplet_list[3].config_connection(
        { 1:fifo_3_1,2:fifo_3_2,3:fifo_1_3,4:fifo_2_3 }
    )


    for  i in range(16):
        for chiplet in chiplet_list:
            chiplet.l3_memory.direct_write(100+i,i,True,num_elements=128,num_batch_size=1,element_bytes=1)

    SimSession.scheduler.run()
    print(f"Simulation Finish at {SimSession.sim_time}")