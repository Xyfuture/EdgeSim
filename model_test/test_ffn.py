
from Desim.Core import SimSession

from EdgeSim.Chip import EdgeChiplet
from EdgeSim.Commands import ComputeCommand, VectorCommand, FFNCommand


def gen_compute_command()->list[ComputeCommand]:
    compute_command_list = []

    base_src_addr = 100
    base_dst_addr = 200
    unit_per_group = 8
    for group_id in range(2):
        compute_command_list.append(ComputeCommand(
            opcode='Compute',
            batch_size=1,
            chunk_size=128,
            unit_id=[group_id*unit_per_group + i for i in range(unit_per_group)],

            dst=base_dst_addr + group_id * 16,
            dst_chunk_num=16,

            src_dict={group_id * unit_per_group + i: base_src_addr + i * 2 for i in range(unit_per_group)},
            src_chunk_num_dict={group_id*unit_per_group+i:2 for i in range(unit_per_group)},

        ))

    # 第二波指令
    base_src_addr = 300
    base_dst_addr = 400
    unit_per_group = 8
    for group_id in range(2):
        compute_command_list.append(ComputeCommand(
            opcode='Compute',
            batch_size=1,
            chunk_size=128,
            unit_id=[group_id * unit_per_group + i for i in range(unit_per_group)],

            dst=base_dst_addr + group_id * 8,
            dst_chunk_num=8,

            src_dict={group_id * unit_per_group + i: base_src_addr + i * 2 for i in
                      range(unit_per_group)},
            src_chunk_num_dict={group_id * unit_per_group + i: 2 for i in range(unit_per_group)},

        ))

    return compute_command_list


def gen_vector_command()->list[VectorCommand]:
    vector_command_list = [
        FFNCommand(
            opcode='Vector',
            batch_size=1,
            chunk_size=128,

            dst=300,
            dst_chunk_num=16,

            src_chunk_num=16,
            src_activation=200,
            src_mul=200+16,

            activation=True,
            mul=True,

        )
    ]


    return vector_command_list



if __name__ == '__main__':
    SimSession.reset()
    SimSession.init()

    chiplet = EdgeChiplet()

    compute_command_list = gen_compute_command()
    vector_command_list = gen_vector_command()

    chiplet.load_commands(compute_command_list,vector_command_list)

    chiplet.config_connection()

    # 写入初始数据
    for i in range(16):
        chiplet.l3_memory.direct_write(
            addr=100 + i,
            data=i,
            check_write_tag=True,
            num_elements=128,
            num_batch_size=1,
            element_bytes=2
        )

    SimSession.scheduler.run()

    print(f"Simulation Finish at {SimSession.sim_time}")

    print(f"L3 Memory: {chiplet.l3_memory.memory_data}")

