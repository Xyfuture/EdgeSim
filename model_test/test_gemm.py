
from Desim.Core import SimSession

from EdgeSim.Chip import EdgeChiplet
from EdgeSim.Commands import ComputeCommand


def gen_compute_command()->list[ComputeCommand]:
    compute_command_list = []

    for unit in range(4):
        compute_command_list.append(ComputeCommand(
            opcode='Compute',
            batch_size=1,
            chunk_size=128,
            unit_id=[unit*4,unit*4+1,unit*4+2,unit*4+3],

            dst=200 + unit*4,
            dst_chunk_num=4,

            src_dict={unit*4: 100,unit*4+1:100+4,unit*4+2:100+4*2,unit*4+3:100+4*3},
            src_chunk_num_dict={unit*4: 4,unit*4+1:4,unit*4+2:4,unit*4+3:4},

        ))

    # 第二波指令
    for unit in range(4):
        compute_command_list.append(ComputeCommand(
            opcode='Compute',
            batch_size=1,
            chunk_size=128,
            unit_id=[unit*4,unit*4+1,unit*4+2,unit*4+3],

            dst=300 + unit*4,
            dst_chunk_num=4,

            src_dict={unit*4: 200,unit*4+1:200+4,unit*4+2:200+4*2,unit*4+3:200+4*3},
            src_chunk_num_dict={unit*4: 4,unit*4+1:4,unit*4+2:4,unit*4+3:4},

        ))

    return compute_command_list

if __name__ == '__main__':
    SimSession.reset()
    SimSession.init()

    chiplet = EdgeChiplet()

    compute_command_list = gen_compute_command()

    chiplet.load_commands(compute_command_list,[])

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

