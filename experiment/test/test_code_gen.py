from Desim.Core import SimSession

from EdgeSim.Chip import EdgeChiplet
from experiment.code_gen import HardwareConfig, TaskConfig, gen_matrix_command, gen_ffn_cross_command, AttenTaskConfig, \
    gen_attention_command


def test_matrix_code_gen():
    SimSession.reset()
    SimSession.init()

    chiplet = EdgeChiplet()

    hardware_config = HardwareConfig(num_pim_unit=16,pim_unit_id_list=[i for i in range(16)],pim_unit_sa_size=(4,128))
    task_config = TaskConfig(matrix_size=(3584,3584),batch_size=1,chunk_size=128,src_addr=100,dst_addr=200)


    compute_command_list = gen_matrix_command(hardware_config,task_config)

    chiplet.load_commands(compute_command_list,[])
    chiplet.config_connection()
    for i in range(3584//128):
        chiplet.l3_memory.direct_write(
            addr = 100 + i,
            data = i,
            check_write_tag=True,
            num_elements=128,
            num_batch_size=1,
            element_bytes=2
        )

    SimSession.scheduler.run()

    print(f"Simulation finished at {SimSession.sim_time}")
    print(f"L3 Memory: {chiplet.l3_memory.memory_data}")




def test_ffn_code_gen():
    SimSession.reset()
    SimSession.init()

    chiplet = EdgeChiplet()

    batch_size = 1

    hardware_config = HardwareConfig(num_pim_unit=16, pim_unit_id_list=[i for i in range(16)],
                                     pim_unit_sa_size=(4, 128))
    task_config_a = TaskConfig(matrix_size=(3584, 18944), batch_size=batch_size, chunk_size=128, src_addr=1000, dst_addr=2000)
    task_config_b = TaskConfig(matrix_size=(3584, 18944), batch_size=batch_size, chunk_size=128, src_addr=3000, dst_addr=4000)
    task_config_c = TaskConfig(matrix_size=(18944,3584), batch_size=batch_size, chunk_size=128, src_addr=5000,dst_addr=6000)
    compute_command_list,vector_command_list = gen_ffn_cross_command(hardware_config,task_config_a,task_config_b,task_config_c)

    chiplet.load_commands(compute_command_list,vector_command_list)

    chiplet.config_connection()

    for i in range(3584//128):
        chiplet.l3_memory.direct_write(
            addr = 1000 + i,
            data = i,
            check_write_tag=True,
            num_elements=128,
            num_batch_size=1,
            element_bytes=2
        )

    for i in range(3584//128):
        chiplet.l3_memory.direct_write(
            addr = 3000 + i,
            data = i,
            check_write_tag=True,
            num_elements=128,
            num_batch_size=1,
            element_bytes=2
        )

    SimSession.scheduler.run()

    print(f"Simulation finished at {SimSession.sim_time}")
    print(f"L3 Memory: {chiplet.l3_memory.memory_data}")




def test_atten_code_gen():
    SimSession.reset()
    SimSession.init()
    chiplet = EdgeChiplet()

    hardware_config = HardwareConfig(num_pim_unit=16, pim_unit_id_list=[i for i in range(16)],
                                     pim_unit_sa_size=(4, 128))

    atten_task_config = AttenTaskConfig(
        batch_size=1,
        chunk_size=128,

        num_q_head = 28,
        num_kv_head = 4,

        head_size=128,
        sequence_length=1024,

        src_addr = 10000,
        dst_addr = 40000,
        temp_s_dst_addr= 20000,
        temp_softmax_dst_addr= 30000,
    )

    compute_command_list,vector_command_list = gen_attention_command(hardware_config,atten_task_config)

    chiplet.load_commands(compute_command_list,vector_command_list)

    chiplet.config_connection()

    for i in range(28):
        # 28 个 chunk
        chiplet.l3_memory.direct_write(
            addr=10000 + i,
            data=i,
            check_write_tag=True,
            num_elements=128,
            num_batch_size=1,
            element_bytes=2
        )


    SimSession.scheduler.run()
    print(f"Simulation finished at {SimSession.sim_time}")
    print(f"L3 Memory: {chiplet.l3_memory.memory_data}")






if __name__ == "__main__":
    # test_matrix_code_gen()

    test_ffn_code_gen()

    # test_atten_code_gen()