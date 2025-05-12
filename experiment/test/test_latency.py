from Desim.Core import SimSession

from EdgeSim.Chip import EdgeChiplet
from experiment.code_gen import HardwareConfig, TaskConfig, gen_matrix_command


def test_latency():
    SimSession.reset()
    SimSession.init()

    chiplet = EdgeChiplet()

    hardware_config = HardwareConfig(
        num_pim_unit= 16, pim_unit_id_list=[i for i in range(16)],
        pim_unit_sa_size= (4,128)
    )

    task_config = TaskConfig(
        matrix_size=(3584,3584),batch_size=1,chunk_size=128,src_addr=1000,dst_addr=2000
    )

    compute_command_list = gen_matrix_command(hardware_config,task_config)

    chiplet.load_commands(compute_command_list,[])
    chiplet.config_connection()

    for i in range(3584//128):
        chiplet.l3_memory.direct_write(
            addr = 1000+i,
            data = i,
            check_write_tag = True,
            num_elements=128,
            num_batch_size=1,
            element_bytes=2 ,
        )

    SimSession.scheduler.run()

    print(f"Simulation finished at {SimSession.sim_time}")
    print(f"L3 Memory: {chiplet.l3_memory.memory_data}")



if __name__ == "__main__":
    test_latency()
