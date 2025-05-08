from Desim.Core import SimSession

from EdgeSim.Chip import EdgeChiplet
from EdgeSim.Commands import ComputeCommand, VectorCommand


def gen_compute_command()->list[ComputeCommand]:
    pass


def gen_vector_command()->list[VectorCommand]:
    pass




if __name__ == "__main__":
    SimSession.reset()
    SimSession.init()

    chiplet = EdgeChiplet()

    compute_command_list = gen_compute_command()
    vector_command_list = gen_vector_command()

    chiplet.load_commands(compute_command_list,vector_command_list)
    chiplet.config_connection()

    # 设置初始默认的数据

    SimSession.scheduler.run()

    print(f"Simulation Finish at {SimSession.sim_time}")
    print(f"L3 Memory: {chiplet.l3_memory.memory_data}")


