from dataclasses import dataclass
from ftplib import all_errors
from os.path import commonpath
from typing import Optional, Callable

from Desim.Core import SimModule, SimSession, SimCoroutine, SimTime
from Desim.Sync import SimSemaphore
from Desim.memory.Memory import ChunkMemory, ChunkMemoryPort, ChunkPacket
from Desim.module.FIFO import FIFO
from IPython.terminal.shortcuts.filters import PassThrough

from EdgeSim.Commands import ComputeCommand, AttenComputeCommand


@dataclass
class PIMEngineConfig:
    pim_unit_num:int = 16


@dataclass
class PIMUnitConfig:
    sa_rows:int = 4
    sa_cols:int = 128

    rram_bandwidth = 128 # 对应下来是 2T 左右的带宽
    dram_bandwidth = 300

    quantization_unit_num = 16
    dequantization_unit_num = 16




class PIMUnit(SimModule):
    def __init__(self,unit_id:int = -1):
        super().__init__()

        self.pim_unit_config = PIMUnitConfig()

        self.load_fifo:FIFO = FIFO(100)
        self.store_fifo:FIFO = FIFO(100)

        self.unit_id = unit_id

        self.quantize_engine_command_queue:FIFO[ComputeCommand] = FIFO(1)
        self.compute_engine_command_queue:FIFO[ComputeCommand] = FIFO(1)
        self.dequantize_engine_command_queue:FIFO[ComputeCommand] = FIFO(1)

        self.quantize_to_compute_fifo = FIFO(100)
        self.compute_to_dequantize_fifo = FIFO(100)


        self.register_coroutine(self.quantize_engine)
        self.register_coroutine(self.compute_engine)
        self.register_coroutine(self.dequantize_engine)

    # def issue_command(self,command:ComputeCommand):
    #     self.quantize_engine_command_queue.write(command)
    #     self.compute_engine_command_queue.write(command)
    #     self.dequantize_engine_command_queue.write(command)

    def load_command(self,command_list:list[ComputeCommand]):
        command_size = len(command_list)

        self.quantize_engine_command_queue = FIFO(command_size,command_size,command_list)
        self.compute_engine_command_queue = FIFO(command_size,command_size,command_list)
        self.dequantize_engine_command_queue = FIFO(command_size,command_size,command_list)



    def quantize_engine(self):
        while True:
            command:ComputeCommand = self.quantize_engine_command_queue.read()

            for i in range(command.src_chunk_num_dict[self.unit_id]):
                chunk_packet = self.load_fifo.read()

                latency = command.chunk_size // self.pim_unit_config.quantization_unit_num

                SimModule.wait_time(SimTime(latency))

                self.quantize_to_compute_fifo.write(
                    ChunkPacket(
                        None,
                        command.chunk_size,
                        command.batch_size,
                        1
                    )
                )



    def compute_engine(self):
        while True:
            command:ComputeCommand = self.compute_engine_command_queue.read()

            # 这里需要进行一个拆分, compute 和 atten compute 需要拆分开

            if type(command) == ComputeCommand:
                for i in range(command.dst_chunk_num):
                    for j in range(command.src_chunk_num_dict[self.unit_id]):
                        if i == 0 :
                            chunk_packet = self.quantize_to_compute_fifo.read()


                        input_size = (command.batch_size,command.chunk_size)
                        matrix_size = (command.chunk_size,command.chunk_size)

                        last_time = False
                        if j == command.src_chunk_num_dict[self.unit_id] - 1:
                            last_time = True

                        memory_bandwidth = self.pim_unit_config.rram_bandwidth

                        latency = self.calc_execution_time(input_size,matrix_size,memory_bandwidth,last_time)

                        # if self.unit_id == 0:
                        #     print(f'PIM Unit {self.unit_id} compute_start_time {SimSession.sim_time} - compute_latency {latency}')

                        SimModule.wait_time(SimTime(latency))

                    # 运算完一个块
                    self.compute_to_dequantize_fifo.write(
                        ChunkPacket(
                            None,
                            command.chunk_size,
                            command.batch_size,
                            4
                        )
                    )

                    # print(f"PIM Unit {self.unit_id} compute at dst{i}")
            elif type(command) == AttenComputeCommand and isinstance(command,AttenComputeCommand): # 为了type hint
                # 针对 attention 计算的部分

                for i in range(command.dst_chunk_num):
                    for j in range(command.src_chunk_num_dict[self.unit_id]):
                        if i == 0:
                            chunk_packet = self.quantize_to_compute_fifo.read()

                        input_size = (command.batch_size, command.chunk_size)
                        matrix_size = (command.chunk_size, command.chunk_size)

                        last_time = False
                        if j == command.src_chunk_num_dict[self.unit_id] - 1:
                            last_time = True

                        memory_bandwidth = int((command.running_head / command.total_head) * self.pim_unit_config.dram_bandwidth)

                        latency = self.calc_execution_time(input_size, matrix_size,memory_bandwidth , last_time)

                        SimModule.wait_time(SimTime(latency))

                    # 运算完一个块
                    self.compute_to_dequantize_fifo.write(
                        ChunkPacket(
                            None,
                            command.chunk_size,
                            command.batch_size,
                            4
                        )
                    )

                    # print(f"PIM Unit {self.unit_id} compute at dst{i}")

            # if self.unit_id == 0 :
            #     print(f"PIM Unit {self.unit_id} finish command at time {SimSession.sim_time} - dst_addr {command.dst}")


    def dequantize_engine(self):
        while True:
            command:ComputeCommand = self.dequantize_engine_command_queue.read()

            for i in range(command.dst_chunk_num):
                chunk_packet = self.compute_to_dequantize_fifo.read()

                latency = command.chunk_size // self.pim_unit_config.quantization_unit_num

                SimModule.wait_time(SimTime(latency))

                self.store_fifo.write(
                    ChunkPacket(
                        None,
                        command.chunk_size,
                        command.batch_size,
                        2
                    )
                )


    def calc_execution_time(self, input_size:tuple[int,int], matrix_size:tuple[int,int], memory_bandwidth:int,last_time:bool=False):
        # 计算执行矩阵操作的执行时间 直接按照output stationary的方式进行

        # 取 计算时间和memory时间比较长的那一个

        # sa_compute_latency = 0
        # sa_memory_latency = 0


        sa_rows,sa_cols = self.pim_unit_config.sa_rows,self.pim_unit_config.sa_cols


        # 模拟纯粹的计算延迟
        assert input_size[0] <= self.pim_unit_config.sa_rows
        assert input_size[1] == matrix_size[0]

        if not last_time:
            sa_compute_latency = input_size[1]
            sa_memory_latency = matrix_size[0] * matrix_size[1] // memory_bandwidth
        else:
            sa_compute_latency = input_size[1] + sa_cols + sa_rows
            sa_memory_latency = matrix_size[0] * matrix_size[1] // memory_bandwidth

        return max(sa_compute_latency,sa_memory_latency)








class PIMEngine(SimModule):
    def __init__(self):
        super().__init__()

        self.external_l3_memory: Optional[ChunkMemory] = None
        self.compute_command_queue: Optional[FIFO] = None

        self.pim_unit_list: list[PIMUnit] = []

        self.pim_unit_load_semaphore_list: list[SimSemaphore] = []
        self.pim_unit_store_semaphore_list: list[SimSemaphore] = []

        self.load_engine_command_queue:FIFO = FIFO(100)
        self.store_engine_command_queue:FIFO = FIFO(100)

        self.pim_engine_config = PIMEngineConfig()

        for i in range(self.pim_engine_config.pim_unit_num):
            self.pim_unit_list.append(PIMUnit(i))
            self.pim_unit_load_semaphore_list.append(SimSemaphore(1))
            self.pim_unit_store_semaphore_list.append(SimSemaphore(1))

        self.register_coroutine(self.load_engine)
        self.register_coroutine(self.store_engine)

        # self.register_coroutine(self.process)


    def load_command(self,command_list:list[ComputeCommand]):
        command_size = len(command_list)

        self.compute_command_queue = FIFO(command_size,command_size,command_list)
        self.load_engine_command_queue = FIFO(command_size,command_size,command_list)
        self.store_engine_command_queue = FIFO(command_size,command_size,command_list)

        all_pim_unit_command_list = [[] for i in range(self.pim_engine_config.pim_unit_num)]

        for command in command_list:
            for unit_id in command.unit_id:
                all_pim_unit_command_list[unit_id].append(command)

        for unit_id, pim_unit_command_list in enumerate(all_pim_unit_command_list):
            self.pim_unit_list[unit_id].load_command(pim_unit_command_list)



    def load_engine(self):

        while True:
            if self.load_engine_command_queue.is_empty():
                return

            command:ComputeCommand = self.load_engine_command_queue.read()

            # 申请资源
            for unit_id in command.unit_id:
                self.pim_unit_load_semaphore_list[unit_id].wait()

            for unit_id in command.unit_id:
                cur_handler = self.load_helper(command,unit_id)
                SimSession.scheduler.add_coroutine(SimCoroutine(cur_handler))

    def load_helper(self,command:ComputeCommand,target_unit_id)->Callable:
        def load_handler():
            src_addr = command.src_dict[target_unit_id]
            src_chunk_num = command.src_chunk_num_dict[target_unit_id]

            l3_memory_read_port = ChunkMemoryPort()
            l3_memory_read_port.config_chunk_memory(self.external_l3_memory)

            for i in range(src_chunk_num):
                data = l3_memory_read_port.read(src_addr+i,
                                                1,
                                                False,
                                                command.chunk_size,
                                                command.batch_size,
                                                2)

                packet = ChunkPacket(
                    data,
                    command.chunk_size,
                    command.batch_size,
                    2
                )

                self.pim_unit_list[target_unit_id].load_fifo.write(packet)

            # 释放资源
            self.pim_unit_load_semaphore_list[target_unit_id].post()

        return load_handler


    def store_engine(self):
        # 起到 reduce 的作用

        while True:
            if self.store_engine_command_queue.is_empty():
                return

            command:ComputeCommand = self.store_engine_command_queue.read()

            # 申请资源
            for unit_id in command.unit_id:
                self.pim_unit_store_semaphore_list[unit_id].wait()

            handler = self.store_helper(command)
            SimSession.scheduler.add_coroutine(SimCoroutine(handler))

    def store_helper(self,command:ComputeCommand)->Callable:
        def store_handler():
            l3_memory_write_port = ChunkMemoryPort()
            l3_memory_write_port.config_chunk_memory(self.external_l3_memory)

            reduce_fifo_list = [self.pim_unit_list[unit_id].store_fifo for unit_id in command.unit_id]
            dst = command.dst
            for i in range(command.dst_chunk_num):
                for fifo in reduce_fifo_list:
                    packet = fifo.read()

                l3_memory_write_port.write(
                    dst + i,
                    None,
                    True,
                    command.chunk_size,
                    command.batch_size,
                    2
                )

            # 释放资源
            for unit_id in command.unit_id:
                self.pim_unit_store_semaphore_list[unit_id].post()


        return store_handler

    def config_connection(self,l3_memory:ChunkMemory):

        self.external_l3_memory = l3_memory




