from dataclasses import dataclass
from typing import Optional, Callable

from Desim.Core import SimModule, SimSession, SimCoroutine
from Desim.Sync import SimSemaphore
from Desim.memory.Memory import ChunkMemory, ChunkMemoryPort, ChunkPacket
from Desim.module.FIFO import FIFO

from EdgeSim.Commands import ComputeCommand


@dataclass
class PIMEngineConfig:
    num_pim_unit:int = 16



class PIMUnit(SimModule):
    def __init__(self,unit_id:int = -1):
        super().__init__()

        self.load_fifo:FIFO = FIFO(100)
        self.store_fifo:FIFO = FIFO(100)

        self.unit_id = unit_id

        self.quantize_engine_command_queue:FIFO = FIFO(100)
        self.compute_engine_command_queue:FIFO = FIFO(100)
        self.dequantize_engine_command_queue:FIFO = FIFO(100)




    def quantize_engine(self):
        pass

    def compute_engine(self):
        pass

    def dequantize_engine(self):
        pass



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

        for i in range(self.pim_engine_config.num_pim_unit):
            self.pim_unit_list.append(PIMUnit(i))
            self.pim_unit_load_semaphore_list.append(SimSemaphore(0))
            self.pim_unit_store_semaphore_list.append(SimSemaphore(0))

        self.register_coroutine(self.load_engine)
        self.register_coroutine(self.store_engine)

        # self.register_coroutine(self.process)


    # def process(self):
    #     while True:
    #         if self.compute_command_queue.is_empty():
    #             return
    #
    #         current_command:ComputeCommand = self.compute_command_queue.read()
    #
    #         # 构建 load 和 store 的queue
    #
    #         for unit_id in current_command.unit_id:
    #             pim_unit = self.pim_unit_list[unit_id]
    #             # output_fifo = FIFO(current_command.dst_chunk_num)
    #             # output_fifo_list.append(output_fifo)
    #             # pim_macro.issue_compute_command(current_command,output_fifo)
    #
    #         # 配置 store handler
    #         reduce_handler = self.reduce_helper(output_fifo_list,current_command)
    #         SimSession.scheduler.add_coroutine(SimCoroutine(reduce_handler))
    #
    #         pass

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
                                                1)

                packet = ChunkPacket(
                    data,
                    command.chunk_size,
                    command.batch_size,
                    1
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
                    packet.element_bytes
                )

            # 释放资源
            for unit_id in command.unit_id:
                self.pim_unit_store_semaphore_list[unit_id].post()


        return store_handler