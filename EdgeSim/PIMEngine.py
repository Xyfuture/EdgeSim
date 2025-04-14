from dataclasses import dataclass
from typing import Optional

from Desim.Core import SimModule
from Desim.memory.Memory import ChunkMemory, ChunkMemoryPort
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

        self.pim_engine_config = PIMEngineConfig()

        for i in range(self.pim_engine_config.num_pim_unit):
            self.pim_unit_list.append(PIMUnit(i))


        self.register_coroutine(self.process)


    def process(self):
        while True:
            if self.compute_command_queue.is_empty():
                return

            current_command:ComputeCommand = self.compute_command_queue.read()
            # 解析并发射
            output_fifo_list = []
            for unit_id in current_command.unit_id:
                pim_unit = self.pim_unit_list[unit_id]
                # output_fifo = FIFO(current_command.dst_chunk_num)
                # output_fifo_list.append(output_fifo)
                # pim_macro.issue_compute_command(current_command,output_fifo)

            # 配置 store handler
            reduce_handler = self.reduce_helper(output_fifo_list,current_command)
            SimSession.scheduler.add_coroutine(SimCoroutine(reduce_handler))

            pass

    def load_helper(self,command:ComputeCommand):
        pass

    def store_helper(self,command:ComputeCommand):
        # 起到 reduce 的作用
        def store_handler():
            reduce_fifo_list = [self.pim_unit_list[unit_id].store_fifo for unit_id in command.unit_id]

            dst = command.dst
            for i in range(command.dst_chunk_num):
                for fifo in reduce_fifo_list:
                    packet = fifo.read()

                l3_memory_write_port.write(
                    dst + i ,
                    None,
                    True,
                    command.chunk_size,
                    command.batch_size,
                    packet.element_bytes
                )


        l3_memory_write_port = ChunkMemoryPort()
        l3_memory_write_port.config_chunk_memory(self.external_l3_memory)


        return store_handler
