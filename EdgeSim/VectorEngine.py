from dataclasses import dataclass

from Desim.Core import SimModule, SimTime
from Desim.memory.Memory import ChunkMemory, ChunkMemoryPort, ChunkPacket
from Desim.module.FIFO import FIFO

from EdgeSim.Commands import FFNCommand


@dataclass
class VectorEngineConfig:
    pass


class FNNEngine(SimModule):
    def __init__(self):
        super().__init__()

        self.external_l3_memory:ChunkMemory = None


        self.load_to_activation_fifo = FIFO(100)
        self.load_to_mul_fifo = FIFO(100)
        self.activation_to_mul_fifo = FIFO(100)
        self.mul_to_store_fifo = FIFO(100)


        self.load_activation_command_queue:FIFO[FFNCommand] = FIFO(100)
        self.load_mul_command_queue:FIFO[FFNCommand] = FIFO(100)
        self.store_command_queue:FIFO[FFNCommand] = FIFO(100)
        self.mul_command_queue:FIFO[FFNCommand] = FIFO(100)
        self.load_activation_command_queue:FIFO[FFNCommand] = FIFO(100)


        pass




    def load_activation_engine(self):
        l3_memory_read_port = ChunkMemoryPort()
        l3_memory_read_port.config_chunk_memory(self.external_l3_memory)
        while True:
            command = self.load_activation_command_queue.read()

            assert command.activation and command.src_activation != -1

            src_addr = command.src_activation
            for i in range(command.src_chunk_num):
                data = l3_memory_read_port.read(
                    src_addr + i,
                    1,
                    False,
                    command.chunk_size,
                    command.batch_size
                )

                self.load_to_activation_fifo.write(
                    ChunkPacket(
                        data,
                        command.chunk_size,
                        command.batch_size,
                        2
                    )
                )

                SimModule.wait_time(SimTime(1))


    def load_mul_engine(self):
        l3_memory_read_port = ChunkMemoryPort()
        l3_memory_read_port.config_chunk_memory(self.external_l3_memory)
        while True:
            command = self.load_mul_command_queue.read()

            if not command.mul:
                continue

            src_addr = command.src_mul
            for i in range(command.src_chunk_num):
                data = l3_memory_read_port.read(
                    src_addr + i,
                    1,
                    False,
                    command.chunk_size,
                    command.batch_size
                )

                self.load_to_mul_fifo.write(
                    ChunkPacket(
                        data,
                        command.chunk_size,
                        command.batch_size,
                        2
                    )
                )

                SimModule.wait_time(SimTime(1))

    def store_engine(self):
        l3_memory_write_port  = ChunkMemoryPort()
        l3_memory_write_port.config_chunk_memory(self.external_l3_memory)

        while True:
            command = self.store_command_queue.read()

            dst_addr = command.dst
            for i in range(command.dst_chunk_num):
                data:ChunkPacket = self.mul_to_store_fifo.read()

                l3_memory_write_port.write(
                    dst_addr + i,
                    data.payload,
                    True,
                    command.chunk_size,
                    command.batch_size,
                    2
                )

                SimModule.wait_time(SimTime(1))

            # TODO 释放资源?
    def mul_engine(self):

        while True:
            command = self.mul_command_queue.read()

            if command.mul:
                for i in range(command.src_chunk_num):
                    data_1 = self.activation_to_mul_fifo.read()
                    data_2 = self.load_to_mul_fifo.read()

                    # TODO 乘法
            else:
                # 直接将 activation 的结果传输到 memory 中
                pass

    def activation_engine(self):
        while True:
            command = self.activation_to_mul_fifo.read()

            assert command.activation

            pass









class VectorEngine(SimModule):

    def __init__(self):
        super().__init__()


        self.command_queue = FIFO(100)


        pass








