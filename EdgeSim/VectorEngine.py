from dataclasses import dataclass
from typing import Optional

from Desim.Core import SimModule, SimTime
from Desim.memory.Memory import ChunkMemory, ChunkMemoryPort, ChunkPacket
from Desim.module.FIFO import FIFO

from EdgeSim.Commands import FFNCommand, SoftmaxCommand, VectorCommand


@dataclass
class VectorEngineConfig:
    pass


class FFNEngine(SimModule):
    def __init__(self):
        super().__init__()

        self.external_l3_memory:ChunkMemory = None


        self.load_to_activation_fifo = FIFO(100)
        self.load_to_mul_fifo = FIFO(100)
        self.activation_to_mul_fifo = FIFO(100)
        self.mul_to_store_fifo = FIFO(100)


        self.load_activation_command_queue:FIFO[FFNCommand] = FIFO(1)
        self.load_mul_command_queue:FIFO[FFNCommand] = FIFO(1)
        self.store_command_queue:FIFO[FFNCommand] = FIFO(1)
        self.mul_command_queue:FIFO[FFNCommand] = FIFO(1)
        self.activation_command_queue:FIFO[FFNCommand] = FIFO(1)


        self.register_coroutine(self.load_activation_engine)
        self.register_coroutine(self.load_mul_engine)
        self.register_coroutine(self.store_engine)
        self.register_coroutine(self.mul_engine)
        self.register_coroutine(self.activation_engine)


    def issue_command(self, command:FFNCommand):
        assert isinstance(command,FFNCommand)

        # 需要把command的长度设置为1, 这样就能限制每次只读取一条指令了

        self.load_activation_command_queue.write(command)
        self.load_mul_command_queue.write(command)
        self.store_command_queue.write(command)
        self.mul_command_queue.write(command)
        self.activation_command_queue.write(command)


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

            assert command.dst_chunk_num == command.src_chunk_num

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

                    self.mul_to_store_fifo.write(
                        ChunkPacket(
                            None,
                            command.chunk_size,
                            command.batch_size,
                            2
                        )
                    )
                    SimModule.wait_time(SimTime(1))

            else:
                # 直接将 activation 的结果传输到 memory 中
                for i in range(command.src_chunk_num):
                    data = self.activation_to_mul_fifo.read()

                    self.mul_to_store_fifo.write(
                        ChunkPacket(
                            data.payload,
                            command.chunk_size,
                            command.batch_size,
                            2
                        )
                    )

                    SimModule.wait_time(SimTime(1))


    def activation_engine(self):


        while True:
            command = self.activation_command_queue.read()

            # 暂时假设一定会进行 activation 操作
            assert command.activation

            for i in range(command.src_chunk_num):
                data = self.load_to_activation_fifo.read()

                self.activation_to_mul_fifo.write(
                    ChunkPacket(
                        None,
                        command.chunk_size,
                        command.batch_size,
                        2
                    )
                )

                SimModule.wait_time(SimTime(1))

    def config_connection(self,l3_memory:ChunkMemory):
        self.external_l3_memory = l3_memory


class SoftmaxEngine(SimModule):
    def __init__(self):
        super().__init__()

        self.external_l3_memory:Optional[ChunkMemory] = None


        self.load_command_queue:FIFO[SoftmaxCommand] = FIFO(1)
        self.softmax_compute_command_queue:FIFO[SoftmaxCommand] = FIFO(1)
        self.store_command_queue:FIFO[SoftmaxCommand] = FIFO(1)

        self.load_to_compute_fifo = FIFO(100)
        self.compute_to_store_fifo = FIFO(100)


        self.register_coroutine(self.load_engine)
        self.register_coroutine(self.softmax_compute_engine)
        self.register_coroutine(self.store_engine)


    def issue_command(self, command:SoftmaxCommand):

        assert isinstance(command,SoftmaxCommand)

        self.load_command_queue.write(command)
        self.softmax_compute_command_queue.write(command)
        self.store_command_queue.write(command)



    def load_engine(self):
        l3_memory_read_port = ChunkMemoryPort()
        l3_memory_read_port.config_chunk_memory(self.external_l3_memory)

        while True:
            command = self.load_command_queue.read()

            src_addr = command.src
            for i in range(command.chunk_num):
                data = l3_memory_read_port.read(
                    src_addr + i,
                    1,
                    False,
                    command.chunk_size,
                    command.batch_size,
                    2
                )

                SimModule.wait_time(SimTime(1))

                self.load_to_compute_fifo.write(
                    ChunkPacket(
                        None,
                        command.chunk_size,
                        command.batch_size,
                        2
                    )
                )

    def softmax_compute_engine(self):
        while True:

            command = self.softmax_compute_command_queue.read()

            for i in range(command.chunk_num):
                data = self.load_to_compute_fifo.read()

            SimModule.wait_time(SimTime(1))

            for i in range(command.chunk_num):
                self.compute_to_store_fifo.write(
                    ChunkPacket(
                        None,
                        command.chunk_size,
                        command.batch_size,
                        2
                    )
                )

    def store_engine(self):
        l3_memory_write_port = ChunkMemoryPort()
        l3_memory_write_port.config_chunk_memory(self.external_l3_memory)

        while True:

            command = self.store_command_queue.read()

            dst_addr = command.dst
            for i in range(command.chunk_num):
                data = self.compute_to_store_fifo.read()
                l3_memory_write_port.write(
                    dst_addr + i,
                    None,
                    True,
                    command.chunk_size,
                    command.batch_size,
                    2
                )

                SimModule.wait_time(SimTime(1))

    def config_connection(self,l3_memory:ChunkMemory):
        self.external_l3_memory = l3_memory



class VectorEngine(SimModule):

    def __init__(self):
        super().__init__()


        self.command_queue:FIFO[VectorCommand] = FIFO(100)

        self.ffn_engine = FFNEngine()
        self.softmax_engine = SoftmaxEngine()


        self.register_coroutine(self.process)





    def process(self):
        while True:
            if self.command_queue.is_empty():
                return

            current_command = self.command_queue.read()

            if isinstance(current_command,FFNCommand):
                self.ffn_engine.issue_command(current_command)
            elif isinstance(current_command,SoftmaxCommand):
                self.softmax_engine.issue_command(current_command)
            else:
                raise ValueError


    def load_commands(self, command_list:list[VectorCommand]):
        command_size = len(command_list)

        self.command_queue = FIFO(command_size,command_size,command_list)


    def config_connection(self,l3_memory:ChunkMemory):

        self.softmax_engine.config_connection(l3_memory)
        self.ffn_engine.config_connection(l3_memory)


