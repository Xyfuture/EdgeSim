from typing import Optional, TypeAlias

from Desim.Core import SimModule, SimTime
from Desim.Sync import SimSemaphore
from Desim.memory.Memory import ChunkMemory, ChunkMemoryPort, ChunkPacket
from Desim.module.FIFO import FIFO
from Desim.module.Pipeline import PipeGraph, PipeStage

from EdgeSim.Commands import ForwardCommand


PipeArg:TypeAlias = Optional[dict[str,FIFO]]




class ForwardEngine(SimModule):
    def __init__(self):
        super().__init__()

        # TODO 改变 semaphore 的逻辑，使之支持按照wait的顺序进行post

        self.external_fifo_dict:Optional[dict[int,FIFO]] = None
        self.external_l3_memory:Optional[ChunkMemory] = None

        self.fifo_semaphore_dict:Optional[dict[int,SimSemaphore]] = None

        self.forward_command_queue:Optional[FIFO] = None


        self.register_coroutine(self.process)

    def process(self):
        while True:
            if self.forward_command_queue.is_empty():
                return

            # 首先严格顺序执行吧
            current_command:ForwardCommand = self.forward_command_queue.read()

            # 获取资源
            if current_command.fifo_dst_flag:
                self.fifo_semaphore_dict[current_command.fifo_dst].wait()

            if current_command.fifo_src_flag:
                self.fifo_semaphore_dict[current_command.fifo_src].wait()

            # 获取到了 fifo 的访问权限，开始构建需要的流水线

            pipe_graph = PipeGraph()
            # 直接按照统一的方式构建，然后在stage内部判断具体的执行情况

            memory_read_stage = PipeStage.dynamic_create(
                self.l3_read_dma_helper(current_command)
            )

            fifo_read_stage = PipeStage.dynamic_create(
                self.fifo_read_helper(current_command)
            )

            reduce_stage = PipeStage.dynamic_create(
                self.reduce_helper(current_command)
            )

            memory_write_stage = PipeStage.dynamic_create(
                self.l3_write_dma_helper(current_command)
            )

            fifo_write_stage = PipeStage.dynamic_create(
                self.fifo_write_helper(current_command)
            )


            pipe_graph.add_stage(memory_read_stage,'memory_read_stage')
            pipe_graph.add_stage(fifo_read_stage,'fifo_read_stage')
            pipe_graph.add_stage(reduce_stage,'reduce_stage')
            pipe_graph.add_stage(memory_write_stage,'memory_write_stage')
            pipe_graph.add_stage(fifo_write_stage,'fifo_write_stage')

            pipe_graph.add_edge('memory_read_stage','reduce_stage','memory_to_reduce',10,0)
            pipe_graph.add_edge('fifo_read_stage','reduce_stage','fifo_to_reduce',10,0)
            pipe_graph.add_edge('reduce_stage','memory_write_stage','reduce_to_memory',10,0)
            pipe_graph.add_edge('reduce_stage','fifo_write_stage','reduce_to_fifo',10,0)

            pipe_graph.build_graph()
            pipe_graph.start_pipe_graph()

            SimModule.wait_time(SimTime(1))



    def l3_read_dma_helper(self,command:ForwardCommand):
        def l3_read_dma_handler(input_fifo_map:PipeArg,output_fifo_map:PipeArg)->bool:
            if not command.memory_reduce_src_flag:
                return False

            l3_memory_read_port = ChunkMemoryPort()
            l3_memory_read_port.config_chunk_memory(self.external_l3_memory)


            for i in range(command.chunk_num):
                data = l3_memory_read_port.read(command.memory_reduce_src + i, 1, False, command.chunk_size, command.batch_size, 4)

                for fifo_name, fifo in output_fifo_map.items():
                    fifo.write(ChunkPacket(
                        data,
                        command.chunk_size,
                        command.batch_size,
                        4
                    ))

                SimModule.wait_time(SimTime(1))

            return False

        return l3_read_dma_handler


    def l3_write_dma_helper(self,command:ForwardCommand):
        def l3_write_dma_handler(input_fifo_map:PipeArg,output_fifo_map:PipeArg)->bool:
            if not command.memory_dst_flag:
                return False

            l3_memory_write_port = ChunkMemoryPort()
            l3_memory_write_port.config_chunk_memory(self.external_l3_memory)

            input_fifo = input_fifo_map['reduce_to_memory']
            for i in range(command.chunk_num):
                chunk_packet = input_fifo.read()
                # 直接写入 量化之后的
                l3_memory_write_port.write(command.memory_dst+i,chunk_packet.payload,True,chunk_packet.num_elements,chunk_packet.batch_size,1)

                SimModule.wait_time(SimTime(1))
            return False


        return l3_write_dma_handler

    def fifo_read_helper(self,command:ForwardCommand):
        def fifo_read_handler(input_fifo_map:PipeArg,output_fifo_map:PipeArg)->bool:
            if not command.fifo_src_flag:
                return False

            assert command.fifo_src in [3,4]

            for i in range(command.chunk_num):
                chunk_packet = self.external_fifo_dict[command.fifo_src].read()

                for fifo_name, fifo in output_fifo_map.items():
                    fifo.write(chunk_packet)

                SimModule.wait_time(SimTime(1))

            self.fifo_semaphore_dict[command.fifo_src].post()

            return False

        return fifo_read_handler


    def fifo_write_helper(self,command:ForwardCommand):
        def fifo_write_handler(input_fifo_map:PipeArg,output_fifo_map:PipeArg)->bool:
            if not command.fifo_dst_flag:
                return False

            assert command.fifo_dst in [1,2 ]

            for i in range(command.chunk_num):
                chunk_packet = input_fifo_map['reduce_to_fifo'].read()
                self.external_fifo_dict[command.fifo_dst].write(chunk_packet)
                SimModule.wait_time(SimTime(1))

            self.fifo_semaphore_dict[command.fifo_dst].post()
            return False

        return fifo_write_handler

    def reduce_helper(self,command:ForwardCommand):
        def reduce_handler(input_fifo_map:PipeArg,output_fifo_map:PipeArg)->bool:
            if command.reduce:
                # 必然是两个 read 过来
                assert command.memory_reduce_src_flag and command.fifo_src_flag

                for i in range(command.chunk_num):
                    chunk_packet_1:ChunkPacket = input_fifo_map['memory_to_reduce'].read()
                    chunk_packet_2:ChunkPacket = input_fifo_map['fifo_to_reduce'].read()

                    assert chunk_packet_1.num_elements == chunk_packet_2.num_elements

                    output_chunk_packet = ChunkPacket(
                        chunk_packet_1.payload,
                        chunk_packet_1.num_elements,
                        chunk_packet_1.batch_size,
                        chunk_packet_1.num_elements
                    )

                    if command.memory_dst_flag:
                        output_fifo_map['reduce_to_memory'].write(output_chunk_packet)

                    if command.fifo_dst_flag:
                        output_fifo_map['reduce_to_fifo'].write(output_chunk_packet)
            else:
                # 需要进行 bypass 操作
                assert command.fifo_src_flag ^ command.memory_reduce_src_flag

                if command.memory_reduce_src_flag:
                    input_fifo = input_fifo_map['memory_to_reduce']
                if command.fifo_src_flag:
                    input_fifo = input_fifo_map['fifo_to_reduce']

                for i in range(command.chunk_num):
                    chunk_packet = input_fifo.read()

                    if command.memory_dst_flag:
                        output_fifo_map['reduce_to_memory'].write(chunk_packet)
                    if command.fifo_dst_flag:
                        output_fifo_map['reduce_to_fifo'].write(chunk_packet)

            return False

        return reduce_handler


    def config_connection(self,l3_memory:ChunkMemory,fifo_dict:dict[int,FIFO]):
        self.external_fifo_dict = fifo_dict
        self.external_l3_memory = l3_memory

        self.fifo_semaphore_dict = {}
        for key in self.external_fifo_dict.keys():
            self.fifo_semaphore_dict[key] = SimSemaphore(1)



    def load_command(self,command_list:list[ForwardCommand]):
        command_size = len(command_list)
        self.forward_command_queue = FIFO(command_size,command_size,command_list)


