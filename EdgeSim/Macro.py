import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from Desim.Core import SimModule, SimSession, SimCoroutine, SimTime
from Desim.memory.Memory import ChunkMemory, ChunkMemoryPort, ChunkPacket
from Desim.module.FIFO import FIFO

from EdgeSim.Commands import ComputeCommand


@dataclass
class PIMMacroConfig:
    sa_number:int = 2
    sa_height:int = 4
    sa_width:int = 64


class PIMMacro(SimModule):
    def __init__(self,macro_id:int = -1):
        super().__init__()


        self.macro_config = PIMMacroConfig()

        self.macro_id = macro_id

        self.l3_memory_read_port = ChunkMemoryPort()
        self.external_l3_memory = ChunkMemory()

        self.load_engine_command_queue = FIFO(10)
        self.compute_engine_command_queue = FIFO(10)
        self.compute_engine_fifo_queue = FIFO(10)

        self.load_to_compute_fifo = FIFO(100)


        self.register_coroutine(self.load_engine)
        self.register_coroutine(self.compute_engine)


        # self.register_coroutine(self.process)

    def process(self):
        pass

    def config_connection(self,external_l3_memory:ChunkMemory):
        self.external_l3_memory = external_l3_memory
        self.l3_memory_read_port.config_chunk_memory(self.external_l3_memory)





    def load_engine(self):
        while True:
            current_command:ComputeCommand = self.load_engine_command_queue.read()

            # 从l3 memory 中读取数据
            src_addr = current_command.src_dict[self.macro_id]
            src_chunk_num = current_command.src_chunk_num_dict[self.macro_id]
            for i in range(src_chunk_num):
                data = self.l3_memory_read_port.read(src_addr,1,False,
                                                     current_command.chunk_size,current_command.batch_size,1)
                packet = ChunkPacket(
                    data,
                    current_command.chunk_size,
                    current_command.batch_size,
                    1
                )

                self.load_to_compute_fifo.write(packet)

                if self.macro_id == 8:
                    print(f"load {i}   time {SimSession.sim_time}")

    def compute_engine(self):
        while True:
            current_command:ComputeCommand = self.compute_engine_command_queue.read()
            output_fifo:FIFO = self.compute_engine_fifo_queue.read()

            packet_buffer = []

            src_chunk_num = current_command.src_chunk_num_dict[self.macro_id]
            # 执行这一条指令
            for out_index in range(current_command.dst_chunk_num):

                for in_index in range(src_chunk_num):
                    if out_index == 0:
                        packet = self.load_to_compute_fifo.read()
                        packet_buffer.append(packet)
                    else:
                        packet = packet_buffer[in_index]
                    
                    # 基础的计算周期数
                    compute_cycle = current_command.chunk_size
                         
                    if in_index == src_chunk_num - 1:
                        # 所有的元素都全部进入
                        assert current_command.batch_size <= self.macro_config.sa_height
                        compute_cycle += self.macro_config.sa_width + current_command.batch_size

                        compute_cycle += 4 # 固定的merge acc 阶段的周期数
                        compute_cycle += current_command.batch_size # 读出数据

                    # if self.macro_id == 0 :
                    #     print(f"output chunk num {out_index}  input chunk num {in_index}  time->{SimSession.sim_time}  cycle->{compute_cycle}")
                    SimModule.wait_time(SimTime(compute_cycle))
                
                # 计算完成一个 Chunk，将结果写入output fifo
                assert current_command.chunk_size == self.macro_config.sa_width * self.macro_config.sa_number
                output_fifo.write(ChunkPacket(
                    None,
                    current_command.chunk_size,
                    current_command.batch_size,
                    4
                )) # 变成int32


    def issue_compute_command(self,command:ComputeCommand,output_fifo:FIFO):
        self.compute_engine_command_queue.write(command)
        self.load_engine_command_queue.write(command)

        self.compute_engine_fifo_queue.write(output_fifo)
                    
                
    @lru_cache(maxsize=10)
    def systolic_array_latency(self,input_size,weight_size):

        sa_rows = self.macro_config.sa_height
        sa_cols = self.macro_config.sa_width

        assert input_size[1] == weight_size[0]
        m,k,n = input_size[0],input_size[1],weight_size[1]


        """
        模拟输出 stationary 脉动阵列执行矩阵乘法所需的周期数

        参数：
          sa_rows, sa_cols: 脉动阵列的行数和列数
          m, k, n: 分别表示 A 的行数、A 与 B 的公共维度、B 的列数
                 即计算 C (m x n) = A (m x k) x B (k x n)

        返回：
          总的运算周期数（假设带宽无限大，数据加载/输出不受限）
        """
        # 按照阵列大小对输出矩阵 C 进行 tiling
        num_tiles_m = math.ceil(m / sa_rows)  # m 方向上 tile 数量
        num_tiles_n = math.ceil(n / sa_cols)  # n 方向上 tile 数量

        total_cycles = 0

        for i in range(num_tiles_m):
            # 若最后一个 tile 不满，则实际行数为 m % sa_rows（当 m 不能被整除时）
            tile_m = sa_rows if (i < num_tiles_m - 1 or m % sa_rows == 0) else m % sa_rows

            for j in range(num_tiles_n):
                tile_n = sa_cols if (j < num_tiles_n - 1 or n % sa_cols == 0) else n % sa_cols

                # 对于一个 tile，其输出各元素累加 k 个乘加操作，
                # 并且由于数据沿行和列方向传递，会产生 (tile_m-1) 和 (tile_n-1) 的额外延迟
                tile_cycles = k + (tile_m - 1) + (tile_n - 1)

                # 可以打印每个 tile 的信息，帮助调试
                # print(f"Tile ({i}, {j}) 大小: {tile_m}x{tile_n}, 所需周期: {tile_cycles}")

                #用于完成 merge acc 阶段的额外cycle
                tile_cycles = tile_cycles + 4  # 目前固定设置为4

                # 用于输出的额外cycle
                tile_cycle = tile_cycles + self.macro_config.sa_height



                total_cycles += tile_cycle



        return total_cycles





class PIMMacroManager(SimModule):
    def __init__(self):
        super().__init__()
        self.external_l3_memory:Optional[ChunkMemory] = None
        self.compute_command_queue:Optional[FIFO] = None
        
        self.pim_macro_list:list[PIMMacro] = []

        # 默认暂时有16个 macro
        for i in range(16):
            self.pim_macro_list.append(PIMMacro(i))




        self.register_coroutine(self.process)


    def process(self):
        # 发射指令,注册reduce handler

        while True:
            if self.compute_command_queue.is_empty():
                return
            current_command:ComputeCommand = self.compute_command_queue.read()
            # 解析并发射
            output_fifo_list = []
            for macro_id in current_command.macro_id:
                pim_macro = self.pim_macro_list[macro_id]
                output_fifo = FIFO(current_command.dst_chunk_num)
                output_fifo_list.append(output_fifo)
                pim_macro.issue_compute_command(current_command,output_fifo)

            # 配置 reduce handler
            reduce_handler = self.reduce_helper(output_fifo_list,current_command)
            SimSession.scheduler.dynamic_add_coroutine(SimCoroutine(reduce_handler))

                



    def reduce_helper(self, fifo_list:list[FIFO], command:ComputeCommand):
        def reduce_handler():
            output_chunk_num = command.dst_chunk_num
            dst = command.dst
            for i in range(output_chunk_num):
                # 进行 reduce 操作
                for fifo in fifo_list:
                    packet:ChunkPacket = fifo.read()

                l3_memory_write_port.write(
                    dst+i,
                    None,
                    True,
                    command.chunk_size,
                    command.batch_size,
                    packet.element_bytes
                )
                    
        l3_memory_write_port = ChunkMemoryPort()
        l3_memory_write_port.config_chunk_memory(self.external_l3_memory)
        assert len(fifo_list) == len(command.macro_id)


        return reduce_handler
    

    def config_connection(self,external_l3_memory:ChunkMemory):
        self.external_l3_memory = external_l3_memory

        for macro in self.pim_macro_list:
            macro.config_connection(external_l3_memory)


    def load_command(self, command_list:list[ComputeCommand]):
        command_size = len(command_list)
        self.compute_command_queue = FIFO(command_size,command_size,command_list)