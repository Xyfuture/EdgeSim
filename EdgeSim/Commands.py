from dataclasses import dataclass, field


@dataclass
class ComputeCommand:
    opcode:str = ''

    batch_size:int = -1

    chunk_size:int = -1

    macro_id:list[int] = field(default_factory=list)

    dst:int = -1
    dst_chunk_num:int = -1

    src_dict:dict[int,int] = field(default_factory=dict)
    src_chunk_num_dict:dict[int,int] = field(default_factory=dict)



@dataclass
class ForwardCommand:
    opcode:str =''

    batch_size:int = -1
    chunk_size:int = -1
    chunk_num:int = -1

    # TODO 数据格式的问题

    memory_dst_flag:bool = False
    memory_dst:int = -1
    fifo_dst_flag:bool = False
    fifo_dst:int = -1

    memory_src_flag:bool = False
    memory_src:int = -1
    fifo_src_flag:bool = False
    fifo_src:int = -1

    reduce:bool = False



