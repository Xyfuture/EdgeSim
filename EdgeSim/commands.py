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