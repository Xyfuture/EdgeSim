from dataclasses import dataclass, field


@dataclass
class ComputeCommand:
    opcode:str = ''

    batch_size:int = -1

    chunk_size:int = -1

    unit_id:list[int] = field(default_factory=list)

    dst:int = -1
    dst_chunk_num:int = -1

    src_dict:dict[int,int] = field(default_factory=dict)
    src_chunk_num_dict:dict[int,int] = field(default_factory=dict)


@dataclass
class AttenComputeCommand(ComputeCommand):
    opcode:str = 'AttenCompute'

    # 下面这些内容应该是同上的
    # batch size | chunk size | unit id | dst | dst_chunk_num | src_dict | src_chunk_num_dict

    # 下面这些是为了支持 attention 新增的

    running_head:int = -1
    total_head:int = -1
    head_id:int = -1



@dataclass
class RRAMComputeCommand(ComputeCommand):
    batch_size:int = -1
    chunk_size:int = -1

    unit_id:list[int] = field(default=list)

    dst:int = -1
    dst_chunk_num:int = -1

    src_dict:dict[int,int] = field(default_factory=list)
    src_chunk_num_dict:dict[int,int] = field(default_factory=dict)


@dataclass
class DRAMComputeCommand(ComputeCommand):

    batch_size:int = -1
    chunk_size:int = -1
    marco_id:int = -1

    dst:int = -1 #  算出来的中间结果写入到 l3 上
    dst_chunk_num:int = -1

    src:int = -1
    src_chunk_num:int = -1

    gqa_share:int = -1






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


    fifo_src_flag:bool = False
    fifo_src:int = -1

    memory_reduce_src_flag:bool = False
    memory_reduce_src:int = -1
    reduce:bool = False

    memory_add_src_flag:bool = False
    memory_add_src:int = -1
    add:bool = False


    memory_mul_src_flag:bool = False
    memory_mul_src:int = -1
    mul:bool = False # 针对RMSNorm的操作



# @dataclass
# class VectorCommand:
#     opcode:str = ''
#
#     batch_size:int = -1
#     chunk_size:int = -1
#
#     dst:int = -1
#     dst_chunk_num:int = -1
#
#     src_chunk_num:int =-1
#     src_0:int = -1 # 对应 silu 的线
#     src_1:int = -1
#
#     silu:bool = False
#     mul:bool = False


@dataclass
class VectorCommand:
    opcode:str = ''

    # batch_size:int = -1
    # chunk_size:int = -1
    #
    # dst:int = -1
    # dst_chunk_num:int = -1
    #
    # src_chunk_num:int =-1
    # src_0:int = -1 # 对应 silu 的线
    # src_1:int = -1
    #
    # silu:bool = False
    # mul:bool = False



@dataclass
class FFNCommand(VectorCommand):
    # opcode:str = ''

    batch_size:int = -1
    chunk_size:int = -1

    dst:int = -1
    dst_chunk_num:int = -1

    src_chunk_num:int =-1
    src_activation:int = -1 # 对应 silu 操作的地址
    src_mul:int = -1 # 如果是-1 就不启用

    activation:bool = False
    mul:bool = False


@dataclass
class SoftmaxCommand(VectorCommand):
    # opcode:str = ''

    # 最单纯的softmax, 一个src, 一个dst, 但是不太能做重叠

    batch_size:int = -1
    chunk_size:int = -1
    chunk_num:int = -1

    dst:int = -1

    src:int = -1
