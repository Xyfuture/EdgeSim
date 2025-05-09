from dataclasses import dataclass
from math import ceil
from turtledemo.nim import computerzug

from EdgeSim.Commands import ComputeCommand, FFNCommand, AttenComputeCommand, VectorCommand, SoftmaxCommand


# FFN 部分的 code gen

@dataclass
class HardwareConfig:
    num_pim_unit:int
    pim_unit_id_list:list
    pim_unit_sa_size:tuple[int,int]



@dataclass
class TaskConfig:
    matrix_size:tuple[int,int]
    batch_size:int
    chunk_size:int

    src_addr:int
    dst_addr:int

    @property
    def matrix_chunk(self)->tuple[int,int]:
        return self.matrix_size[0]//self.chunk_size,self.matrix_size[1]//self.chunk_size


@dataclass
class AttenTaskConfig:
    batch_size:int # 对于 prefill阶段比较有意义
    chunk_size:int

    num_q_head:int
    num_kv_head:int

    head_size:int # head_dim
    sequence_length:int

    src_addr:int
    dst_addr:int
    temp_s_dst_addr:int
    temp_softmax_dst_addr:int



def gen_matrix_command(hardware_config:HardwareConfig,task_config:TaskConfig):

    compute_command_list = []

    assert hardware_config.num_pim_unit % 2 == 0
    assert task_config.batch_size <= hardware_config.pim_unit_sa_size[0]

    num_row_pim_unit = 2
    num_col_pim_unit = hardware_config.num_pim_unit//2

    base_src_addr = task_config.src_addr
    base_dst_addr = task_config.dst_addr

    num_round = ceil( task_config.matrix_size[1] / hardware_config.pim_unit_sa_size[1]  / num_col_pim_unit)


    assert  hardware_config.pim_unit_sa_size[1] % task_config.chunk_size == 0
    num_output_chunk_per_pim_unit = hardware_config.pim_unit_sa_size[1] // task_config.chunk_size
    num_output_chunk_per_round = num_col_pim_unit * num_output_chunk_per_pim_unit

    num_input_chunk_per_pim_unit = ceil(task_config.matrix_chunk[0] // num_row_pim_unit)


    for round_id in range(num_round):
        # TODO 这里应该特殊处理, 矩阵最后 截断的问题, 不过暂时不处理时间上也是正确的.
        for group_id  in range(num_col_pim_unit):
            compute_command_list.append(
                ComputeCommand(
                    opcode = 'Compute',
                    batch_size = task_config.batch_size,
                    chunk_size = task_config.chunk_size,

                    unit_id = [hardware_config.pim_unit_id_list[group_id*num_row_pim_unit + i ]
                               for i in range(num_row_pim_unit)],

                    dst =  base_dst_addr + round_id *num_output_chunk_per_round + group_id * num_output_chunk_per_pim_unit,
                    dst_chunk_num = num_output_chunk_per_pim_unit,

                    src_dict = {
                        hardware_config.pim_unit_id_list[group_id * num_row_pim_unit+i] : base_src_addr + i*num_input_chunk_per_pim_unit
                        for i in range(num_row_pim_unit)
                    },

                    src_chunk_num_dict={
                        hardware_config.pim_unit_id_list[group_id * num_row_pim_unit + i]: num_input_chunk_per_pim_unit
                        for i in range(num_row_pim_unit)
                    }
                )
            )

    return compute_command_list



def gen_ffn_cross_command(hardware_config: HardwareConfig,task_config_a:TaskConfig,task_config_b:TaskConfig,dst_addr:int):
    # 首先将 hardware 进行拆分

    assert hardware_config.num_pim_unit % 2 == 0

    half_num = hardware_config.num_pim_unit//2
    hard_ware_config_a = HardwareConfig(
        num_pim_unit = half_num ,
        pim_unit_id_list =  hardware_config.pim_unit_id_list[:half_num],
        pim_unit_sa_size= hardware_config.pim_unit_sa_size
    )

    hard_ware_config_b = HardwareConfig(
        num_pim_unit = half_num ,
        pim_unit_id_list= hardware_config.pim_unit_id_list[half_num:],
        pim_unit_sa_size= hardware_config.pim_unit_sa_size
    )


    command_a = gen_matrix_command(hard_ware_config_a, task_config_a)
    command_b = gen_matrix_command(hard_ware_config_b, task_config_b)

    assert len(command_a) == len(command_b)


    # 不确定这里是否最后是有用的
    compute_command_list = []


    for i in range(len(command_a)):
        compute_command_list.append(command_a[i])
        compute_command_list.append(command_b[i])


    # 设置vector 相关的指令

    vector_command_list = [
        FFNCommand(
            opcode = 'Vector',
            batch_size=task_config_a.batch_size,
            chunk_size=task_config_a.chunk_size,

            dst = dst_addr,
            dst_chunk_num=task_config_a.matrix_chunk[1],

            src_chunk_num=task_config_a.matrix_chunk[1],
            src_activation=task_config_a.dst_addr,
            src_mul=task_config_b.dst_addr,

            activation=True,
            mul=True


        )
    ]


    return compute_command_list,vector_command_list


def gen_attention_command(hardware_config: HardwareConfig,atten_task_config:AttenTaskConfig):
    # 首先划分 q * k 的部分

    compute_command_list = []
    vector_command_list = []

    assert atten_task_config.head_size % atten_task_config.chunk_size == 0
    num_chunk_per_q_head = atten_task_config.head_size // atten_task_config.chunk_size

    num_chunk_per_s_head = atten_task_config.sequence_length // atten_task_config.chunk_size




    # 直接按照head进行划分
    for head_id in range(atten_task_config.num_q_head):
        pim_unit_id = head_id % hardware_config.num_pim_unit

        qk_command = AttenComputeCommand(
            opcode = 'AttenCompute',
            batch_size = atten_task_config.batch_size,
            chunk_size = atten_task_config.chunk_size,
            unit_id = [pim_unit_id],

            dst =atten_task_config.temp_s_dst_addr + head_id * num_chunk_per_s_head,
            dst_chunk_num = num_chunk_per_s_head,

            src_dict={pim_unit_id:atten_task_config.src_addr + head_id * num_chunk_per_q_head},
            src_chunk_num_dict={pim_unit_id:num_chunk_per_q_head},

            running_head = 1,
            total_head= atten_task_config.num_kv_head,

            head_id= head_id,
        )

        softmax_command = SoftmaxCommand(
            opcode = 'softmax',

            batch_size= atten_task_config.batch_size,
            chunk_size=  atten_task_config.chunk_size,

            chunk_num=num_chunk_per_s_head,

            dst = atten_task_config.temp_softmax_dst_addr + head_id*num_chunk_per_s_head,
            src = atten_task_config.temp_s_dst_addr + head_id*num_chunk_per_s_head,
        )


        sv_command = AttenComputeCommand(
            opcode = 'AttenCompute',
            batch_size= atten_task_config.batch_size,
            chunk_size=  atten_task_config.chunk_size,
            unit_id= [pim_unit_id],

            dst = atten_task_config.dst_addr + head_id*num_chunk_per_q_head,
            dst_chunk_num=  num_chunk_per_q_head,

            src_dict= {pim_unit_id:atten_task_config.temp_softmax_dst_addr + head_id * num_chunk_per_s_head},
            src_chunk_num_dict = {
                pim_unit_id:num_chunk_per_s_head
            },

            running_head = 1,
            total_head = atten_task_config.num_kv_head,
            head_id = head_id,
        )

        compute_command_list.append(qk_command)
        compute_command_list.append(sv_command)

        vector_command_list.append(softmax_command)


    return compute_command_list,vector_command_list

