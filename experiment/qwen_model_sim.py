import json
import os
from tempfile import tempdir
from typing import Literal, Optional

from build.lib.Desim.Core import SimSession
from pydantic import  BaseModel

from EdgeSim.Chip import EdgeChiplet
from EdgeSim.Commands import ComputeCommand, VectorCommand
from experiment.code_gen import TaskConfig, HardwareConfig, gen_matrix_command, gen_ffn_cross_command, AttenTaskConfig, \
    gen_attention_command



class ModelConfig(BaseModel):
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 0
    vocab_size: int = 0
    model_type:Literal['llama','gpt','mixtral','phi','mistral','phi3','qwen2'] = 'llama' # use different ffn design

    @property
    def per_head_size(self)->int:
        return self.hidden_size // self.num_attention_heads


class SequenceConfig(BaseModel):
    batch_size: int = 1
    prefill_length: int = 1024
    decoding_length: int = 1024

    @property
    def total_length(self)->int:
        return self.prefill_length + self.decoding_length

def load_predefined_model_config(model_name,path:Optional[str]='experiment/model_cards')->ModelConfig:
    if path is None:
        current_file_path = os.path.abspath(__file__)
        # 获取当前文件所在的目录
        current_directory = os.path.dirname(current_file_path)
        parent_directory = os.path.dirname(current_directory)
    else:
        parent_directory = path
    with open(os.path.join(parent_directory, "model_cards", f"{model_name}.json"), "r") as f:
        json_dict = json.load(f)
        lm_config = ModelConfig(**json_dict)

    return lm_config





class ModelRuner:
    def __init__(self,model_config:ModelConfig,sequence_config:SequenceConfig,hardware_config:HardwareConfig,tp_size:int=1,chunk_size:int=128):
        self.lm_config = model_config
        self.sequence_config = sequence_config
        self.hardware_config = hardware_config

        self.chunk_size=chunk_size
        self.tp_size = tp_size

        # 矩阵形状
        self.qkv_proj_shape:Optional[tuple[int,int]] = None
        self.o_proj_shape:Optional[tuple[int,int]] = None

        self.ffn_up_shape:Optional[tuple[int,int]] = None
        self.ffn_gate_shape:Optional[tuple[int,int]] = None
        self.ffn_down_shape:Optional[tuple[int,int]] = None

        self.base_src_addr = 10000
        self.addr_step = 10000

        self.compute_command_list:list[ComputeCommand] = []
        self.vector_command_list:list[VectorCommand] = []




    def gen_matrix_shape(self):
        # 处理 tp 的情况

        self.qkv_proj_shape = (self.lm_config.hidden_size ,
                                self.lm_config.per_head_size *
                                    (self.lm_config.num_attention_heads + 2*self.lm_config.num_key_value_heads) // self.tp_size )

        self.o_proj_shape = (self.lm_config.hidden_size // self.tp_size,self.lm_config.hidden_size)

        self.ffn_up_shape = (self.lm_config.hidden_size,self.lm_config.intermediate_size // self.tp_size)
        self.ffn_gate_shape = (self.lm_config.hidden_size,self.lm_config.intermediate_size // self.tp_size)

        self.ffn_down_shape = (self.lm_config.intermediate_size // self.tp_size,self.lm_config.hidden_size)

    def gen_layer_code(self):

        tmp_addr = self.base_src_addr

        # 首先是 qkv proj 环节
        qkv_proj_task_config = TaskConfig(
            matrix_size=self.qkv_proj_shape,
            batch_size=self.sequence_config.batch_size,
            chunk_size=self.chunk_size,
            src_addr=tmp_addr,
            dst_addr=tmp_addr+self.addr_step,
        )

        tmp_addr += self.addr_step

        qkv_proj_compute_command_list = gen_matrix_command(self.hardware_config,qkv_proj_task_config)

        # 然后是attention部分,暂时跳过
        # TODO
        atten_src_addr = tmp_addr
        atten_tmp_s_dst_addr = atten_src_addr + self.addr_step
        atten_tmp_softmax_dst_addr = atten_tmp_s_dst_addr + self.addr_step
        atten_dst_addr = atten_tmp_softmax_dst_addr + self.addr_step

        tmp_addr = atten_dst_addr

        atten_task_config = AttenTaskConfig(
            batch_size= self.sequence_config.batch_size,
            chunk_size= self.chunk_size,

            num_q_head= self.lm_config.num_attention_heads,
            num_kv_head= self.lm_config.num_key_value_heads,

            head_size= self.lm_config.per_head_size,
            sequence_length=self.sequence_config.total_length,

            src_addr= atten_src_addr,
            dst_addr = atten_dst_addr,
            temp_s_dst_addr = atten_tmp_s_dst_addr,
            temp_softmax_dst_addr = atten_tmp_softmax_dst_addr

        )

        atten_compute_command_list,atten_vector_command_list = gen_attention_command(
            hardware_config=self.hardware_config,
            atten_task_config=atten_task_config
        )



        # output_projection 部分
        o_proj_task_config = TaskConfig(
            matrix_size=self.o_proj_shape,
            batch_size=self.sequence_config.batch_size,
            chunk_size= self.chunk_size,
            src_addr= tmp_addr,
            dst_addr= tmp_addr+self.addr_step,
        )
        tmp_addr += self.addr_step

        o_proj_compute_command_list = gen_matrix_command(self.hardware_config,o_proj_task_config)


        # ffn 相关指令
        assert self.lm_config.model_type in ['qwen2','llama']

        ffn_base_src_addr = tmp_addr
        ffn_up_dst_addr = ffn_base_src_addr + self.addr_step
        ffn_gate_dst_addr = ffn_up_dst_addr + self.addr_step

        ffn_down_src_addr = ffn_gate_dst_addr + self.addr_step
        ffn_down_dst_addr = ffn_down_src_addr + self.addr_step

        tmp_addr = ffn_down_dst_addr

        up_task_config = TaskConfig(
            matrix_size=self.ffn_up_shape,
            batch_size=self.sequence_config.batch_size,
            chunk_size=self.chunk_size,

            src_addr = ffn_base_src_addr,
            dst_addr = ffn_up_dst_addr,
        )

        gate_task_config = TaskConfig(
            matrix_size=self.ffn_gate_shape,
            batch_size=self.sequence_config.batch_size,
            chunk_size=self.chunk_size,

            src_addr = ffn_base_src_addr,
            dst_addr = ffn_gate_dst_addr
        )

        down_task_config = TaskConfig(
            matrix_size=self.ffn_down_shape,
            batch_size=self.sequence_config.batch_size,
            chunk_size=self.chunk_size,

            src_addr = ffn_down_src_addr,
            dst_addr = ffn_down_dst_addr
        )


        ffn_compute_command_list,ffn_vector_command_list = gen_ffn_cross_command(
            hardware_config=self.hardware_config,
            task_config_a=up_task_config,
            task_config_b=gate_task_config,
            task_config_c=down_task_config,
        )


        self.compute_command_list.extend(qkv_proj_compute_command_list)
        self.compute_command_list.extend(atten_compute_command_list)
        self.compute_command_list.extend(o_proj_compute_command_list)
        self.compute_command_list.extend(ffn_compute_command_list)

        self.vector_command_list.extend(atten_vector_command_list)
        self.vector_command_list.extend(ffn_vector_command_list)

    def get_all_reduce_sync_latency(self)->int:
        # 计算进行 all reduce 的延迟
        # 包括Norm的延迟和残差的延迟

        pass


    def run_sim(self):
        SimSession.reset()
        SimSession.init()


        self.gen_matrix_shape()
        self.gen_layer_code()

        chiplet = EdgeChiplet()
        chiplet.load_commands(self.compute_command_list,self.vector_command_list)

        chiplet.config_connection()

        SimSession.scheduler.run()

        print(f"Simulation finished at {SimSession.sim_time}")
        print(f"L3 Memory: {chiplet.l3_memory.memory_data}")

