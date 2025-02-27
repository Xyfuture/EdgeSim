from typing import Optional

from Desim.Core import SimModule
from Desim.memory.Memory import ChunkMemory
from Desim.module.FIFO import FIFO

from EdgeSim.Commands import ComputeCommand, ForwardCommand
from EdgeSim.ForwardEngine import ForwardEngine
from EdgeSim.Macro import PIMMacroManager


class EdgeChiplet(SimModule):
    def __init__(self):
        super().__init__()


        self.l3_memory = ChunkMemory()

        self.forward_engine = ForwardEngine()

        self.pim_macro_manager = PIMMacroManager()

        self.external_fifo_dict:Optional[dict[int,FIFO]] = {} # 暂时通过外面进行初始化操作


    def load_commands(self, compute_command_list:list[ComputeCommand], forward_command_list:list[ForwardCommand]):
        self.forward_engine.load_command(forward_command_list)
        self.pim_macro_manager.load_command(compute_command_list)

    def config_connection(self,external_fifo_dict:dict[int,FIFO]):
        self.external_fifo_dict = external_fifo_dict

        self.forward_engine.config_connection(self.l3_memory,self.external_fifo_dict)
        self.pim_macro_manager.config_connection(self.l3_memory)






