from time import sleep
from typing import Optional

from Desim.Core import SimModule
from Desim.memory.Memory import ChunkMemory
from Desim.module.FIFO import FIFO

from EdgeSim.Commands import ComputeCommand, VectorCommand
from EdgeSim.PIMEngine import PIMEngine
from EdgeSim.VectorEngine import VectorEngine


class EdgeChiplet(SimModule):
    def __init__(self):
        super().__init__()


        self.l3_memory = ChunkMemory()

        self.pim_engine = PIMEngine()
        self.vector_engine = VectorEngine()


    def load_commands(self,compute_command_list:list[ComputeCommand],vector_command_list:list[VectorCommand]):
        self.pim_engine.load_command(compute_command_list)
        self.vector_engine.load_commands(vector_command_list)

    def config_connection(self):

        self.pim_engine.config_connection(self.l3_memory)
        self.vector_engine.config_connection(self.l3_memory)







