from Desim.Core import SimModule
from Desim.module.FIFO import FIFO
from sortedcontainers import SortedList

class BasicUnit(SimModule):
    def __init__(self):
        super().__init__()

        self.fifo_list:list[FIFO] = []

        
    pass 

