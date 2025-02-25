from dataclasses import dataclass
from Desim.Core import SimModule
from Desim.memory.Memory import ChunkMemory,ChunkMemoryPort

@dataclass
class PIMMacroConfig:
    sa_number:int = 2
    sa_height:int = 4
    sa_width:int = 64


class PIMMacro(SimModule):
    def __init__(self):
        super().__init__()
        
        self.l3_memory_read_port = ChunkMemoryPort()
        
        self.register_coroutine(self.process)

    def process(self):
        pass



    def l3_memory_read_dma_handler(self):
        pass 

    def gemm_handler(self):
        pass

    def l3_memory_write_dma_handler(self):
        pass
    

    def systolic_array_latency(self,input_size,weight_size):
        pass