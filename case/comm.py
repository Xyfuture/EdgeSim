from Desim.Core import SimModule, SimTime, SimSession
from Desim.module.FIFO import FIFO, DelayFIFO

num_elements=20
link_delay = 10



class BasicUnit(SimModule):
    def __init__(self, unit_id:int,num:int):
        super().__init__()
        self.num = num
        self.unit_id = unit_id

        self.in_fifo:dict[int,DelayFIFO] = {} # 接收 外部发送的数据
        # 0 for part data  1 for forward data

        self.out_fifo:dict[int,DelayFIFO] = {} # 向外发送数据
        # 0 for part data 1 for forward data
        
        # 还需要内部的fifo 用于维护 reduce的操作
        self.reduce_fifo = DelayFIFO(1,0)

        self.register_coroutine(self.process_send_part)
        self.register_coroutine(self.process_recv_part)
        self.register_coroutine(self.process_send_forward)
        self.register_coroutine(self.process_recv_forward)



    def process_send_part(self):
        for i in range(self.num):
            fifo = self.out_fifo[0]
            fifo.delay_write(i,SimTime(link_delay))
            SimModule.wait_time(SimTime(1))
        print(f"Unit {self.unit_id} send part finish at {SimSession.sim_time}")

    def process_send_forward(self):
        for i in range(self.num):
            data = self.reduce_fifo.read()
            self.out_fifo[1].delay_write(data,SimTime(link_delay))
            SimModule.wait_time(SimTime(1))
        print(f"Unit {self.unit_id} send forward finish at {SimSession.sim_time}")

    def process_recv_part(self):
        for i in range(self.num):
            data = self.in_fifo[0].read()
            self.reduce_fifo.delay_write(data,SimTime(1))
            SimModule.wait_time(SimTime(1))

        print(f"Unit {self.unit_id} recv part finish at {SimSession.sim_time}")



    def process_recv_forward(self):
        for i in range(self.num):
            data = self.in_fifo[1].read()
            SimModule.wait_time(SimTime(1))
        print(f"Unit {self.unit_id} recv forward finish at {SimSession.sim_time}")



def build_units():

    unit_1 = BasicUnit(1,num_elements)
    unit_2 = BasicUnit(2,num_elements)
    unit_3 = BasicUnit(3,num_elements)
    unit_4 = BasicUnit(4,num_elements)

    fifo_1_2 = DelayFIFO(num_elements)
    fifo_2_1 = DelayFIFO(num_elements)
    unit_1.out_fifo[0] = fifo_1_2
    unit_1.in_fifo[0] = fifo_2_1
    unit_2.in_fifo[0] = fifo_1_2
    unit_2.out_fifo[0] = fifo_2_1

    fifo_1_3 = DelayFIFO(num_elements)
    fifo_3_1 = DelayFIFO(num_elements)
    unit_1.out_fifo[1] = fifo_1_3
    unit_1.in_fifo[1] = fifo_3_1
    unit_3.in_fifo[1] = fifo_1_3
    unit_3.out_fifo[1] = fifo_3_1

    fifo_4_3 = DelayFIFO(num_elements)
    fifo_3_4 = DelayFIFO(num_elements)
    unit_4.out_fifo[0] = fifo_4_3
    unit_4.in_fifo[0] = fifo_3_4
    unit_3.in_fifo[0] = fifo_4_3
    unit_3.out_fifo[0] = fifo_3_4

    fifo_4_2 = DelayFIFO(num_elements)
    fifo_2_4 = DelayFIFO(num_elements)
    unit_4.out_fifo[1] = fifo_4_2
    unit_4.in_fifo[1] = fifo_2_4
    unit_2.in_fifo[1] = fifo_4_2
    unit_2.out_fifo[1] = fifo_2_4



if __name__ == "__main__":
    SimSession.reset()
    SimSession.init()

    build_units()

    SimSession.scheduler.run()

    # 完成时间大约是 2个 latency 的时间 +  传输的时间