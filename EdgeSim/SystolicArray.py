from dataclasses import dataclass

from prompt_toolkit.filters import shift_selection_mode


@dataclass
class SystolicArrayConfig:
    width:int = 128
    height:int = 4



class SystolicArrayOS:
    def __init__(self):

        self.systolic_array_config:SystolicArrayConfig = SystolicArrayConfig()

        self.current_task_width = 0
        self.current_task_height = 0

        self.current_task_len = 0

        self.running = False

        self.total_latency = 0


    def reset(self):
        self.running = False

        self.current_task_len = 0
        self.current_task_height = 0
        self.current_task_width = 0

        self.total_latency = 0


    def set_up(self,task_output_rows:int, task_output_cols:int):
        self.running = True

        assert task_output_rows <= self.systolic_array_config.height
        assert task_output_cols <= self.systolic_array_config.width

        self.current_task_height = task_output_rows # 4x128 的 4
        self.current_task_width = task_output_cols # 4x128 的 128




    def execute(self, input_len:int):
        self.current_task_len += input_len

        # 执行时间 和输入长度是一致的
        self.total_latency += input_len

        return input_len


    def read_out(self):
        # 把尾部的时间更新上
        tail_latency = self.current_task_width + self.current_task_height # 全部运算完

        read_out_latency = tail_latency + self.current_task_height # 最后读取出来


        self.total_latency += read_out_latency

        self.reset()

        return read_out_latency


