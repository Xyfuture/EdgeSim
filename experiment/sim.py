
from Desim.Core import SimSession

from experiment.code_gen import HardwareConfig
from experiment.qwen_model_sim import SequenceConfig
from qwen_model_sim import ModelRuner,load_predefined_model_config




if __name__ == '__main__':

    tp_config = {
        'Qwen2.5-7B-Instruct':1,
        'Qwen2.5-14B-Instruct':2,
        'Qwen2.5-32B-Instruct':4,
    }

    model_name = 'Qwen2.5-7B-Instruct'
    model_config = load_predefined_model_config(model_name,r'D:\code\EdgeSim\experiment')

    sequence_config = SequenceConfig(
        batch_size=1,
        decoding_length=1024,
        prefill_length=1024,
    )

    hardware_config = HardwareConfig(
        num_pim_unit=16,
        pim_unit_id_list= [i for i in range(16)],
        pim_unit_sa_size= (4,128),
    )


    model_runner = ModelRuner(model_config,sequence_config,hardware_config,tp_size=tp_config[model_name],chunk_size=128)

    # SimSession.init()
    layer_latency = model_runner.get_layer_latency()
    model_latency = model_runner.get_model_latency()
    print(f'layer latency: {layer_latency}  model_latency: {model_latency}')
    # SimSession.scheduler.run()