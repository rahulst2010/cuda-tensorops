from peft import LoraConfig, TaskType
from peft.utils import get_peft_model
from cuda_llama import CUDALinear

class CUDALoraConfig(LoraConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, task_type=TaskType.CAUSAL_LM)

    def _create_new_module(self, config, adapter_name, target):
        if isinstance(target, CUDALinear):
            in_features, out_features = target.in_features, target.out_features
            new_module = CUDALinear(in_features, out_features, bias=target.bias is not None)
            
            # Copy original weights
            new_module.weight = target.weight
            new_module.bias = target.bias
            
            # Add LoRA parameters
            new_module.lora_A = nn.Parameter(
                torch.empty(config.r, in_features))
            new_module.lora_B = nn.Parameter(
                torch.empty(out_features, config.r))
            new_module.lora_alpha = config.lora_alpha
            
            nn.init.kaiming_uniform_(new_module.lora_A, a=math.sqrt(5))
            nn.init.zeros_(new_module.lora_B)
            
            return new_module
        return super()._create_new_module(config, adapter_name, target)
