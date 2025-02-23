import torch
import ctypes
import torch.nn as nn
from transformers import LlamaForCausalLM

# Load CUDA library
cuda_ops = ctypes.CDLL('./build/libcuda_tensor_ops.so')

# Define CUDA function signatures
cuda_ops.cuda_gemm.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_float, ctypes.c_void_p
]

cuda_ops.cuda_layer_norm.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_float, ctypes.c_void_p
]

class CUDAGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, M, N, K, lora_A=None, lora_B=None, lora_alpha=1.0):
        C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
        stream = torch.cuda.current_stream().cuda_stream
        
        lora_A_ptr = ctypes.POINTER(ctypes.c_float)(0)
        lora_B_ptr = ctypes.POINTER(ctypes.c_float)(0)
        lora_rank = 0
        
        if lora_A is not None and lora_B is not None:
            lora_A_ptr = lora_A.contiguous().data_ptr()
            lora_B_ptr = lora_B.contiguous().data_ptr()
            lora_rank = lora_A.shape[1]

        cuda_ops.cuda_gemm(
            A.contiguous().data_ptr(), B.contiguous().data_ptr(), C.data_ptr(),
            M, N, K,
            lora_A_ptr, lora_B_ptr,
            lora_rank, ctypes.c_float(lora_alpha),
            ctypes.c_void_p(stream)
        )
        return C

class CUDALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.lora_A = None
        self.lora_B = None
        self.lora_alpha = 1.0

    def forward(self, x):
        x_flat = x.view(-1, x.size(-1))
        result = CUDAGemmFunction.apply(
            x_flat, self.weight.T, 
            x_flat.size(0), self.weight.size(0), self.weight.size(1),
            self.lora_A, self.lora_B, self.lora_alpha
        )
        if self.bias is not None:
            result += self.bias
        return result.view(*x.shape[:-1], -1)

class CUDALayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.epsilon = 1e-5

    def forward(self, x):
        output = torch.empty_like(x)
        stream = torch.cuda.current_stream().cuda_stream
        
        cuda_ops.cuda_layer_norm(
            x.contiguous().data_ptr(), output.data_ptr(),
            x.size(0), x.size(1),
            self.weight.data_ptr(), self.bias.data_ptr(),
            ctypes.c_float(self.epsilon),
            ctypes.c_void_p(stream)
        )
        return output

def load_cuda_llama():
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            cuda_linear = CUDALinear(module.in_features, module.out_features, module.bias is not None)
            cuda_linear.weight = module.weight
            cuda_linear.bias = module.bias
            setattr(model, name, cuda_linear)
        elif isinstance(module, nn.LayerNorm):
            cuda_norm = CUDALayerNorm(module.normalized_shape)
            cuda_norm.weight = module.weight
            cuda_norm.bias = module.bias
            setattr(model, name, cuda_norm)
    return model
