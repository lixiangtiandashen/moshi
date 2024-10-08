# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""
from pathlib import Path

from safetensors.torch import load_model
import torch

from .compression import MimiModel
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer

from ..modules.transformer import StreamingTransformerLayer
import bitsandbytes as bnb
from torch import nn

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'kyutai/moshiko-pytorch-bf16'


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_mimi(filename: str | Path,
             device: torch.device | str = 'cpu') -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        load_model(model, filename)
    else:
        pkg = torch.load(filename, "cpu")
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(8)
    return model


# def get_moshi_lm(filename: str | Path,
#                  device: torch.device | str = 'cpu') -> LMModel:
#     dtype = torch.bfloat16
#     model = LMModel(
#         device=device,
#         dtype=dtype,
#         **_lm_kwargs,
#     ).to(device=device, dtype=dtype)
#     model.eval()
#     if _is_safetensors(filename):
#         load_model(model, filename)
#     else:
#         pkg = torch.load(
#             filename,
#             "cpu",
#         )
#         model.load_state_dict(pkg["fsdp_best_state"]["model"])
#     return model


class QuantizedStreamingTransformerLayer(StreamingTransformerLayer):
    """量化后的 StreamingTransformerLayer，使用 bitsandbytes 的 Linear8bitLt 进行线性层量化。

    根据 gating 是否启用，分别量化不同的线性层。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = next(self.parameters()).device  # 获取当前设备

        if self.gating is None:
            self._quantize_linear_layers()
        else:
            self._quantize_gating_layers()

    def _quantize_linear_layers(self):
        """量化 linear1 和 linear2 层"""
        for attr in ["linear1", "linear2"]:
            linear = getattr(self, attr)
            if linear is not None:
                setattr(self, attr, self._quantize_layer(linear, attr))

    def _quantize_gating_layers(self):
        """量化 gating 模块中的线性层"""
        if isinstance(self.gating, nn.ModuleList):
            for i, gating_module in enumerate(self.gating):
                self._quantize_single_gating(gating_module, f"ModuleList[{i}]")
        else:
            self._quantize_single_gating(self.gating, "Single")

    def _quantize_single_gating(self, gating_module, module_name):
        """量化单个 gating 模块"""
        for attr in ["linear_in", "linear_out"]:
            if hasattr(gating_module, attr):
                linear = getattr(gating_module, attr)
                setattr(gating_module, attr, self._quantize_layer(linear, f"Gating {module_name}: {attr}"))
            else:
                print(f"Gating {module_name}: 未找到 {attr}，跳过量化。")

    def _quantize_layer(self, linear, layer_name):
        """将给定的线性层量化为 Linear8bitLt"""
        quantized_linear = bnb.nn.Linear8bitLt(
            input_features=linear.in_features,
            output_features=linear.out_features,
            bias=linear.bias is not None,
            has_fp16_weights=False,  # 权重已转换为 float32
            threshold=6.0,
            index=True,
            device=self.device,
        )
        # 转换权重为 float32，因为bitsandbytes不支持bf16
        if linear.weight.dtype == torch.bfloat16:   
            quantized_linear.weight.data = linear.weight.data.to(dtype=torch.float32)
        else:
            quantized_linear.weight.data = linear.weight.data
        if linear.bias is not None:
            if linear.bias.dtype == torch.bfloat16:
                quantized_linear.bias.data = linear.bias.data.to(dtype=torch.float32)
            else:
                quantized_linear.bias.data = linear.bias.data
        # quantized_linear = quantized_linear.to(self.device)
        # print(f"{layer_name} 已成功量化为 Linear8bitLt。")
        
        # 删除原始线性层
        del linear
        torch.cuda.empty_cache()
        
        return quantized_linear


def get_moshi_lm(filename: str | Path, device: torch.device | str = "cpu") -> LMModel:
    dtype = torch.bfloat16

    lm_kwargs = _lm_kwargs.copy()
    lm_kwargs["layer_class"] = QuantizedStreamingTransformerLayer

    model = LMModel(
        device='cpu',
        dtype=dtype,
        **lm_kwargs,
    )

    # 加载模型权重
    if _is_safetensors(filename):
        load_model(model, filename)
    else:
        pkg = torch.load(
            filename,
            "cpu",
        )
        model.load_state_dict(pkg["fsdp_best_state"]["model"])
    
    # 将模型移动到指定设备并设置数据类型，触发量化
    model.to(device=device)
    
    # 设置为评估模式
    model.eval()

    # 清理梯度
    for param in model.parameters():
        param.requires_grad = False

    # 清理 CUDA 缓存
    torch.cuda.empty_cache()
    
    return model

