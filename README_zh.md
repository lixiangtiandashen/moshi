# Moshi：一个用于实时对话的语音-文本基础模型

![precommit badge](https://github.com/kyutai-labs/moshi/workflows/precommit/badge.svg)
![rust ci badge](https://github.com/kyutai-labs/moshi/workflows/Rust%20CI/badge.svg)

[[阅读论文]][moshi] [[演示]](https://moshi.chat) [[Hugging Face]](https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd)

[Moshi][moshi] 是一个语音-文本基础模型和**全双工**语音对话框架。
它使用了[Mimi][moshi]，一个最先进的流式神经音频编解码器。Mimi以24 kHz的频率处理音频，压缩到12.5 Hz的表示形式，带宽为1.1 kbps，完全流式处理（延迟为80ms，帧大小），但性能优于现有的非流式编解码器，如[SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)（50 Hz，4kbps）或[SemantiCodec](https://github.com/haoheliu/SemantiCodec-inference)（50 Hz，1.3kbps）。

Moshi建模了**两条音频流**：一条对应于Moshi，另一条对应于用户。
在推理时，用户的音频流来自音频输入，而Moshi的音频流则从模型的输出中采样。除了这两条音频流，Moshi还预测与其自身语音对应的文本标记，即其**内心独白**，这大大提高了其生成质量。一个小型的深度Transformer模型用于建模给定时间步长的代码簿依赖关系，而一个大型的7B参数时间Transformer模型用于建模时间依赖关系。Moshi实现了理论上的160ms延迟（Mimi的帧大小为80ms + 声学延迟80ms），在L4 GPU上实际整体延迟低至200ms。

[现在与Moshi对话](https://moshi.chat) 在我们的实时演示中。

<p align="center">
<img src="./moshi.png" alt="表示Moshi结构的图表。Moshi建模了两条音频流：一条对应于Moshi，另一条对应于用户。在推理时，用户的音频流来自音频输入，而Moshi的音频流则从模型的输出中采样。除此之外，Moshi还预测与其自身语音对应的文本标记，以提高准确性。一个小型的深度Transformer模型用于建模给定步骤的代码簿依赖关系。"
width="650px"></p>

Mimi基于之前的神经音频编解码器，如[SoundStream](https://arxiv.org/abs/2107.03312)和[EnCodec](https://github.com/facebookresearch/encodec)，在编码器和解码器中添加了Transformer，并调整了步幅以匹配12.5 Hz的整体帧率。这使得Mimi更接近文本标记的平均帧率（约3-4 Hz），并限制了Moshi中的自回归步骤数量。与SpeechTokenizer类似，Mimi使用蒸馏损失，使第一个代码簿标记匹配来自[WavLM](https://arxiv.org/abs/2110.13900)的自监督表示，这使得可以用一个模型建模语义和声学信息。有趣的是，虽然Mimi是完全因果和流式的，但它学会了很好地匹配WavLM的非因果表示，而不会引入任何延迟。最后，与[EBEN](https://arxiv.org/pdf/2210.14090)类似，Mimi仅使用对抗训练损失以及特征匹配，尽管其比特率低，但在主观质量方面显示出显著改进。

<p align="center">
<img src="./mimi.png" alt="表示我们提出的神经编解码器Mimi结构的图表。Mimi在其编码器和解码器中包含Transformer，并实现了更接近文本标记的帧率。这使我们能够减少Moshi中的自回归步骤数量，从而减少模型的延迟。"
width="800px"></p>

## 仓库组织

该仓库中有三个不同版本的moshi推理堆栈。
- 使用PyTorch的Python版本在[`moshi/`](moshi/)目录中。
- 使用MLX的Python版本适用于M系列Mac，在[`moshi_mlx/`](moshi_mlx/)目录中。
- 用于生产的Rust版本在[`rust/`](rust/)目录中。特别包含了Rust实现的Mimi，并提供了Python绑定，名为`rustymimi`。

最后，实时演示的代码在[`client/`](client/)目录中提供。

## 模型

我们发布了三个模型：
- 我们的语音编解码器Mimi，
- 在男性合成语音（Moshiko）上微调的Moshi，
- 在女性合成语音（Moshika）上微调的Moshi。

根据后端的不同，文件格式和量化方式会有所不同。以下是每个模型的HuggingFace仓库列表。Mimi捆绑在每个模型中，并始终使用相同的检查点格式。

- PyTorch版Moshika（bf16）：[kyutai/moshika-pytorch-bf16](https://huggingface.co/kyutai/moshika-pytorch-bf16)。
- PyTorch版Moshiko（bf16）：[kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16)。
- MLX版Moshika（int4, int8, bf16）：[kyutai/moshika-mlx-q4](https://huggingface.co/kyutai/moshika-mlx-q4)，[kyutai/moshika-mlx-q8](https://huggingface.co/kyutai/moshika-mlx-q8)，[kyutai/moshika-mlx-bf16](https://huggingface.co/kyutai/moshika-mlx-bf16)。
- MLX版Moshiko（int4, int8, bf16）：[kyutai/moshiko-mlx-q4](https://huggingface.co/kyutai/moshiko-mlx-q4)，[kyutai/moshiko-mlx-q8](https://huggingface.co/kyutai/moshiko-mlx-q8)，[kyutai/moshiko-mlx-bf16](https://huggingface.co/kyutai/moshiko-mlx-bf16)。
- Rust/Candle版Moshika（int8, bf16）：[kyutai/moshika-candle-q8](https://huggingface.co/kyutai/moshika-candle-q8)，[kyutai/moshika-mlx-bf16](https://huggingface.co/kyutai/moshika-candle-bf16)。
- Rust/Candle版Moshiko（int8, bf16）：[kyutai/moshiko-candle-q8](https://huggingface.co/kyutai/moshiko-candle-q8)，[kyutai/moshiko-mlx-bf16](https://huggingface.co/kyutai/moshiko-candle-bf16)。

所有模型均在CC-BY 4.0许可证下发布。

## 要求

你至少需要Python 3.10，推荐使用3.12。有关具体要求，请查看各个后端目录。你可以使用以下命令安装PyTorch和MLX客户端：

```bash
pip install moshi      # moshi PyTorch, from PyPI
pip install moshi_mlx  # moshi MLX, from PyPI, best with Python 3.12.
# Or the bleeding edge versions for Moshi and Moshi-MLX.
pip install -e "git+https://git@github.com/kyutai-labs/moshi.git#egg=moshi&subdirectory=moshi"
pip install -e "git+https://git@github.com/kyutai-labs/moshi.git#egg=moshi_mlx&subdirectory=moshi_mlx"

pip install rustymimi  # mimi, rust implementation with Python bindings from PyPI
```

如果你没有使用Python 3.12，安装`moshi_mlx`或`rustymimi`（`moshi_mlx`依赖于它）时可能会遇到错误。此时，你需要安装[Rust工具链](https://rustup.rs/)，或者切换到Python 3.12。

虽然我们希望当前的代码库能在Windows上运行，但我们不提供官方支持。我们已经在MacBook Pro M3上测试了MLX版本。目前，我们不支持PyTorch版本的量化，因此你需要一块具有大量内存（24GB）的GPU。

要使用Rust后端，你需要一个最新版本的[Rust工具链](https://rustup.rs/)。要编译GPU支持，你还需要为你的GPU正确安装[CUDA](https://developer.nvidia.com/cuda-toolkit)，特别是`nvcc`。

## Python（PyTorch）

基于PyTorch的API可以在`moshi`目录中找到。它提供了音频标记器（mimi）和语言模型（moshi）的流式版本。

要以交互模式运行，你需要启动一个服务器来运行模型，然后你可以使用Web UI或命令行客户端。

启动服务器：
```bash
python -m moshi.server [--gradio-tunnel] [--hf-repo kyutai/moshika-pytorch-bf16]
```

然后在[localhost:8998](http://localhost:8998)访问Web UI。
如果你的GPU在远程机器上，这将不起作用，因为使用http的网站不允许使用音频工作区API。有两种方法可以解决这个问题：
- 使用ssh的`-L`标志将远程8998端口转发到你的本地主机。然后如前所述连接到[localhost:8998](http://localhost:8998)。
- 使用`--gradio-tunnel`参数，这会设置一个可以从任何地方访问的URL的隧道。请记住，这个隧道通过美国，可能会增加显著的延迟（从欧洲最多500ms）。你可以使用`--gradio-tunnel-token`设置一个固定的秘密令牌，并随着时间的推移重复使用相同的地址。

你可以使用`--hf-repo`选择不同的预训练模型，通过设置适当的Hugging Face仓库。

通过http访问非本地主机的服务器可能会导致在Web UI中使用麦克风的问题（在某些浏览器中，这仅允许使用https）。

本地客户端也可用，如下所示：
```bash
python -m moshi.client [--url URL_TO_GRADIO]
```
但请注意，与Web浏览器不同，此客户端是简陋的：它不执行任何回声消除，也不尝试通过跳过帧来补偿不断增长的延迟。

有关更多信息，特别是如何直接使用API，请查看[moshi/README.md](moshi/README.md)。

## Python（MLX）用于macOS上的本地推理

安装`moshi_mlx`后，你可以运行：
```bash
python -m moshi_mlx.local -q 4   # 权重量化到4位
python -m moshi_mlx.local -q 8   # 权重量化到8位
# 使用不同的预训练模型：
python -m moshi_mlx.local -q 4 --hf-repo kyutai/moshika-mlx-q4
python -m moshi_mlx.local -q 8 --hf-repo kyutai/moshika-mlx-q8
# 注意始终匹配`-q`和`--hf-repo`标志。
```

此命令行界面也是简陋的。它不执行任何回声消除，也不尝试通过跳过帧来补偿不断增长的延迟。

或者你可以运行`python -m moshi_mlx.local_web`来使用Web UI，连接通过http，地址为[localhost:8998](http://localhost:8998)。

## Rust

要运行Rust推理服务器，请在`rust`目录中使用以下命令：
```bash
cargo run --features cuda --bin moshi-backend -r -- --config moshi-backend/config.json standalone
```

在使用macOS时，你可以将`--features cuda`替换为`--features metal`。

或者你可以使用`config-q8.json`而不是`config.json`来使用量化的q8模型。你可以通过更改任一文件中的`"hf_repo"`键来选择不同的预训练模型，例如Moshika。

一旦服务器打印出“standalone worker listening”，你就可以使用Web UI。默认情况下，Rust服务器使用https，因此地址为[localhost:8998](https://localhost:8998)。

你会收到有关站点不安全的警告。在使用Chrome时，你可以通过选择“详细信息”或“高级”，然后“访问此不安全站点”或“继续访问localhost（不安全）”来绕过这些警告。

## 客户端

我们推荐使用Web UI，因为它提供了额外的回声消除，有助于整体模型质量。请注意，大多数命令将直接在提供的URL中提供此UI，通常无需做更多操作。

另外，我们提供了Rust和Python版本的命令行界面，协议与Web UI相同，因此服务器端无需更改。

作为参考，以下是Moshi的客户端列表。

### Rust命令行

在`rust`目录中，运行以下命令：
```bash
cargo run --bin moshi-cli -r -- tui --host localhost
```

### 使用PyTorch的Python

```bash
python -m moshi.client
```

### WebUI

可以通过以下步骤从此仓库构建Web UI（这些步骤需要安装`npm`）。
```bash
cd client
npm install
npm run build
```

然后可以在`client/dist`目录中找到Web UI。

## 开发

如果你希望从此仓库的克隆中安装，也许是为了进一步开发Moshi，你可以执行以下操作：
```bash
# 从仓库的根目录
pip install -e 'moshi[dev]'
pip install -e 'moshi_mlx[dev]'
pre-commit install
```

如果你希望本地构建`rustymimi`（假设你已正确安装Rust）：
```bash
pip install maturin
maturin dev -r -m rust/mimi-pyo3/Cargo.toml
```

## 常见问题

在打开问题之前，请查看[常见问题](FAQ.md)部分。

## 许可证

当前代码的Python部分在MIT许可证下提供，Rust后端在Apache许可证下提供。
Web客户端代码在MIT许可证下提供。
请注意，部分代码基于[AudioCraft](https://github.com/facebookresearch/audiocraft)，并在MIT许可证下发布。

模型的权重在CC-BY 4.0许可证下发布。

## 引用

如果你使用了Mimi或Moshi，请引用以下论文：

```
@techreport{kyutai2024moshi,
    author = {Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and Am\'elie Royer and
			  Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
    title = {Moshi: a speech-text foundation model for real-time dialogue},
    institution = {Kyutai},
    year={2024},
    month={September},
    url={http://kyutai.org/Moshi.pdf},
}
```

[moshi]: https://kyutai.org/Moshi.pdf