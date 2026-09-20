# 默认启动与全工作流验证 · 2026-09-21

16 个已发布的非 streaming 工作流，在三种运行配置下各连续生成两次：**96/96 次成功**。保留默认工作流参数，只改变 seed 和输出文件名；第二次采样节点均未命中缓存，且两次输出文件哈希不同。无需按工作流设置 `--reserve-vram` 或关闭 Triton。

## 环境与计时口径

每个请求使用单张 H100 80GB。测试使用同一节点的两张 H100，各运行一个独立 ComfyUI 进程。普通配置的 16 个工作流分配为两组，每个工作流的两次生成连续在同一进程、同一张卡上完成。

ComfyUI 0.35.0 / `29dcf48dd1e260f172dc055e774edaa1513460b1`，核心源码未改；comfy-kitchen 0.2.33，comfy-aimdo 0.5.3。

| 配置 | PyTorch | 实际运行路径 |
|---|---|---|
| 普通 | 2.7.1+cu126 | 传统加载器；Standard 使用 PyTorch attention，Flash 自动启用可用的 FA3/融合优化；INT8 使用 Triton |
| cu130 DynamicVRAM | 2.11.0+cu130 | DynamicVRAM 实际启用；PyTorch attention；原生 CUDA INT8 |
| FA2 + DynamicVRAM | 2.10.0+cu128 | 真实 FlashAttention 2.8.3.post1 扩展；DynamicVRAM 实际启用；INT8 使用 Triton |

FA2 配置以 `--use-flash-attention` 启动，并用 `LYNNREAL_FA3=0` 关闭可选 FA3 覆盖，以确保测到 FA2。其余两组采用默认启动。三组均保留默认全局显存预留。cu130 环境中的 flash-attn 兼容 shim 未被当作真实 FA2 验证。

下表是第二次的服务端完整执行时间：从 execution_start 到 execution_success，包含采样、解码和视频保存，不含排队。相同提示词与素材的编码缓存可以复用，采样必须重新执行；这是热身后换 seed 的交互场景。首轮包含模型装载、切换与编译，可耗时数分钟，原始首轮和第二轮时间见[机器可读记录](verification/20260921-results.json)。

## 热身时间（秒）

| 工作流 | 普通 cu126 | cu130 DynamicVRAM | FA2 + DynamicVRAM |
|---|---:|---:|---:|
| Standard T2V | 50.03 | 45.63 | 46.42 |
| Standard I2V | 27.22 | 25.74 | 26.65 |
| Standard V2V | 73.33 | 67.68 | 68.13 |
| Standard R2V | 111.65 | 101.97 | 100.92 |
| Standard Pose | 136.51 | 130.53 | 127.33 |
| Standard Lite T2V | 45.71 | 43.65 | 44.44 |
| Standard Lite I2V | 24.77 | 24.09 | 25.02 |
| Standard Lite V2V | 69.65 | 68.30 | 69.95 |
| Standard Lite R2V | 110.15 | 101.80 | 101.75 |
| Standard Lite Pose | 131.91 | 130.42 | 127.55 |
| Flash T2V | 12.31 | 16.70 | 16.98 |
| Flash TI2V | 12.81 | 18.34 | 18.05 |
| Flash Ref2V | 27.23 | 39.79 | 39.66 |
| Flash Lite T2V | 12.07 | 18.56 | 18.39 |
| Flash Lite TI2V | 15.54 | 20.65 | 20.40 |
| Flash Lite Ref2V | 24.76 | 41.19 | 40.27 |

全部主输出为 124 帧 / 24fps，含音频。Standard I2V 为 800×800，Pose 为 1216×672，其余为 1344×768；V2V 还保存拼接续写视频，耗时包含这一步。参考图 `max` 保留官方 2048 像素短边，不以降低参考分辨率换取速度。

普通 T2V 约 50 秒，与原先默认预留配置的水平一致。DynamicVRAM 下不兼容的融合 block 和解码器编译会自动回退到兼容实现，因此 Flash 比普通模式慢；这些回退后的时间已完整计入表格，用户无需手动处理。

## 质量与安全校验

- 108 段基准 MP4 全量解码成功，尺寸、帧数、帧率符合预期；全部双声道音频为有效、非静音的有限值信号，音画时长差小于 0.25 秒。
- 48 个热身主输出均抽取首、中、尾帧目视检查，未见数值损坏导致的黑屏、噪声或分块。鼠标 I2V 提示词明确要求最终淡出，因此允许该示例的暗尾帧，仍严格检查其首帧和中间帧。
- 双次基准前还完成 20 个质量/切换回归用例、22 段输出，包含 BF16 R2V → BF16 Pose 768p、完整 INT8 Pose 768p、Lite Pose 768p、三主体 Flash 参考图，以及回切普通 T2V。
- CUDA 和 Triton 后端各 12 组分块/未分块 INT8 对比全部逐值一致（max_abs=0），覆盖 BF16/FP16、无激活/SwiGLU/GELU、整张量/逐通道缩放；另验证 torch.ops 路径和重复安装。
- 编码器内存估计对照真实 Qwen 图像预处理，覆盖四种尺寸及像素上限；高负载请求之后的普通文本预算复位为 0；缺少可选 INT8 registry 时仍安装内存估计。
- 待发布的 17 个源码、启动器与验证文件和远端实测副本哈希一致；16 个工作流 JSON 与三组基准记录哈希一致。

## 修复内容

`sampling_safety.py` 在公共 INT8 实现分发处约束大矩阵的行数，覆盖 Python 和 torch.ops 两条入口，并保留所选后端。H3 采样内存估计加入参考与文本条件；文本/视觉编码器在加载前按展开后的 Qwen token 和视觉 patch 估计工作空间，避免从已驻留 BF16 模型切换到大参考图时 OOM。预算只作用于当前请求。

`reference_size.py` 恢复官方 `max` 参考图的 2048 像素短边及 32 对齐，包括放大小图。修复位于节点包内；所有已发布工作流都应安装当前节点包。

## 复现

从 ComfyUI 目录用同一命令启动并依次打开工作流：

```bash
python main.py
```

每个工作流先生成一次，随后只修改 seed 再生成，读取第二次 `Prompt executed in ... seconds`。也可使用 `tools/launch_comfyui.sh`，通过环境或解释器选择所需 PyTorch 配置。数值校验命令见[节点包说明](custom_nodes/ComfyUI-LynnReal/README.md)。

上述结论覆盖本表的默认非 streaming 工作流、所列软件栈及单请求独占 H100 80GB；不外推为任意显卡、视频长度、参考数量、同卡并发负载或未来 ComfyUI 版本的保证。
