# ComfyUI-LynnReal 节点包 · 验证记录（2026-09-15）

把原先写在 `comfy/` 里的三处改动（Light VAE 的 26 层构造、272/16 tile 几何、
`compile(decoder.forward)`）搬进节点包，改成 **`LynnRealH3VAELoader` 节点**，
并在 H100 上做了静态 + 端到端验证。

环境：`test-lynnreal1` H100 80GB ×4（本实验用 GPU 0/1），ComfyUI 0.35.0
（`29dcf48d`，`git diff` 为空 = 核心未改），`conda/envs/lynnreal-comfyui`
（torch 2.7.1+cu126），1344×768 / 124 帧 / 3 步 / seed 261662374822964。

## 1. 交付物

| 位置 | 内容 |
|---|---|
| `ComfyUI/custom_nodes/ComfyUI-LynnReal/` | 节点包：`flash_compression.py`、`light_vae.py`、`aligned_reference.py`、`backends.py` |
| `.../user/default/workflows/video_minimax_h3_t2v_flash_int8_3step.json` | flash t2v 工作流，节点 119 已换成 `LynnRealH3VAELoader` |
| `.../lynnreal_release/weight/comfyui/` | 同一份工作流 + 节点包副本（发布用） |
| `.../lynnreal_release/.../custom_nodes_backup/lynnreal_core_patch_20260915.patch` | 原核心补丁存档（含 `LYNNREAL_GPU_TIMING` 计时钩子），`git stash` 里也留了一份 |

## 2. 静态验证（CPU 节点 7223，无需 GPU）

`tools/verify_node_pack.py`：

- 按 ComfyUI 的目录加载方式（`spec_from_file_location` + `comfy_entrypoint()`）加载节点包，
  注册出 4 个节点：`LynnRealFlashTokenCompression` / `LynnRealH3VAELoader` /
  `LynnRealAlignedReference` / `LynnRealInt8Backend`；
- `LynnRealH3VAELoader` 的 widget 顺序 = `vae_name, num_layers, tile_size, tile_overlap,
  compile_decoder`，且没有 `control_after_generate` 占位，与工作流 JSON 的
  `widgets_values` 完全对齐；
- Light VAE（`lynnreal_omni_light_vae_fp16`）→ 检测 26 层、解码器 26 块、tile 272/16，
  **442/442 keys 全部落位**（无 missing、无 unexpected）；
- 官方 VAE（`minimax_h3_video_vae_fp16`）→ 仍是 36 块 + 256/64，562/562 keys 落位；
- 反面复现：同一份 Light VAE 走 stock `comfy.sd.VAE()` → 36 块 + 256/64，
  120 个模型权重没有 checkpoint 对应（10 个 transformer 块随机初始化），
  ComfyUI 只打印 `Missing VAE keys [...]` 就继续解码。

## 3. GPU 端到端（H100，节点包 + 未改动的 ComfyUI 核心）

日志证据（`output/e2e_*/server-*.log`）：

```
LynnReal: enabled the comfy-kitchen Triton backend for INT8 models...
Import times for custom nodes: ... custom_nodes/ComfyUI-LynnReal
LynnReal: H3 video VAE loaded -- 26 decoder blocks (checkpoint 26), tile 272/16, decoder compiled.
Prompt executed in 70.0 seconds        # 冷启动，含 40GB DiT + 15GB 文本编码器装载
```

- 启动命令里**没有** `--enable-triton-backend`（用 `COMFYUI_TRITON=0` 起的服务），
  triton 后端由节点包自己打开；
- 采样命中缓存、只重跑解码 + 封装时：**5.0s**（与之前记录的解码 3.38s + 封装 2.62s 同量级，
  不是官方 VAE 几何的 ~8s）；
- 产物是正常的城市场景（人跃过楼顶、天际线、飞行器），与官方脚本同题材，不是灰帧/噪点。

## 4. 与旧核心补丁的数值一致性

同 seed、同 prompt，两两逐帧比较（`tools/compare_videos.py`）：

| 对比 | 说明 | PSNR |
|---|---|---|
| pack_cold vs pack_warm | **同一份代码**、同一服务（采样缓存命中） | 44.46 dB |
| pack_cold vs core_cold | 节点包 vs 旧核心补丁（编译解码器） | 44.62 dB |
| a_pack vs pack_cold | 同一份代码、两次冷启动 | 43.98 dB |
| pack_eager vs core_eager | 两侧都关掉 decoder 编译 | 44.42 dB |

结论：**跨实现差异（44.62 dB）落在"同一份代码重跑"的固有抖动（43.98–44.46 dB）之内**，
即无法归因于节点包。差异结构也一致：0.7–1.4% 的像素超过 8/255，集中在高频细节/前景
（背景 mean|d| ≈ 1.2，差异像素 ≈ 8），没有分块接缝、没有位移，部分帧逐位相同。
来源是 cuDNN/triton 的运行时 kernel 选择，与本次改动无关（关掉编译后依然存在）。

## 5. 复现命令

```bash
# 指向你的 ComfyUI 目录与它的 python
C=<ComfyUI>
E=<ComfyUI 的 python>

# 静态检查
cd $C && $E tools/verify_node_pack.py

# 端到端 A/B（节点包 vs 旧核心补丁；A 用 GPU0，B 用 GPU1）
cd $C && setsid nohup bash tools/e2e_light_vae_ab.sh mytag > /tmp/e2e.log 2>&1 &

# 控制实验：同码重跑 / 采样缓存后重跑解码 / 双侧关编译
cd $C && setsid nohup bash tools/e2e_light_vae_control.sh ctl  > /tmp/ctl.log  2>&1 &
cd $C && setsid nohup bash tools/e2e_light_vae_eager.sh eager  > /tmp/eager.log 2>&1 &

# 逐帧比较 / 差异定位
$E tools/compare_videos.py a.mp4 b.mp4
$E tools/frame_diff.py a.mp4 b.mp4 --frame 3
```

## 6. 遗留事项

1. **同 seed 输出不可比**：官方脚本与 ComfyUI 的噪声源不同；即使同在 ComfyUI，
   编译解码器也会带来上面量级的运行间抖动。要严格比数值，只能比单步 / 单个解码器。
2. **速度**：ComfyUI 只有 FA2 通道，DiT 每步 3.49s（官方 FA3 1.74s / FA2 2.70s），
   同口径慢 29%，来源是 int8 GEMM 与 token 压缩的主机侧索引，与本次改动无关。
3. 原核心补丁（含 GPU 计时钩子）已从 `comfy/` 撤出，保存在 stash 与
   `custom_nodes_backup/lynnreal_core_patch_20260915.patch`；需要复现旧计时表时
   `git apply` 即可。
4. **DynamicVRAM 上的 decoder 编译**：见 §8。已决定暂缓——收益 5s 片子约 1.7s/条（15s 约 5s/条），
   而编译一次性 ≈26s（空缓存）/≈5s（命中缓存），且实现要求 aimdo 的惰性权重先落地，存在换页后
   固化成旧权重的正确性风险。DynamicVRAM 栈上更大的差距在注意力后端（SDPA vs FA3 单这一项
   3.6s），优先级排在前面。

## 7. Flash 在 DynamicVRAM 上的 `aimdo memory compile error`（2026-09-18 修复）

现象：在自动开启 DynamicVRAM 的机器上（torch ≥ 2.8 + cu13x，`main.py` 自己就会打开），
Flash 工作流在第一个采样步崩：

```
[INFO] LynnReal: fused DiT blocks running (first block has 64 rows, 1 segments, table (1, 2688)).
[INFO] LynnReal: fused DiT blocks verified on this GPU (relative error 0.0000, max abs 0.0000).
[INFO] Comfy model compiler graph breaks: 0, rogues: 12
[ERROR] !!! Exception during processing !!! aimdo memory compile error
```

原因：comfy-aimdo 会为每个 DiT block 记录 **malloc graph**，而融合 block 的两个 Triton
kernel 的分配模式它不接受（`--disable-comfy-compiler` 或没有 aimdo 的栈都正常，把 block 包在
`pause_malloc_graph()` 里也不行）。包里本来就有"malloc graph 活跃时自动关闭融合"的守卫，
但 2026-09-17 把它从 `aimdo_enabled` 改成 `malloc_graph_enabled(device)` 时漏传了 device：
`fast_blocks.probe(blocks[0])` 走默认参数，`is_device_cuda(None)` 恒为 False，守卫从未生效。
（同一轮改动里的 RoPE 守卫用的是全局 `aimdo_enabled`，所以一直正常。）

修复：`fast_blocks.probe()` 在查询前取 `comfy.model_management.get_torch_device()`；判定命中时
按设计关闭融合 block 并打日志，Flash 图回落到 ComfyUI 的 block 数学。

验证（7228，同一张 H100，同一工作流 `t2v_lynnreal_flash_3_step.json`，5s/1344×768）：

| 栈 | 修复前 | 修复后 |
|---|---|---|
| cu13x + DynamicVRAM（自动开启 aimdo） | `rogues: 12` → `aimdo memory compile error` | 日志 `fused DiT blocks disabled (...)`，`Prompt executed in 118.03s`，正常出片 |
| cu126（无 aimdo） | 正常（FA3 + 融合） | 不变：融合仍然启用，探测 `relative error 0.0000` |

`tools/verify_node_pack.py` 增加了第 4 项检查：用哨兵 block 驱动 `probe()`，守卫必须在接触
block 之前就退出（旧代码被判 FAIL，修复后通过），这样同类回归在无 GPU 的机器上也能拦住。

附注：同一份日志里 `FlashAttention 3 disabled (probe: ... 32 argument(s))` 是无害回退——FA3
只在 Hopper（sm_90a）上有 kernel，RTX 5090 是 sm_120，本就走不了 FA3；FA2 里有 sm_120。

## 8. 待办：DynamicVRAM 下的 decoder 编译（2026-09-18 记录 · 暂缓）

**现象**：DynamicVRAM（comfy-aimdo）开启的机器上，Light VAE 的 decoder 编译总是失败并静默退回
eager，日志只有一条 warning：

```text
light_vae.py: RuntimeWarning: Decoder compilation unavailable; using the eager decoder:
Unsupported method call / Dynamo does not know how to trace method `__mul__` of class
`PyCSimpleType`  (Developer debug context: ctypes.c_uint.__mul__ [ConstantVariable(int: 2)])
```

影响：解码 3.49s（eager）对 1.78s（编译）；5s 片子 end-to-end 13.6s 对 11.9s（DiT 不变）。

**根因**：`comfy/utils.py:164` 在 `comfy.memory_management.aimdo_enabled` 为真时走 comfy-aimdo 的
ctypes mmap 读取器（`load_safetensors`），返回惰性 tensor；第一次解码才真正读盘，这段读取落在
`torch.compile(fullgraph=True)` 追踪的图里，Dynamo 不支持 ctypes 调用 → 编译失败 →
`_recoverable_compile_failure()` 判定可恢复 → 换回 eager。cu126/cu128 栈 aimdo 是关的，tensor 在
编译前已经落地，所以一直正常。

**当前规则（修复前）**：aimdo 关 → 能编译；aimdo 开 → 必定退 eager。

| 启动条件 | aimdo | decoder 编译 |
|---|---|---|
| torch < 2.8 或 CUDA < 13（cu126 / cu128 环境） | 不启用 | ✅（cu126 实测 `decoder compiled`，解码 1.78s） |
| cu13x + `--disable-dynamic-vram`（或 `--highvram`/`--novram`/`--gpu-only`/`--cpu`） | 关 | ✅（dyn13 实测 `decoder compiled`，无 eager warning） |
| cu13x 默认（含 5090 的 torch 2.13+cu312） | 开 | ❌ eager |
| cu13x + `--disable-comfy-compiler` | 仍开 | ❌（该 flag 只关 DiT 的 malloc graph，不改变权重加载分支） |

**实测（7228，H100，flash t2v 5s/1344×768，进程重启后第一次提交）**：

| 状态 | 客户端总耗时 | generate | video_decode |
|---|---:|---:|---:|
| 冷进程 + 空 Inductor 缓存 | 91.0s | 38.3s | 28.1s（编译 ≈26s + 解码 1.8s） |
| 冷进程 + 磁盘缓存命中 | 75.2s | 16.3s | 6.9s（编译 ≈5s + 解码 1.8s） |
| 同进程第二次提交 | 15.0s | 8.0s | 1.78s |

冷启动的大头是权重加载 ≈59s（75.2 − 16.3），与编译无关；（此前记录的 248.5s 是异常值：进程反复
重启 + 共享盘首读冷 + 空缓存叠加。）

**为什么暂缓（2026-09-18 决定）**：收益 5s 片子 1.7s/条、15s 约 5s/条；编译一次性 ≈26s（空缓存）
/ ≈5s（缓存命中），回本点 15 条（空缓存）/ 3 条（命中）——只有会话里连点多条或跑长片才划算；而实现
必须让 aimdo 管理的惰性权重先落地，一旦 aimdo 在会话中换页，编译图可能固化成旧权重（正确性风险，
比慢更糟）。DynamicVRAM 栈上更大的差距在注意力后端（SDPA vs FA3，5s 单这一项 3.6s），
所以先做便宜且低风险的两件事：注意力后端、eager 解码本身的调优。

**若以后要做，方案**：

1. 在 VAE 加载阶段（`LynnRealH3VAELoader` 内）先把 decoder 权重 materialize（或跑一次极小 latent 的
   预热解码），再 `torch.compile`；保留现有可恢复失败回退。首条片子也能吃上编译速度，用户看到的是
   "加载 VAE"。
2. 或：首条 eager 正常出片，跑完后在后台编译，第二条起变快（首条永不加速）。

验证要求：同一 latent 的 eager vs 编译逐帧对比（`tools/compare_videos.py`、`tools/frame_diff.py`），
外加人为显存压力后重测；对照脚本 `tools/probe_decoder_compile.py`（工作区）。
