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

## 5. 复现命令（在 7227 上）

```bash
C=/inspire/qb-ilm/project/3d-display/linpeijia-240108120084/ComfyUI
E=/inspire/qb-ilm/project/3d-display/public/conda/envs/lynnreal-comfyui/bin/python

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
