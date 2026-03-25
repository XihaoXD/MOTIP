# MOTIP-FFE

MOTIP-FFE是在MOTIP基础上集成了FLD (Feature Learning and Distillation) 模块的多目标跟踪系统。

## 功能介绍

- **FLD模块**：集成了HAT-MASA中的LDA (Linear Discriminant Analysis) 特征转换功能，用于提升ReID特征的判别性。
- **可配置性**：用户可以通过参数控制FLD的开启和关闭，以及调整FLD的相关参数。

## 快速开始

### 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装FLD相关依赖
pip install scikit-learn
```

### 使用方法

#### 基本用法（不使用FLD）

```python
from models.runtime_tracker import RuntimeTracker

# 初始化跟踪器（默认不使用FLD）
tracker = RuntimeTracker(
    model=your_model,
    sequence_hw=(height, width),
    # 其他参数...
)
```

#### 启用FLD

```python
from models.runtime_tracker import RuntimeTracker

# 初始化跟踪器并启用FLD
tracker = RuntimeTracker(
    model=your_model,
    sequence_hw=(height, width),
    # 启用FLD
    use_fld=True,
    # FLD相关参数
    fld_hist_len=60,           # 历史队列长度
    fld_factor_thr=4.0,        # 触发FLD的阈值因子
    fld_use_standard_scaler=False,  # 是否使用标准化
    fld_direct_inter_class_diff=True,  # 是否使用直接类间差异
    fld_use_weighted_class_mean=True,  # 是否使用加权类均值
    # 其他参数...
)
```

## FLD参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_fld` | bool | False | 是否启用FLD模块 |
| `fld_hist_len` | int | 60 | 历史特征队列长度 |
| `fld_factor_thr` | float | 4.0 | 触发FLD的阈值因子，当历史特征数量超过当前跟踪目标数的该倍数时触发 |
| `fld_similarity_alpha` | float | 1.0 | 特征相似度融合权重 |
| `fld_use_standard_scaler` | bool | False | 是否对特征进行标准化 |
| `fld_direct_inter_class_diff` | bool | True | 是否使用直接类间差异计算 |
| `fld_use_weighted_class_mean` | bool | True | 是否使用加权类均值 |
| `fld_transfer_dtype` | torch.dtype | torch.float32 | FLD转换的数据类型 |

## 实现细节

1. **FLD集成**：将HAT-MASA中的LDA特征转换功能集成到MOTIP的`_get_id_pred_labels`方法中。

2. **历史队列**：使用FIFO队列存储每个ID的历史特征，用于LDA模型的训练。

3. **条件触发**：只有当历史特征数量足够多时（超过当前跟踪目标数的`fld_factor_thr`倍），才会触发FLD。

4. **特征转换**：使用训练好的LDA模型对当前特征和轨迹特征进行转换，提升特征的判别性。

5. **兼容性**：FLD功能默认关闭，不会影响MOTIP的原始功能。

### 可学习升维（FLDProjector）与训练损失

- **`FLDProjector`**：`nn.Linear(NUM_ID_VOCABULARY - 1 → FEATURE_DIM)` + `LayerNorm` + `GELU`，将 LDA 输出（固定宽度、不足则填零）映射到与 DETR 一致的维度，供 `trajectory_modeling` / ID Decoder 使用。由配置 **`USE_FLD_PROJECTOR`**（默认 True）在 `build` 时创建。
- **推理**：`TRACKING_MODE=motip_fld` 时，若 checkpoint 含 `fld_projector` 权重，则优先用投影；否则回退为零填充到 256 维。
- **训练**：设置 **`USE_FLD_TRAIN: True`** 时，在每个 clip 上对可见轨迹点拟合 LDA（`no_grad`），再用 `FLDProjector` 替换 `seq_info` 中的轨迹/未知特征，**主监督仍为原有 `id_loss`（交叉熵）**。
- **对齐损失**：**`FLD_ALIGN_LOSS_WEIGHT`** × **`fld_align_loss`**，其中 `fld_align_loss = 1 - mean(cosine(proj_lda, det_feat))`，仅在当前帧 unknown 上统计，用于约束投影后的向量不要偏离原始 DETR 特征方向（与 `id_loss` 配合；LDA 本身无梯度，梯度仅流经 `FLDProjector`）。
- **加载旧权重**：`load_checkpoint` 使用 `strict=False`，便于在旧 checkpoint 上新增 `fld_projector` 后微调。

## 性能提升

FLD模块通过学习类间差异和类内相似性，能够有效提升ReID特征的判别性，从而提高多目标跟踪的准确性，尤其是在目标外观变化较大的场景中。

## 注意事项

- FLD模块会增加一定的计算开销，特别是在历史特征较多时。
- 对于短序列或目标数量较少的场景，FLD可能不会带来明显的性能提升。
- 建议根据具体场景调整FLD的参数，以获得最佳效果。