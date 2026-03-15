# image-picker

本地摄影筛片 CLI。

它会对图片同时做两类评分：

- `LAION aesthetic predictor`：审美偏好分
- `pyiqa`：技术质量分

然后输出两种最终分数：

- `final_score_batch`
  - 适合当前这一次运行里的排序
- `final_score_global`
  - 适合不同批次之间做粗略横向比较

`final_score` 目前保留，等同于 `final_score_batch`，用于兼容旧结果。

## 默认用法

现在最简单的命令就是只传一个文件夹：

```powershell
uv run score-images "D:\共青约拍\Capture"
```

默认行为：

- 输出文件：`<输入目录>\scores.csv`
- 设备：`cuda`
- `batch-size = 8`
- `max-image-side = 2048`
- 不递归子目录

如果想覆盖默认参数，再额外加选项：

```powershell
uv run score-images "D:\共青约拍\Capture" --batch-size 4 --max-image-side 2048
```

如果图片在子目录里，再加 `--recursive`：

```powershell
uv run score-images "D:\photos" --recursive
```

## 导出分桶结果

打分后按 `top / mid / low` 复制图片：

```powershell
uv run export-buckets --scores-csv "D:\共青约拍\Capture\scores.csv" --destination "D:\共青约拍\Capture\bucketed"
```

## 支持格式

- `JPG`
- `JPEG`
- `PNG`
- `WEBP`
- `TIFF`
- `BMP`
- `RW2`

程序会自动跳过 `.cos/.cot/.cop/.cof` 这类伴生文件。

## 针对大图和 RW2 的建议

如果目录里同时有 `JPG` 和 `RW2`，而且很多图接近 `6000x4000`，更推荐：

```powershell
uv run score-images "D:\共青约拍\Capture" --batch-size 4 --max-image-side 2048
```

原因：

- `RW2` 已支持读取，但 RAW 解码会比 JPG 慢
- `2048` 的缩边能明显减轻显存和吞吐压力
- `3070 8GB` 下，`batch-size 4` 比较稳

如果显存不足，再降到：

```powershell
uv run score-images "D:\共青约拍\Capture" --batch-size 2
```

## 输出列

CSV 里主要会有这些字段：

- `filepath`
- `filename`
- `width`
- `height`
- `aesthetic_score_raw`
- `quality_score_raw`
- `aesthetic_score_norm`
- `quality_score_norm`
- `aesthetic_score_global_norm`
- `quality_score_global_norm`
- `final_score`
- `final_score_batch`
- `final_score_global`
- `bucket`
- `feedback`
- `comment`

如果应用了个人校准器，还会新增：

- `personal_score`

## 分数计算方式

批内分：

```text
final_score_batch = 0.6 * aesthetic_score_norm + 0.4 * quality_score_norm
```

全局分：

```text
final_score_global = 0.6 * aesthetic_score_global_norm + 0.4 * quality_score_global_norm
```

当前默认固定区间：

- 审美分：`1.0 - 10.0`
- 质量分：`0.0 - 1.0`

如果你想改全局分的映射区间，也可以传：

- `--aesthetic-global-min`
- `--aesthetic-global-max`
- `--quality-global-min`
- `--quality-global-max`

## 个人校准器

如果你已经有自己满意的图片集，最推荐的方式不是重训大模型，而是在当前分数上训练一个轻量校准器。

这个校准器会学习：

- 什么样的图是你喜欢的
- 什么样的图虽然基础模型打分不低，但你其实不会选

训练后，它会输出一个新的分数：

- `personal_score`

以后你实际筛片时，优先看 `personal_score`。

### 训练校准器

准备两类图片：

- 正样本：你满意、会保留、会交付、会发出的图
- 负样本：你不满意、会淘汰的图

训练命令：

```powershell
uv run train-calibrator --positive-dir "D:\松下" --negative-dir "D:\26顾村\Capture\low" --output-model ".\artifacts\personal_calibrator.pkl" --output-dataset ".\artifacts\personal_training_dataset.csv" --device cuda --batch-size 4 --max-image-side 2048
```

输出文件：

- 模型：`personal_calibrator.pkl`
- 训练表：`personal_training_dataset.csv`

训练时做了什么：

1. 先对正负样本都跑一遍基础打分
2. 自动提取这些特征：
   - `aesthetic_score_raw`
   - `quality_score_raw`
   - `aesthetic_score_global_norm`
   - `quality_score_global_norm`
   - `final_score_global`
   - `width`
   - `height`
   - `aspect_ratio`
   - `megapixels`
   - `is_raw`
3. 用逻辑回归训练一个轻量二分类器
4. 输出每张图属于“你会喜欢”的概率，也就是 `personal_score`

### 使用校准器

先跑基础打分：

```powershell
uv run score-images "D:\共青约拍\Capture" --batch-size 4
```

再应用校准器：

```powershell
uv run apply-calibrator --scores-csv "D:\共青约拍\Capture\scores.csv" --model ".\artifacts\personal_calibrator.pkl"
```

默认会生成：

```text
D:\共青约拍\Capture\scores.personal.csv
```

如果想指定输出路径：

```powershell
uv run apply-calibrator --scores-csv "D:\共青约拍\Capture\scores.csv" --model ".\artifacts\personal_calibrator.pkl" --output "D:\共青约拍\Capture\scores_personal.csv"
```

### 以后你自己怎么训练

推荐工作流：

1. 先持续收集正样本和负样本
2. 每隔一段时间重训一次校准器
3. 用新的 `personal_calibrator.pkl` 替换旧模型

最实用的样本策略：

- 正样本：你最满意的图
- 负样本：不要只放“明显废片”
- 最好额外加入一些“基础模型打分不低，但你其实不会选”的图

这是最关键的，因为它能真正把模型往你的个人审美拉过去。

建议的数据量：

- 起步：正负各 `100-200`
- 更稳：正负各 `300-1000`
- 如果类别不平衡，也能训，但最好别让负样本太少

推荐你以后维护两个目录：

- `D:\photo_labels\positive`
- `D:\photo_labels\negative`

以后重训只要一条命令：

```powershell
uv run train-calibrator --positive-dir "D:\photo_labels\positive" --negative-dir "D:\photo_labels\negative" --output-model ".\artifacts\personal_calibrator.pkl" --output-dataset ".\artifacts\personal_training_dataset.csv" --device cuda --batch-size 4 --max-image-side 2048
```

### 关于训练结果的理解

如果你看到训练输出里 `ROC-AUC` 很高，比如接近 `1.0`，先不要把它当成“模型已经完美”。

因为当前输出的是训练集上的自测分，它只能说明：

- 这批样本分得开

不代表：

- 对以后所有新照片都一样好用

真正决定效果的，是你负样本里是否包含那些“你不喜欢但并不明显差”的图。

## 首次运行为什么会慢

首次运行通常会更慢，因为会发生这些事：

- 下载或加载 CLIP / IQA 模型权重
- `RW2` 做 RAW 解码
- Windows 上 Hugging Face / torch 缓存初始化

第二次开始通常会快很多。
