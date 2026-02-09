# HTML Text Extractor

本项目是对原始工程 `/Volumes/External2T/INF/htmlextractor/html_extractor` 的整理与整合版本，目标是:

- 保留原有正文抽取算法逻辑（V1/V2/V3/V4）。
- 按 Google Python 风格进行模块化、类型标注和注释增强。
- 清理无效与冗余代码/文件，保留可维护的工程结构。
- 提供约 20 条样例与 Gradio 可视化，方便直接查看抽取效果。

## 1. 项目能力

- 正文抽取: 支持结构相似聚合 + 文本密度筛选。
- 标题抽取: 支持 meta/title/h1-h6 组合判定。
- 子标题候选: 返回标题候选与相似度。
- Demo 展示: 可在网页界面查看 URL、HTML、抽取正文、标题结果。

## 2. 目录结构

```text
/Volumes/External2T/self/html-text-extractor
├── app/
│   └── gradio_app.py                 # Gradio 演示入口
├── samples/
│   └── samples.jsonl                 # 20 条样例（每行一个 JSON）
├── scripts/
│   ├── build_samples.py              # 从大数据集构建样例
│   └── validate_project.py           # 一键验证脚本
├── src/
│   └── html_text_extractor/
│       ├── __init__.py
│       ├── extractor.py              # 核心抽取逻辑（V1~V4）
│       ├── simhash_utils.py          # Simhash 计算与冲突检测
│       └── stopwords.py              # 停用词表
├── tests/
│   └── test_extractor.py             # 回归测试（样例规模/稳定性/全量可运行）
├── pyproject.toml
└── README.md
```

## 3. 抽取思路（保持原项目核心逻辑）

### 3.1 预处理

- 解析 HTML，清理无效标签（`script/style/meta/link/img/...`）。
- 去除噪声节点（广告类 class、隐藏元素、注释节点）。
- 进行 DOM 文本重建（`rebuild_tree`），把深层文本合并到上层结构中。

### 3.2 结构编码与聚类

- 为每个文本节点生成结构 key: `tagpath|class|style`。
- 使用 Simhash 计算结构相似度，合并相似节点文本。
- 依据文本长度、链接文本占比（`atext_len`）筛选主体候选块。

### 3.3 主体文本选择

- V4 中引入“交织判断”（`if_interwoven`），识别正文块是否在页面结构上交替出现。
- 只保留符合长度、链接占比、平均行长阈值的结构块作为主正文集合。
- 最终输出去重后的正文（`clean_main_text`）。

### 3.4 标题与子标题识别

- 优先级: `meta` > 与正文相似度最高的 `h1/h2/h3` > `<title>` > 首个标题标签。
- 子标题通过与正文相似度阈值过滤，输出候选列表（含分数）。

## 4. 安装与运行

```bash
cd /Volumes/External2T/self/html-text-extractor
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 5. 启动 Gradio

```bash
cd /Volumes/External2T/self/html-text-extractor
python app/gradio_app.py
```

启动后可:

- 在“样本”页签中切换 20 条样例查看抽取结果。
- 在“自定义”页签中粘贴任意 HTML，查看标题与正文抽取效果。

## 6. 样例格式

`samples/samples.jsonl` 每行格式:

```json
{"url": "https://example.com", "html": "<html>...</html>"}
```

当前保留样例约 20 条，用于快速回归验证。
