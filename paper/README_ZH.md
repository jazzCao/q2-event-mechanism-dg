# Q2 十次重复方法论文初稿

主入口是 `main.tex`，编译结果为同目录 `main.pdf`。这是独立版本，未覆盖此前 R3t3 稿件或三次重复 Q2 稿件。

作者顺序：Runze Cao；Jie Liu（通讯作者）。单位：School of Reliability and Systems Engineering, Beihang University (BUAA), Beijing, China。

## 本版内容

- 方法：观测锚定的事件机制编辑 Q2；不再沿用旧 R3 的工况分支相减叙述。
- 主实验：PU 四个 3→1 任务、PHM09 二十个 2→1 任务、JNU 三个 2→1 任务，seed 0–9，mean ± sample SD。
- 完整结果：22 个 MDG 配方，以及 PCR 录音级比较、JNU 六个单源方向、半暴露和能量收缩对照。
- 主要统计：固定末步；源验证选模单列。图中配对区间是固定任务面板上的事后描述区间，未当作新的效果门。
- 新的实际计算只有保存指标的汇总和绘图。没有训练、模型加载、原始信号读取或预测。

## 修改位置

- 摘要、引言、相关工作、实验分析、讨论、结论及署名：`main.tex`。
- 公式及机制详述：`generated/method.tex`；数据协议：`generated/protocol.tex`。
- 引用：`references.tex`。
- 整体图：`generated/framework_publication.svg`；机制图：`generated/mechanism_publication.svg`。SVG 保留文字，可用矢量编辑器进一步修改；PDF 是正文使用的矢量输出，PNG 便于预览。
- 数字表格：`generated/*.tex`，由已冻结结果生成。不要为了优化表述改变实验数值。

`build_q2_tenseed_manuscript.py` 会重新生成表格并从旧版已核对公式生成方法/协议文件；手工修改这些生成文件后，不要直接重跑生成器覆盖修改。绘图源是 `draw_q2_publication_figures.py`。流程图和机制图中的波形是解析示意，实际源信号示例使用既有归档图，并有独立图注。

## 编译

在本目录运行两次 `pdflatex -interaction=nonstopmode -halt-on-error main.tex`。本机已使用 `D:/.vscode/texlive/2026/bin/windows/pdflatex.exe` 验证。参考文献以内嵌条目维护，无需 BibTeX。模板文件已随稿保留。

本机应用内自动预览目前误选了缺少 Perl 的 MiKTeX，未能启动它自己的编译；这与稿件编译错误不同。已有的 `main.pdf` 使用上面的 TeX Live 成功生成，可直接打开。无需为阅读 PDF 修改环境或重新训练。

## 投稿前需作者填写的信息

通讯邮箱、基金项目、作者贡献和利益冲突声明尚未由作者提供，未代为编造。声明及是否公开代码的具体安排应由两位作者确认。本稿是完成科学内容和实验整合的初稿，尚未提交期刊。

## 结果追溯

数值来源：`results/wfdg/ress_tenseed_baseline_addon/COMBINED_FULL_TEN_RUN_RESULTS_V2.json`。

SHA-256：`0bd673143afbe0b3faffe08c518fca3a52c3f1d26c323162be7a6e00a4e5b5d8`。

当前三数据集协议与早期 PU 60 个任务的 R3t3 协议不同，不能把两版均值直接混成主表。旧三次重复的校准、最差目标等统计没有冒充十次结果；本版录音错误和重复性使用十次结果重新计算。
