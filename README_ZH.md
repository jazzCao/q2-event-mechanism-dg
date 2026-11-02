# 审稿人入口

这里是最新 Q2 十次重复论文的独立审阅包。先打开 [论文 PDF](paper/main.pdf)，再看 [方法说明](docs/METHOD.md) 和 [实验说明](docs/EXPERIMENTS.md)。不要从旧 R3/CNSR 脚本开始运行。

无需 GPU 或数据集即可复核结果：在本目录依次运行 `python tools/check_integrity.py`、`python tools/reproduce_results.py`。第二个命令读取保存的逐任务、逐 seed 指标和混淆矩阵，重新计算统计量；输出在 `outputs/statistics/`。

`q2/` 是精简后的可读科学代码；`original_code/INDEX.md` 指向与冻结实验锁一致的原实现；`provenance/` 记录两者的对应关系。精简代码不是重新训练得到的新版本，也没有替换论文中的实验实现。

完整文件用途见 [FILE_GUIDE.md](docs/FILE_GUIDE.md)。完整复现的三个层次、数据需求与尚需移植的训练入口见 [REPRODUCTION.md](docs/REPRODUCTION.md)。当前包已经能独立复核统计、检查核心算子并查看原实现；不能把这些检查说成已经在另一台电脑完成了全套重训。

原始数据、权重、逐窗口概率、账本、文献 PDF 和第三方仓库不随包公开。项目许可证与通讯邮箱尚需作者确定。整理没有新增分类实验，也没有改变已有数字。
