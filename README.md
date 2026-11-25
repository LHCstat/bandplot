# Linux 能带绘图工具

`bandplot` 在无图形桌面的 Linux 上绘制 VASP 普通能带和胖能带，并可在终端打印近费米能轨道排名。计算目录提供未压缩的 `PROCAR`、同次计算的 `POSCAR`（或 `CONTCAR`）及线模式 `KPOINTS`。支持非自旋极化和共线自旋极化，检测到非共线、SOC 多分量或相位表时会报错。

## 安装

用 Python 3.9 或以上版本从 GitHub 安装。

```bash
python3 -m venv ~/.venvs/bandplot
source ~/.venvs/bandplot/bin/activate
python -m pip install 'git+https://github.com/LHCstat/bandplot.git'
bandplot --version
bandplot --help
```

离线安装时，先按文末的命令生成 wheel，再用一台可联网且与服务器兼容的 Linux 机器收集依赖。

```bash
python -m pip download --only-binary=:all: --dest wheelhouse \
  dist/jiang_bandplot-0.2.3-py3-none-any.whl
```

把整个 `wheelhouse` 复制到服务器。激活上述虚拟环境后运行：

```bash
python -m pip install --no-index --find-links ./wheelhouse jiang-bandplot==0.2.3
```

## 出图

```bash
cd /path/to/vasp/band
bandplot
bandplot -orbitals Ni:dxy Ni:dz2 O:p
bandplot -orbitals Ni:dxy Ni:dz2 O:p --energy -1 1 --scale 96
```

默认读取当前目录的 `PROCAR`、结构和路径，采用同目录 `OUTCAR` 里最后一个 `E-fermi`，并打印参考值和来源。默认能窗为 −2 至 3 eV，点面积自动调节。线模式 `OUTCAR` 的费米能可能与对应 SCF 不同。可以用 `--fermi-outcar ../scf/OUTCAR` 或 `--fermi` 加上已确认的数值指定参考。若需显示原始能量，可用 `--raw-energy`。[VASP 能带计算说明](https://vasp.at/wiki/Band-structure_calculation_using_density-functional_theory)

- `Ni:dxy` 选该元素的全部位点，`1,2:dz2` 选原子编号 1 和 2，`'*:d'` 选全部位点。编号从 1 开始。用 `O:p` 可合并已列出的 p 轨道。轨道轴沿用原始投影的坐标约定。默认对所选原子求和，`--normalization mean` 按原子数平均。
- 默认最多四个通道叠加，第五个通道起自动分面。前四个通道按输入顺序使用深蓝、朱红、蓝绿、紫色，叠加图和分面使用同一顺序。`plot.json` 保存实际配色，旧配置复画时沿用旧配色。蓝和朱红的色值参考 [Okabe 与 Ito 的配色指南](https://jfly.uni-koeln.de/color/)。
- 可用 `--layout overlay` 或 `--layout panels` 手动选择布局。共线自旋默认分开绘制，`--spin up` 和 `--spin down` 可筛选自旋。
- `--scale auto` 根据可见的非零权重、图幅和点数确定共同的面积系数。每个点使用 `面积参数 (pt²) = 系数 × 权重`。用 `--scale 96` 可固定系数。对比图还应统一投影定义、归一化、能窗和图幅。
- 默认只导出 `bands.png`，另存用于复画的 `plot.json`。用 `--formats png pdf` 可以同时导出 PNG 和矢量 PDF，也可单独选择 `svg` 或 `png-pdf`。连续运行会依次使用 `figures`、`figures-2` 等新目录；可用 `--output` 指定目录。
- 复画时可执行 `bandplot plot --config figures-fat/plot.json --output figures-repeat`。迁移计算目录后，可用 `bandplot plot /new/calc --config figures-fat/plot.json --output figures-repeat`。程序根据记录的 SHA-256 核实输入文件，并沿用能量参考、通道、分段、自旋和投影比例尺。
- 默认在 `${XDG_CACHE_HOME:-$HOME/.cache}/jiang-bandplot` 按 `PROCAR` 的 SHA-256 缓存。权重为 32 位浮点数组，缓存空间约为 `4 × 自旋数 × k点数 × 能带数 × 原子数 × 轨道数` 字节，另有少量元数据和能量。可指定 `--cache-dir` 或 `--no-cache`。

`bandplot --help` 按使用频率列出中文说明。`bandplot inspect --json` 可用于脚本检查输入。若某次计算的 `PROCAR`、结构和 `KPOINTS` 分开存放，可通过 `--structure` 和 `--kpoints` 指定匹配文件。

## 查看重要轨道

```bash
bandplot --energy -0.5 0.5
bandplot rank --energy -0.5 0.5 --rank-limit 12 --rank-decay 0.3
bandplot -orbitals Ni:dxy Ni:dz2 --energy -0.5 0.5
bandplot --plain --energy -0.5 0.5
```

只给能窗时打印排名，指定轨道绘图时会在绘图前打印同一张表。用 `--no-rank` 跳过表格；用 `--plain` 区分只指定能窗的普通能带绘图。排名覆盖所有元素和 PROCAR 轨道列，默认显示前 8 项。

每个轨道的分数为能窗内各态的原始投影权重乘以 k 路径梯形权重和 `exp(-|E-EF|/τ)` 后求和，其中 τ 默认为 0.5 eV。终端百分比以所有元素和轨道的加权分数之和为分母，表示这条 k 路径上的加权投影份额。排名对同一元素的原子求和，绘图可单独设为按原子平均。自旋筛选同时作用于图和排名。

## 从源码打包

```bash
python -m pip wheel --no-deps --wheel-dir dist .
```

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
