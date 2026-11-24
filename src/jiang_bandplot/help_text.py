"""面向终端用户的简短命令参考。"""

HELP_TEXT = """bandplot  绘制 VASP 能带、胖能带和查看重要轨道

快速开始
  bandplot                          画普通能带
  bandplot -orbitals Ni:dxy O:p      画胖能带，同时打印轨道排名
  bandplot --energy -0.5 0.5         只打印费米能上下 0.5 eV 内的排名
  bandplot --plain --energy -1 1    画指定能窗的普通能带
  bandplot inspect                  查看可用轨道、自旋、k 路径和费米能

输入和默认值
  在计算目录运行，或用 bandplot plot <目录> 指定位置。
  输入是未压缩的 PROCAR、POSCAR（或 CONTCAR）和线模式 KPOINTS。
  默认取同目录 OUTCAR 的最后一个费米能，能窗为 -2 到 3 eV。
  点面积自动调节，图和 plot.json 保存到 figures、figures-2 等目录。

常用参数
  -orbitals, --orbitals <通道> ...   用空格分隔，如 Ni:dxy O:p 1,2:dz2。
  -e, --energy <下限> <上限>         指定相对费米能的区间，单位为 eV。
  --fermi <eV>                       手动指定已确认的费米能。
  --fermi-outcar [文件]               从指定 OUTCAR 读费米能，如 ../scf/OUTCAR。
  --output <目录>                    指定一个新目录作为输出位置。
  --scale <auto|数值>                 自动调点面积，或用 96 等数值固定比例。
  --plain                            画普通能带，可和 --energy 一起使用。
  --rank / --no-rank                 绘图时打印或跳过排名。胖能带默认打印。
  --formats <格式> ...               可选 png、pdf、svg、png-pdf，默认全输出。

进阶参数
  --rank-limit <整数>                显示前几项，默认 8。
  --rank-decay <eV>                   排名的能量衰减尺度，默认 0.5 eV。
                                     值越小，越侧重靠近费米能的态。
  --spin <all|up|down>                选择自旋，默认 all。
  --normalization <sum|mean>          同一通道的原子投影求和或取平均，默认 sum。
  --layout <auto|overlay|panels>      自动布局、叠加或分面，默认 auto。
  --title <文字>                     设置图的标题。有空格时用引号包住。
  --dpi <整数>                       设置 PNG 分辨率，默认 300。
  --config <plot.json>               沿用已有参数复画，通常配合 --output。
  --raw-energy                       用 PROCAR 原始能量绘图，跳过费米能平移。
  --structure <文件>                 使用指定的 POSCAR 或 CONTCAR。
  --kpoints <文件>                   使用指定的线模式 KPOINTS。
  --cache-dir <目录>                 指定 PROCAR 缓存目录。
  --no-cache                         本次跳过缓存的读取和写入。
  inspect --json                     以 JSON 输出检查结果。
  --version                          查看程序版本。
  -h, --help                         查看本帮助。

子命令
  bandplot plot --help               查看绘图参数
  bandplot rank --help               查看只排名的参数
  bandplot inspect --help            查看输入检查参数
"""
