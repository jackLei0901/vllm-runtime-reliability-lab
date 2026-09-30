# H800 单次会话准备协议

状态：2026-09-30，执行协议 v1，用户已批准固定 H800 路线，未执行。本地提交冻结；须推送并确认远端同一 commit 后才形成公开冻结，未公开前不开卡。中文版为准；[English](SESSION_PROTOCOL.md) 为译文。遵循[准备包](README.zh-CN.md)的源码、布局和数值预测。

## 本轮收益与范围

只回答：真实 SM90 最小扩展能否构建，以及 B1/B2 guard 是否拒绝已知错误输入而保留正确输入、singleton 和 graph capture/replay。Base/fix 均从 #55537 `7b054aca96cea8be1369d651c3434ad140580b92` 建立，因此 B0 两臂都应拒绝。不会执行完整 vLLM 构建、K5 serving、K6 开销测试或上游发布。成功仍只是本地修复证据，不证明生产兼容性或外部采用。

## 开卡前冻结条件

- 人工复核修正后的 K-major B guard，以及正确 `Bs.T` / 错误 `.contiguous()` fixtures；公开提交本准备包、隔离 runner、测试和本协议，再记录 commit 与逐文件 SHA-256。开卡前不得已有结果。
- 固定源码及 parent、干净 CUTLASS `v4.7.1` 的可重建来源准备好；旧远端源码和已有 extension 不能替代两臂构建。依赖下载和 CMake configure 的可用性尚未实测。
- 新实例重新核对主机、SSH 指纹、SM90、cgroup 内存和两块盘的实际可用空间。不沿用之前规格作为当前事实。

## 固定路线：H800 单次会话，最多 120 分钟

本次仅采用 H800 实例内构建和测试，不保留运行中切换路线的选项。撤回 AutoDL 无卡模式和本地 WSL 构建计划。用户提供的本机为 6 逻辑处理器、约 15.9 GiB RAM；Ubuntu 22.04 / WSL2 当前 7.7 GiB RAM、2 GiB swap，nvcc/ninja/cmake 不在 PATH，python3 无 pip。WSL 虚拟盘报告的空闲容量不代表 Windows 物理盘容量。以上支持“不为本次验证搭建本地环境”的投入决定，不证明本机永远不能编译。

| 阶段 | 上限 | 停止条件 |
| --- | --- | --- |
| 环境/身份、CPU 测试、编译命令核对 | 20 分钟 | torch 2.13.0+cu130、nvcc/ninja/CMake、SM90、内存/空间或源码/依赖身份不符；CPU 检查跳过；无法取得同 pin 的 CMake 参考 |
| Base 最小构建 | 35 分钟 | 注册、链接、空间或时间失败；不修改参数重试 |
| Fix 最小构建 | 35 分钟 | 同上；base 未可用或 guard/参数复核未完成则不进入 |
| 两臂逐例隔离测试 | 各 7 分钟 | 预算耗尽的未运行案例为 unscored，不追加时间或重跑 |
| 封存、下载核验、关机 | 16 分钟 | 保留关机时间，不为补测侵占 |

120 分钟从用户启用计费实例起算，包含传输、依赖准备和排错；启动时间不明确时先确认，不能从 SSH 接通重新计时。阶段预算不得挪用；任一预检失败停止，不能靠安装大型新环境、改 flags、换源码或完整构建补救。既有匹配环境只在重核后复用。两路编译，每个构建传 `--timeout-seconds 2100`；超时终止编译进程组并检查残留进程。若实际计费耗时已不足下一阶段预算，提前封存并关机。停止不是负面 kernel 结论，而是 apparatus unscored。

将 `CUDA_HOME/bin` 和已核验的 `$VENV/bin` 加入 PATH，预检 `command -v nvcc`、`command -v ninja`。使用现有 torch 环境执行两个 CPU suite，实际 tensor 项必须通过而非 skip。从同 pin 的 CMake configure 提取 blockwise `.cu` 编译命令，保存参考 artifact 和摘要；核对 gencode、宏、语言标准、include、优化及链接配置，解释最小 namespace/构建方式引入的差异。不能靠连续修 link error 猜配置。

Scratch 必须是核对后的绝对目录，两臂/结果目录均新建。每阶段前检查空间，至少保留 3 GiB；这是操作储备，不是容量估算证明。空间不足只列出明确可丢弃的缓存/本轮产物供确认，不删除模型、旧源码、环境或结果。不能靠未确认的清理计划通过预检。

准备环境须安装 pytest，并真实执行 `test_collected_id_resolves_from_unrelated_directory`：收集 ID 后从固定 packet cwd/rootdir 重新收集，必须恰好一项。此项和真实 CPU tensor 检查均不得 skip。本地缺依赖导致的 skip 不计作通过。

在同一核验实例构建两臂，保留原 receipt 字节。`binary_file` 在 receipt 所在目录定位 `.so`，核对原摘要，原构建路径只作追溯。构建 receipt 记录主机/系统/libc/Python；另存运行主机/SSH、GPU/驱动、内存/磁盘和 torch/CUDA 身份。若主机或环境发生变化，本轮停止，不将 clone、跨主机迁移或新实例当作续跑。

## 测试执行

解析并核验 README 中所有绝对路径变量后执行：

```bash
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm base --receipt "$BASE_BUILD/build_receipt.json" --out "$BASE_RESULTS" --budget-seconds 420
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm fix --receipt "$FIX_BUILD/build_receipt.json" --out "$FIX_RESULTS" --budget-seconds 420
```

先完整 collect 30 个唯一 node ID，再按 collection 次序逐项启动新进程，每项最多 45 秒且受 arm 总预算约束。每项仅一次，无 `-x`，不因预测失败隐藏剩余案例。Base 预定 3 个错误 capture skip、27 个实际测试；fix 实际测试 30 个。意外 skip、缺失/坏 receipt、进程崩溃、上下文不可用、setup error 或预算耗尽均 unscored。断言失败且 CUDA 上下文仍可用才是 prediction_missed。只允许 base 的三项准确 skip 理由记 planned_skip。

隔离 runner 与测试摘要必须在 build receipt 中绑定。保存 collection、逐例 start/end、返回码、日志、JUnit 和 run_receipt.json。全部预测符合仅记 all_predictions_matched；不改阈值、不重评分、不补跑。Graph guard 在 capture 时检查；replay 不重复检查改变后的数据。

两种 shape 下分别只把 A 或 B 换为 int8 的四项是本轮新预测；历史 B2 只测试过两者同时 int8。每臂 7 分钟窗口未经实测；runner 记录逐例和前三项耗时，投影为一次 collection 耗时加前三项实际运行耗时的十倍。固定 cwd/rootdir 和 `-c` 指向 packet 的空 pytest.ini，配置文件也绑定 receipt。投影超窗不延长预算；末尾 graph 案例可能未运行，须保持 unscored。

## 封存与退出

私有封存源码/CUTLASS/工具身份、patch/harness/binary 摘要、CPU 测试、CMake 参考/差异、build.ninja、构建日志、逐例 artifacts、耗时和空间。下载后重新计算归档 SHA-256；公开记录只给必要配置、计数、结论和限制。停止后按用户授权关机，云控制台停止计费仍需确认。

四小时可行性门槛仍要独立作 continue/stop 决策：结合实际构建资源、测试结果及上游路线限制，明确剩余 K5/K6 是否值得学习投入。本会话不是该项目全部完成。
