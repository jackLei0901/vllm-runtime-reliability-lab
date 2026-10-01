# 最小性能协议：开卡前交接与执行顺序

状态：本地准备稿，未确认公开冻结；不是开卡授权。不得复用旧数据评分。

## 1. 开卡前还必须确认

- 本目录、两版最小协议、`tests/test_sm90_perf_minimal.py` 与后继清单一起提交、推送。
- 公开 commit 下清单可读取，且与本地字节一致。运行工具会再次核验。
- 用户确认累计工作时间与 18 小时预算；没有完整工时账本，不填写虚构累计值。
- 用户确认后继 90 分钟会话预算。历史会话实际为 2,295 秒，即 38 分 15 秒；
  加上本次 90 分钟是 128 分 15 秒，原先“128 分钟”是取整值。
- 检查准备盘仍保留环境、模型、同 pin 源码、CUTLASS 和 CMake reference；
  历史准备完成不等于重启后的现况已确认。
- 两段理解说明已于 2026-10-01 完成初稿→纠正→确认采用。用户提供初稿，
  助手指出泛化对齐直觉不适用于本次分派，用户确认采用以下修正版。
  这是经辅助复核的解释记录，不称为用户独立撰写或独立掌握的证明。

**解释一：** 原代码按 M 能否被 4 整除选择路径：64 不交换 A/B，走
cooperative 128×128 tile；63 交换 A/B 后走 ping-pong 128×16 tile。
候选修改让所有 M≤64 都交换，因此改变了 64 的路径，但保持 63 的路径不变。
需要验证新路径是否保持正确性，并在实际小 M 上更快，不能仅凭对齐判断性能。
64 不等于填满 128 行 tile；也不能将 63 直接称为标量访存的慢路径。
tile 数不是启动 CTA 数，也不能单独证明性能原因。

**解释二：** 若一个 decode step 确实有 63 个 token，而解析配置没有捕获
尺寸 63、下一个尺寸是 64，就预期补齐到 64，以复用已捕获的 CUDA graph。
但客户端并发 63 不保证每个 step 都有 63 个 token；cooperative kernel 只证明
走了对应分派路径，不能单独证明实际 M=64。因此，本次精确 M 仍属于推断，
不是直接测量。不是 Profiler 自动发现“padding 更快”而改变输入尺寸。

**区分：** 框架决定 graph padding，算子根据收到的 M 决定分派，计时才判断
哪条路径更快。没找到边界判断也不能证明 padding 到了 64。

## 2. Linux 工作目录与已有材料

准备根目录用 `$ROOT` 表示；实际私有路径从交接记录读取，不写入公开文件：

- `env/bin/python`：Python 3.12、Torch 2.13.0+cu130。
- `vllm-src`：clean `7b054aca96cea8be1369d651c3434ad140580b92`。
- `cmake-reference-r3`：真实 compile_commands 和 patched Torch headers。
- `verified-model/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3`。
- 保留的父提交 wheel、安装日志及 `serving_build.json`。

重启后只读核查 hostname/SSH host key、cgroup memory、磁盘、GPU 型号/SM 数、
CUDA/nvcc、解释器、模型字节与 source pin。CUTLASS 路径从保留的 CMake
reference/源码目录确认，不猜测路径。新包放新目录，历史 receipts 不覆盖。
传输包与源码 SHA 在本地和远端各核对一次。不再下载模型或重新安装完整 vLLM。

## 3. 会话预算及执行

总上限 90 分钟；阶段上限相加为 85 分钟，余下 5 分钟为缓冲，不是额外重试预算。
开始任何必需阶段前，剩余时间必须覆盖该阶段及最后 10 分钟封存。
两臂构建顺序执行 base→variant，共享 35 分钟墙钟窗口，各自不超过 20 分钟。
E4 仅在内存余量门槛通过时与构建并行；它是辅助，失败不阻断 E3。
不得启动后续工作来弥补超时。

现场在已确认可用的 tmux 持久会话中执行；SSH 断开后 attach 原会话，保留变量和截止时间，不重开已开始的阶段或重置预算。

使用以下占位符：`PY` 为准备环境解释器，`P` 为新包目录，`WORK` 为新建私有
结果目录，`FREEZE` 为此次完整公开 SHA；`SRC_BASE`、`SRC_VARIANT` 是两个
独立 worktree，均在同一个 pin，variant 只应用本目录 patch。

### 3.1 身份与构建（身份 5 分钟，构建/E4 35 分钟）

先运行两臂 `build_perf.py --plan`，核查 base clean、variant 仅一个 dispatcher
变更。第一阶段运行下面的无 CUDA context 预检；编译要求可用内存 ≥32 GiB。
空 GPU PID 清单、单 GPU 显存 ≤128 MiB 必须同时满足；未知清单直接停止。
只有 cgroup/物理有效上限 ≥96 GiB 且可用内存 ≥64 GiB 才允许 E4 并行。
collector 在启动前再次检查内存；余量不足时跳过 E4，不冒险影响必需构建。

```bash
"$PY" "$P/runtime.py" --freeze-commit "$FREEZE" --out "$WORK/public_freeze.json"
export SM90_PUBLIC_FREEZE_RECEIPT="$WORK/public_freeze.json"
"$PY" "$P/apparatus.py" --out "$WORK/preflight" --freeze-commit "$FREEZE"
"$PY" "$P/build_pair.py" --base-src "$SRC_BASE" --variant-src "$SRC_VARIANT" --cutlass-src "$CUTLASS" --cmake-reference "$REFERENCE" --out "$WORK/builds" --freeze-commit "$FREEZE"
```

首个 `runtime.py` 命令是现场唯一 GitHub 核验，在最初五分钟内、平台学术加速/代理
已启用的环境执行（不猜代理地址、不记录凭证）。核对公开原始清单与本地完整字节后，
写入新的私有收据：commit、清单 SHA-256、URL、核验时间与成功状态。
联网命令失败则保存未评分收据，立即停止，不进入构建；不覆盖首轮文件。
之后所有工具只离线核对这个收据、本地清单和包内各文件；收据缺失、错误或本地
字节变化时拒绝运行，绝不自动回退联网。后台 E4、builder 子进程继承同一环境变量。

`build_pair.json` 的 `shared_deadline_utc` 是固定构建截止时间；第二臂超时自动取
min(1200, 剩余秒数)，builder 在启动编译进程前再次计算，过期不启动。
E4 的截止时间不得超过这个共享截止时间。两臂收据位于 `$WORK/builds/base` 和
`$WORK/builds/variant`；设置 `BASE="$WORK/builds/base"` 和
`VARIANT="$WORK/builds/variant"`，下列命令使用它们。失败保留
receipt，不猜参数、不更换源码、不作第三次装置尝试。两臂参数和编译 reference
相同，差异只限 namespace 与一行分派。

E4 在 build 期间运行 `serve_trace.py --configuration forced --mode nsys`：
传入模型、源码、保留的 serving provenance、新输出目录、公开 SHA 和固定
`--deadline-utc`。collector 要求不超过 25 分钟的剩余窗口，且有自己的 startup cap。
没有足够启动/采集时间时跳过 E4，记 `unwitnessed`。

此处不导出或复核 E4；`replay_check.py` 移到 E3 后或关机后的离线阶段。
启动 E4 时在私有记录中保存 collector PID/PGID；server 自建进程组的 PGID
由 collector 即时写入 `collection.json` 的 `server_process_group`。

### 3.2 E2（5 分钟）

构建结束（即使提前）或共享截止时间到达，以先发生者为准，先结束 E4，不能直接开始 E2。
若 E4 仍在运行，核对本次 collector 与收据中的 server PGID 对应的命令和归属后，
仅终止这些进程组；不使用宽泛 pkill。先 SIGTERM，最多给 10 秒清理，再对仍存在的
本次进程组 SIGKILL；不等待采集/flush 自然完成，不延长共享窗口。
中断的 E4 在单独私有阶段记录中标为 `unwitnessed`，保留 collector 原收据与不完整文件。
保留剩余进程/显存检查，不因终止命令成功就假定 GPU 已释放；检查失败不启动 E2/E1。

```bash
"$PY" "$P/apparatus.py" --idle-only --out "$WORK/pre-e2-idle" --freeze-commit "$FREEZE"
nsys profile --trace=cuda,nvtx --force-overwrite=false --output "$WORK/dispatch" "$PY" "$P/identity_probe.py" --base "$BASE/build_receipt.json" --variant "$VARIANT/build_receipt.json" --out "$WORK/probe" --freeze-commit "$FREEZE"
nsys export --type=sqlite --output "$WORK/dispatch.sqlite" "$WORK/dispatch.nsys-rep"
"$PY" "$P/trace_tools.py" --sqlite "$WORK/dispatch.sqlite" --probe "$WORK/probe/probe.json" --out "$WORK/dispatch-check"
```

要求两臂各三个正确归属的 M64 kernel，base cooperative、variant ping-pong。
不是同一个 kernel 才能继续。

### 3.3 E1（10 分钟）

E2 的 nsys/probe/export 必须全部退出。再运行同一空闲检查，通过后才运行 E1。

```bash
"$PY" "$P/apparatus.py" --idle-only --out "$WORK/pre-e1-idle" --freeze-commit "$FREEZE"
"$PY" "$P/correctness.py" --base "$BASE/build_receipt.json" --variant "$VARIANT/build_receipt.json" --out "$WORK/correctness" --freeze-commit "$FREEZE"
```

176 个唯一用例全部通过才能计时。base 全部通过而 variant 数值不匹配，记录
`not_worth_proposing`；其余错误为 `unscored`。保留所有首轮结果，不重跑挑选。

### 3.4 E3（20 分钟，独立运行）

确认两个构建、server、nsys/export 都已结束；不得在其他 GPU 任务运行时开始。

```bash
"$PY" "$P/timing.py" --session S --base "$BASE/build_receipt.json" --variant "$VARIANT/build_receipt.json" --dispatch "$WORK/dispatch-check/dispatch.json" --correctness "$WORK/correctness/correctness.json" --out "$WORK/timing" --freeze-commit "$FREEZE"
```

工具在初始化 CUDA 前记录只读 process/GPU snapshot，预期 GPU 无进程且显存 ≤128 MiB。
首次三个 cell 的投影超出剩余窗口则停止，记 `insufficient_evidence`，保留首轮部分结果。
E3 超时也用相同标签。输出含全部 80 个校准、80 个配对记录，
判定按完整性→44 个双向对照→退化→按 M 的 7/9、3/4 顺序执行。
没有改善不称为等价，缺失不称为没有差距。

### 3.5 封存与关机（预留 10 分钟）

E4 的复核/SQLite export 在 E3 后且时间充足时进行，优先关机后离线执行；不得
侵占最后十分钟封存，不运行于 E2/E1/E3 期间。先封存、下载原始 report 并核对 SHA，
复核前再次核对 report 与 collection 中的 digest，不完整或被终止的采集不升级为 witnessed。
有完整 report 时，在具备兼容 nsys 的离线环境运行：

```bash
export SM90_PUBLIC_FREEZE_RECEIPT="$WORK/public_freeze.json"
"$PY" "$P/replay_check.py" --collection "$WORK/serving/collection.json" --nsys-report "$WORK/serving/forced/serve.nsys-rep" --out "$WORK/replay" --freeze-commit "$FREEZE"
```

检查器自行以 UTC normalize 导出 SQLite 并保留命令和摘要；E4 不影响 E3 门槛。
离线环境需复制并核验原 public-freeze 收据，不新建或修改其身份字段；收据与原始
collection 保持私有。收据是本次首次联网核验的本地记录，不是独立的数字签名证明。

封存原始结果、build logs/receipts、trace、阶段时间与失败记录；SHA 在远端及
下载后的本地核对。原始日志/PID/UUID 保持私有。确认下载成功后按已授权主机
关机，再由用户在平台控制台确认停止计费。失败也执行封存和关机。

## 4. 不作出的承诺

提交范围仅为新包目录、`tests/test_sm90_perf_minimal.py` 和两版最小协议，按明确路径
暂存；不使用 `git add .`。不提交废弃 V1 的 recovery 工具/测试或其他任务修改。
`RESULT_2026-09-30` 的 CUTLASS SHA 修正若需提交，另作独立提交，不混入本包。
移除了对未跟踪恢复文档的链接，无需为修复悬空链接扩大本次提交。

两段解释已有用户初稿、助手修正和用户确认采用；保留这条来源说明，不把辅助复核
冒充独立理解测试。该确认不代替预算确认、公开冻结或 GPU 实证。

CPU 测试通过不是编译成功、GPU 正确性通过或性能改善。35 分钟构建窗口和
20 分钟计时窗口是停止上限，不是已测得的耗时承诺。E4 不强求完成，不能
为了补 E4 推迟必需计时或关机。此次装置失败后，本实验按协议结束。

## 5. 核查通过后才执行的冻结命令（PowerShell）

在 Lab 仓库根目录执行。此处仅提供命令，本次准备不执行暂存、提交或推送。
先确认当前分支为 `lab/runtime-model`，暂存区没有其他任务的文件。

```powershell
if ((git branch --show-current).Trim() -ne 'lab/runtime-model') { throw 'Wrong branch' }
if (git diff --cached --name-only) { throw 'Index contains other work; inspect before staging' }
& .venv/Scripts/python.exe experiments/kernel-operand-contracts/sm90-block-fp8-perf-minimal/freeze_packet.py --check
if ($LASTEXITCODE -ne 0) { throw 'Manifest mismatch' }
git add -- experiments/kernel-operand-contracts/sm90-block-fp8-perf-minimal tests/test_sm90_perf_minimal.py docs/kernel/SM90_BLOCK_FP8_PERF_MINIMAL_PROTOCOL.md docs/kernel/SM90_BLOCK_FP8_PERF_MINIMAL_PROTOCOL.zh-CN.md
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Index whitespace check failed' }
```

逐项核查暂存列表和 diff，仅包括上述范围后，再执行：

```powershell
git commit -m "freeze minimal SM90 dispatch performance protocol r6"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:lab/runtime-model
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
$freezeCommit = (git rev-parse HEAD).Trim()
& .venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'experiments/kernel-operand-contracts/sm90-block-fp8-perf-minimal'); import runtime; print(runtime.verify_public_freeze(sys.argv[1]))" $freezeCommit
if ($LASTEXITCODE -ne 0) { throw 'Public freeze not verified; do not book' }
$freezeCommit
```

清单已在本地生成，但只有提交推送后核对公开字节，才成为公开冻结。
公开核验失败时保持未确认，不据此修改阈值或启动 GPU。
