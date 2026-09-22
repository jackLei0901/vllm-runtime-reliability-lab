# 相同版本，不等于可复现的运行时身份

这次 GPU 实验中，所有冻结的 package version 都能对上，但我们仍然在模型启动前
停止了实验。

这是正确结果。

## 看似合理的捷径

已发布 vLLM 背压实验所使用的 dependency pool 还保留着 symlink，但 symlink 指向的
原安装目录已经丢失。source tree、预编译 vLLM wheel、两个 arm 环境，以及冻结的
package/version 清单仍然存在。

最直接的做法，是在原路径重新安装完全相同的版本，然后继续 native stack 实验。
这样 import 能成功，环境表面上也足够相似。很多复现记录会到此为止。

这个 Lab 不接受这一标准。

## 身份门禁

原实验为 base/fix 两个 arm 保存了关闭形状的 build identity，其中包括 source tree、
wheel digest、native binary digest、dependency provenance、可见 distributions、
RECORD SHA-256 和 verified file count。

环境重建后，使用同一个 generator 重新生成记录，并与已评审记录逐字节比较：

| 检查 | Base | Fix |
| --- | ---: | ---: |
| 版本差异 | 0 | 0 |
| RECORD SHA-256 差异 | 144 | 144 |
| verified-file-count 差异 | 6 | 6 |
| 额外 distribution | 1 | 1 |

环境通过了版本清单，却没有通过证据身份。

## 为什么这个区别重要

package version 只命名一个 release，并不能唯一确定实际 wheel、installer 行为、生成的
entry point、安装布局，以及 runtime 最终 import 的每一个文件。可变的模型分支还会
引入另一条独立漂移来源。

普通开发中，语义近似可能足够；但 base/fix 诊断必须保证预期 patch 是唯一变化量。
否则新的 stack 或行为差异就不能只归因于目标修复。

因此，正确做法不是“先跑，再补一个 caveat”，而是在模型下载、服务启动和 GPU 执行
之前停止。

## 这次失败证明了什么

它没有证明任何关于 vLLM fault、PyStack、py-spy 或 native frame classification 的
结论。它只证明拟使用的环境不是已评审环境，不能继承旧实验结论。

这种拒绝本身就是诊断结果：

```text
相同版本
  != 相同安装产物
  != 相同实验身份
  != 可以复用旧结论
```

## 可执行规则

可复现的 runtime diagnosis 至少应保留：

1. 输入 wheels 及其内容哈希；
2. 完整安装环境或可恢复的不可变归档；
3. 当生成文件包含路径时，固定 canonical installation path；
4. source tree 和 native binary identity；
5. 模型 immutable revision 与文件 manifest；
6. 执行前后独立重新生成的 identity record。

lock file 仍有价值，但它只是输入配方，不是两个安装环境相同的证明。

完整的有界结果见
[`Stage B preflight record`](../results/native-stack-pair-stage-b-preflight-20260922/README.md)。
本次重建环境没有产生任何 GPU fault claim。
