# Q4 痛点样本：候选 ID 快照结果

状态：**候选框架已完整封存；尚未读候选 issue 正文，也未作纳入、排除或标签判定**。

- 公开的协议、A1/A2 修订、M-part 指南和 runtime model `0.1.0` 固定于
  [`62123d0`](https://github.com/jackLei0901/vllm-runtime-reliability-lab/commit/62123d08b0a6bce7dbba75617014b99b776ce879)。
- 执行方法先在本地提交 `aa1d684` 中固定；见
  [执行冻结单](PAIN_POINT_CANDIDATE_SNAPSHOT_FREEZE_2026-09-28.md)及
  [抓取脚本](../../scripts/freeze_pain_point_candidates.py)。
- [完整候选记录](../../data/pain-point-sample/candidate_snapshot_2026-09-28.json)
  只保留查询、UTC 时间、分页完整性和 issue 编号，不含标题、正文或评论。

## 抓取与校验

| 项目 | 结果 |
| --- | --- |
| 仓库和接口 | `vllm-project/vllm`，`GET /search/issues`，由 `gh api` 调用 |
| 时间 | 2026-09-28 07:15:38–07:25:30 UTC；逐片记录时间，**不是原子时点快照** |
| 分片 | 8 个关键词 × 8 个创建月份 = 64；全部 `incomplete_results=false` |
| 分页 | 共 65 页；1 个分片为两页；最大分片 116 条；无需按创建日再拆分 |
| 原始命中 | 1,088 次；其中两个分片为零结果，亦保留记录 |
| 去重并集 | **823 个候选 issue 编号** |
| 并集 SHA-256 | `945af34067cea72ffa35c5e64a98854b751c40d03d67c11906de3f781a470b8c` |
| 固定伪随机顺序 SHA-256 | `36b2ac4e24d7c823b84e949cea9efd4534908dcb8317539b8e2b9f0d19ca630c` |
| JSON 文件 SHA-256（LF 规范化后的提交字节） | `e7753b9f60422802612135dd289f863081eedcd12c9a02e0eda93c9c85a4f685` |

摘要中，编号列表使用 ASCII 十进制、每个编号后一字节 LF。离线重算了
64 个分片各自的排序、唯一性、计数和摘要，再由分片重新构造 823 个编号
的并集及固定顺序；所有断言通过。输出没有 `title`、`body`、`comments`、
`html_url`、`user` 或 `login` 字段。

抓取后只修正了脚本在 Windows 上写出 CRLF 的格式问题，并将现有 JSON
机械转换为 LF；提交字节的摘要列于上表。没有重跑查询、改动候选编号、
字段、筛选规则或随机顺序。

抓取曾在第 10 个分片后遇到匿名 GitHub Search 速率限制，当时记录仍为
`status=incomplete`，没有进行筛选。配额恢复后以 7 秒请求间隔从第 11
个分片继续；完成记录包含每片的 UTC 时间。此过程中没有改变查询词、
日期约束、候选集合规则或随机种子。

下一阶段才按记录中的 `randomized_numbers` 顺序读取报告正文，依协议
逐项记录排除原因，在第 40 个合格报告处停止。不能将 823 解释为故障
总量或由此估计现有 DFX 工具的漏检率。
