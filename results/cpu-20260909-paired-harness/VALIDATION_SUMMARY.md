# CPU paired-overhead harness validation

## Scope

This run validates experiment mechanics only. It uses the repository's fake
HTTP service, not vLLM, and therefore makes no claim about recorder overhead,
model latency, token throughput or production behavior.

## Environment

- Host: Microsoft Windows 10.0.26200
- Python: 3.12.14
- Base commit: `522716796ef2a953c09ebab21bc88d40d6bdba7f`
- Plan SHA-256: `3c4f102ac72796bf861144bc9f060f7c3a8bb20a4f09f0d86cb2ff1202979928`

## Command

```powershell
python experiments/overhead/paired_overhead.py `
  --plan experiments/overhead/config.cpu-example.json `
  --output results/self-test-paired-overhead-v4
```

## Reviewed result

- Four alternating arms completed: disabled, enabled, disabled, enabled.
- All four trials were valid and emitted the same workload signature.
- Both enabled arms kept the recorder alive through the workload and stopped it
  through the recorder's graceful console-signal path.
- Both enabled arms wrote a private run summary with health, metrics, process
  and GPU collector timing.
- No server or recorder process group remained after cleanup.
- The report retained each adjacent pair's relative delta and its median.

Windows returned code 1 for harness-terminated fake servers and code 0 for the
gracefully stopped recorders. Return codes from harness-controlled teardown are
platform-dependent and were not used alone to decide trial validity.

The raw logs and private JSON are intentionally excluded from version control.
The latency and throughput numbers from this fake-service run must not be cited
as product-performance evidence.
