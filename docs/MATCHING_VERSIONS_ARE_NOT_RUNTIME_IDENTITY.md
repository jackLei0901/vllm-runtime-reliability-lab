# Matching package versions are not a reproducible runtime identity

We stopped a GPU experiment before starting the model even though every frozen
package version matched.

That was the correct result.

## The tempting shortcut

The retained environment for a published vLLM backpressure campaign had lost
the installation target behind its dependency-pool symlinks. The source trees,
precompiled vLLM wheel, arm environments, and frozen package/version inventory
were still available.

It was tempting to reinstall those exact versions at the original path and
continue with the native-stack experiment. Imports would work. The software
would look equivalent. Most reproduction notes would call that sufficient.

This lab does not.

## The gate

The original campaign recorded a closed build identity for both experimental
arms. It includes source trees, wheel digests, installed binary digests,
dependency provenance, every visible distribution, RECORD SHA-256 values, and
verified file counts.

After reconstruction, the same generator was run again and compared byte for
byte with the reviewed records.

| Check | Base | Fix |
| --- | ---: | ---: |
| Version mismatches | 0 | 0 |
| RECORD SHA-256 mismatches | 144 | 144 |
| Verified-file-count mismatches | 6 | 6 |
| Extra distributions | 1 | 1 |

The environment matched the version list and failed the evidence identity.

## Why that distinction matters

A package version names a release. It does not uniquely identify the wheel,
installer behavior, generated entry points, installation layout, or every file
that the runtime imports. A mutable model reference adds another independent
source of drift.

For ordinary development, semantic equivalence may be enough. For a base/fix
diagnostic claim, it can hide a second changed variable. Once that happens, a
different stack or outcome can no longer be attributed only to the intended
patch.

The correct action was therefore not to “try it and add a caveat.” It was to
stop before model download, server startup, and GPU execution.

## What the failed run established

It did not establish anything about the vLLM failure, PyStack, py-spy, or native
frame classification. It established that the proposed environment was not the
reviewed environment and could not inherit its claims.

That refusal is part of the diagnostic result:

```text
same versions
  != same installed artifacts
  != same experimental identity
  != permission to reuse an old conclusion
```

## The practical rule

For a reproducible runtime diagnosis, preserve:

1. input wheels and their content hashes;
2. the installed environment or a restorable immutable archive;
3. the canonical installation path when generated files encode it;
4. source tree and native binary identities;
5. the model's immutable revision and file manifest;
6. an independently regenerated identity record before and after execution.

A lock file remains useful, but it is an input recipe, not proof that two
installed environments are identical.

The full bounded result is published in the
[`Stage B preflight record`](../results/native-stack-pair-stage-b-preflight-20260922/README.md).
No GPU fault claim was made from the reconstructed environment.
