# v0.1.0-alpha.2

This corrective alpha preserves the external observer's evidence boundary: it
does not infer target package versions from the recorder's Python environment.

## Correction

- `vllm_version` and `torch_version` now remain `null` by default.
- Operators may supply exact observed-server values with
  `--target-vllm-version` and `--target-torch-version`.
- Two regression tests cover both unknown and explicitly supplied versions.

All `alpha.1` functionality remains: closed schema, bounded cadenced history,
privacy canaries, four-file rotation, fail-open persistence, fake-service tests,
and Markdown summaries.

Fresh GPU validation is being performed against this corrected build. It is not
claimed by the tag until the evidence is published separately.
