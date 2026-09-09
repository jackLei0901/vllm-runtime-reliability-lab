# Contributing

Please keep changes small and evidence-backed.

1. Open an issue describing the externally observable problem and the proposed
   acceptance test.
2. Add or update tests before changing the artifact contract.
3. Run `python -m unittest discover -s tests -v` and
   `python -m compileall -q src tests`.
4. State what the change proves and what remains outside the observation
   boundary.

Any new shareable field requires an allow-list update, JSON Schema update,
privacy-canary coverage and a schema-versioning decision. Never add prompt,
token, credential, command-line or arbitrary configuration fields to the v1
contract.
