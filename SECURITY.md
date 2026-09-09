# Security and privacy

## Supported use

Run the recorder only against services and processes you are authorized to
observe. Signal injection is destructive when `--dry-run` is removed and belongs
only in isolated validation environments.

## Shareable versus private output

`incident-*.json` is produced from a closed allow-list. `snapshot-env`, injection
logs, private timelines and run summaries are lab metadata and must be reviewed
before sharing.

The project does not claim that a static field allow-list is a general DLP
system. Changes to the artifact schema require privacy-canary tests and review.

## Reporting

Please report a vulnerability through GitHub's private security advisory flow.
Do not attach production incident artifacts, credentials, prompts or customer
data to a public issue.
