# Security Policy

## Supported Use

This repository is a research and benchmarking codebase, not a hosted service. Security fixes are handled on a best-effort basis for the latest state of the default branch.

## Reporting

Please do not open a public issue for a suspected security problem that could expose users, credentials, or infrastructure.

Instead, report the issue privately to the repository maintainers through the contact method listed on the project hosting profile. Include:

- a short description of the issue;
- affected files or components;
- reproduction steps or a proof of concept;
- impact assessment and any suggested mitigation.

## Disclosure Guidance

- Do not publish secrets, tokens, or private dataset locations in reports.
- Allow maintainers reasonable time to investigate and respond before public disclosure.
- If the issue only affects local development workflows, clearly state the assumptions required for exploitation.

## Local configuration

Keep real provider credentials in the ignored root `.env` or the process
environment. The template `.env.example` is intended for distribution. Use
`python3 tools/check_env.py --provider azure` (or another provider) to check
configuration presence without printing key values or calling an API.

When sharing logs or bug reports, remove credentials, private endpoints, and
absolute local paths. Inspect source content and document metadata before
distribution; Git commit metadata is separate from working-file contents.
