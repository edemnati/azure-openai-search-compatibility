# Security

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or include credentials, tokens,
customer data, private endpoints, or resource identifiers in an issue.

Use GitHub's private vulnerability reporting feature for this repository. Include a clear
description, reproduction steps, affected versions, and the potential impact.

## Credential handling

This project does not persist credentials. Applications should use Microsoft Entra ID and
managed identity in production. Environment files, live golden datasets, and generated
evaluation results are excluded from source control by default.

If a credential is exposed, revoke or rotate it immediately before reporting the issue.
