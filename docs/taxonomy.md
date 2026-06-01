# Prohibited Content Taxonomy

The taxonomy separates hard blocks from manual-review cases. Each category is
designed to map to detector output, audit logs, and regulator-facing rationale.

| ID | Category | Default action | Rationale |
| --- | --- | --- | --- |
| `sexual` | Sexually explicit content | block | Bank channels must not distribute pornographic or explicit sexual imagery. |
| `sexual_minors` | Sexualized minors or apparent minors | block, fail closed | Illegal and maximum severity. |
| `nudity_contextual` | Nudity without explicit sexual act | review | Some medical, artistic, or educational contexts may be legitimate. |
| `violence_gore` | Graphic violence, gore, severe injury | block | Reputational and policy risk. |
| `violence_contextual` | Non-graphic violence or weapons in benign context | review | News, sport, history, or safety training can be legitimate. |
| `dangerous` | Explosives, weapons construction, terrorism, self-harm instructions | block | Real-world harm risk. |
| `self_harm_instruction` | Instructions, encouragement, or romanticizing self-harm | block, fail closed | High severity user safety risk. |
| `hate_extremism` | Hate, extremist symbols, harassment of protected classes | block | Legal, reputational, and policy risk. |
| `criminal_financial` | Criminal planning, fraud, financial crime | block | Directly relevant to bank threat model. |
| `personal_financial_data` | Cards, account data, passports, IDs, addresses, phones | block, fail closed | Privacy, bank secrecy, and compliance risk. |
| `brand_reputation` | Bank brand in unsafe or humiliating context | review | Requires brand and legal interpretation. |
| `composite_violation` | Inputs are separately benign but jointly unsafe | review or block | Requires multimodal context, e.g. prompt plus image plus OCR. |

## Composition Rules

Some violations only appear when signals are combined:

- prompt requests "make it more realistic" plus input image with weapon use;
- safe-looking portrait plus OCR extremist slogan;
- bank-card mockup plus visible account number;
- historical symbol plus hateful caption;
- medical imagery plus sensational prompt.

The aggregator should preserve all detector evidence so a reviewer or auditor
can reconstruct the decision.

