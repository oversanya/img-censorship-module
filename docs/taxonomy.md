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
| `sexual_pornography` | Sexual content and pornography | block | Explicit sexual content is not acceptable in bank-owned channels. |
| `extremism_terrorism_symbols` | Extremism, terrorism, and prohibited symbols | block, fail closed | Legal, license, and reputation risk. |
| `violence_cruelty_gore` | Graphic violence, cruelty, blood, torture | block | User-safety and reputation risk. |
| `self_harm_suicide` | Self-harm and suicide | block, fail closed | High severity user-safety risk. |
| `drugs_propaganda` | Drugs and drug propaganda | block | Legal and reputation risk. |
| `weapons_explosives_crime_instructions` | Weapons, explosives, criminal instructions | block, fail closed | Real-world harm and criminal enablement risk. |
| `fraud_phishing_social_engineering` | Fraud, phishing, social engineering | block, fail closed | Direct banking abuse risk. |
| `fake_documents_cards_contracts` | Fake documents, cards, certificates, contracts | block, fail closed | Enables fraud and identity abuse. |
| `personal_biometric_data` | Personal and biometric data | block, fail closed | Privacy and compliance risk. |
| `misleading_financial_content` | Financially misleading content | block | Consumer protection and advertising compliance risk. |
| `market_investment_manipulation` | Investment and market manipulation | block | Market abuse and regulatory risk. |
| `discrimination_hate_speech` | Discrimination and hate speech | block | Legal and reputation risk. |
| `political_agitation_symbols` | Political agitation and disputed political symbols | review | Context-dependent legal and reputation risk. |
| `brand_ip_misuse` | Unauthorized brand, logo, or IP use | review | Intellectual property and brand risk. |
| `official_interface_impersonation` | Imitation of official interfaces or government bodies | block | Can enable deception and fraud. |
| `bank_reputation_harm` | Content defaming bank, clients, or employees | review | Requires reputation and legal review. |
| `gambling_casino_quick_money` | Gambling, betting, casinos, quick-money claims | review | Compliance and reputation risk. |
| `sanctions_military_geopolitical` | Sanctions, military, geopolitical risk | review | Context-dependent sanctions and geopolitical risk. |
| `fraudulent_qr_payment_push` | Fraudulent QR codes, payment forms, fake push notifications | block, fail closed | Direct payment fraud risk. |

## Composition Rules

Some violations only appear when signals are combined:

- prompt requests "make it more realistic" plus input image with weapon use;
- safe-looking portrait plus OCR extremist slogan;
- bank-card mockup plus visible account number;
- historical symbol plus hateful caption;
- medical imagery plus sensational prompt.

The aggregator should preserve all detector evidence so a reviewer or auditor
can reconstruct the decision.
