const form = document.querySelector("#moderationForm");
const promptInput = document.querySelector("#prompt");
const scenarioInput = document.querySelector("#scenario");
const imageInput = document.querySelector("#imageInput");
const dropzone = document.querySelector("#dropzone");
const dropCopy = document.querySelector("#dropCopy");
const preview = document.querySelector("#preview");
const submitButton = document.querySelector("#submitButton");
const resetButton = document.querySelector("#resetButton");
const requestMeta = document.querySelector("#requestMeta");
const healthLabel = document.querySelector("#healthLabel");
const healthLed = document.querySelector(".status-led");
const emptyState = document.querySelector("#emptyState");
const resultContent = document.querySelector("#resultContent");
const verdictLabel = document.querySelector("#verdictLabel");
const confidenceRing = document.querySelector("#confidenceRing");
const confidenceValue = document.querySelector("#confidenceValue");
const reasonText = document.querySelector("#reasonText");
const categoryList = document.querySelector("#categoryList");
const evidenceList = document.querySelector("#evidenceList");
const signalTable = document.querySelector("#signalTable");
const signalCount = document.querySelector("#signalCount");
const signalSummary = document.querySelector("#signalSummary");
const requestId = document.querySelector("#requestId");
const taxonomyGrid = document.querySelector("#taxonomyGrid");
const auditLog = document.querySelector("#auditLog");
const auditSummary = document.querySelector("#auditSummary");
const auditCount = document.querySelector("#auditCount");
const guardrailNav = document.querySelector('.nav-list a[href="#moderation"]');

let imageBase64 = "";
let imageName = "";
let lastPayload = null;

const categoryLabels = {
  sexual: "Сексуальный контент",
  sexual_minors: "Сексуальный контент с несовершеннолетними",
  violence_gore: "Насилие и графический контент",
  self_harm: "Самоповреждение и суицид",
  hate_extremism: "Экстремизм и запрещенная символика",
  illegal_activity: "Незаконная активность",
  drugs: "Наркотики",
  weapons_crime: "Оружие и криминальные инструкции",
  deception_fraud: "Мошенничество и фишинг",
  forged_documents: "Поддельные документы",
  personal_biometric_data: "Персональные и биометрические данные",
  fraudulent_qr_payment: "Мошеннические QR и платежные формы",
  harassment: "Харассмент и оскорбления",
  financial_misleading: "Финансовое введение в заблуждение",
  investment_manipulation: "Манипуляции с инвестициями",
  discrimination_hate: "Дискриминация и hate speech",
  political_persuasion: "Политическая агитация",
  health_misinformation: "Медицинская дезинформация",
  spam_scams: "Спам и скам",
  brand_ip_abuse: "Незаконное использование брендов",
  official_interface_impersonation: "Имитация официальных интерфейсов",
  bank_reputation_risk: "Репутационный риск банка",
  gambling_fast_money: "Азартные игры и быстрые деньги",
  sanctions_geopolitical: "Санкционные и геополитические риски",
  shocking: "Шок-контент",
};

const categoryDescriptions = {
  sexual: "Обнаженность, порнография и сексуализированные изображения.",
  sexual_minors: "Любой сексуализированный контент с несовершеннолетними.",
  violence_gore: "Кровь, пытки, жестокость и графическое насилие.",
  self_harm: "Самоповреждение, суицидальные сцены или инструкции.",
  hate_extremism: "Экстремизм, терроризм и запрещенная символика.",
  illegal_activity: "Криминальные действия, незаконные инструкции, оружие, наркотики или содействие преступлениям.",
  drugs: "Наркотики, производство, продажа или пропаганда.",
  weapons_crime: "Оружие, взрывчатка и криминальные инструкции.",
  deception_fraud: "Фишинг, мошенничество и социальная инженерия.",
  forged_documents: "Поддельные документы, карты, справки и договоры.",
  personal_biometric_data: "Паспорта, номера карт, биометрия и чувствительные данные.",
  fraudulent_qr_payment: "Мошеннические QR-коды, платежные формы и push-уведомления.",
  harassment: "Оскорбления, унижение, токсичные обращения и травля.",
  financial_misleading: "Обещания, вводящие клиента в заблуждение по финансовым условиям.",
  investment_manipulation: "Манипуляции рынком и гарантированная инвестиционная прибыль.",
  discrimination_hate: "Дискриминация и ненавистническая речь.",
  political_persuasion: "Политическая агитация и спорная политическая символика.",
  health_misinformation: "Недостоверные медицинские утверждения, небезопасные советы или ложные рекомендации.",
  spam_scams: "Спам, скам, фишинг и подозрительное продвижение.",
  brand_ip_abuse: "Незаконное использование брендов, логотипов и ИС.",
  official_interface_impersonation: "Имитация интерфейсов банка, госорганов или регуляторов.",
  bank_reputation_risk: "Контент, порочащий банк, клиентов или сотрудников.",
  gambling_fast_money: "Ставки, казино и схемы быстрых денег.",
  sanctions_geopolitical: "Санкционные, военные и геополитические риски.",
  shocking: "Шокирующий или отталкивающий графический контент.",
};

const hardBlockCategories = new Set([
  "sexual",
  "sexual_minors",
  "violence_gore",
  "self_harm",
  "hate_extremism",
  "illegal_activity",
  "drugs",
  "weapons_crime",
  "deception_fraud",
  "forged_documents",
  "personal_biometric_data",
  "fraudulent_qr_payment",
  "financial_misleading",
  "investment_manipulation",
  "discrimination_hate",
  "political_persuasion",
  "spam_scams",
  "brand_ip_abuse",
  "official_interface_impersonation",
  "bank_reputation_risk",
  "gambling_fast_money",
  "sanctions_geopolitical",
  "shocking",
]);

const signalLabels = {
  ui_policy_guard: "UI-фильтр запрещенных тем",
  text_guard_stub: "Текстовый фильтр промпта",
  text_guard_heuristic: "Текстовый фильтр промпта",
  ocr_text_guard_stub: "Текстовый фильтр OCR",
  ocr_text_guard_heuristic: "Текстовый фильтр OCR",
  policy_judge_heuristic: "Эвристический арбитр политики",
  policy_fusion: "Сведение сигналов (fusion)",
  policy_judge_shieldgemma: "Мультимодальный арбитр ShieldGemma",
  ocr_adapter: "OCR текста на изображении",
  visual_classifier: "Визуальный классификатор",
  explicit_content_detector: "Детектор откровенного контента",
};

const statusLabels = {
  ok: "отработал",
  skipped: "пропущен",
  error: "ошибка",
};

const stageLabels = {
  input: "до генерации",
  output: "после генерации",
};

const verdictLabels = {
  allow: "Разрешено",
  review: "На проверку",
  block: "Заблокировано",
};

const examples = {
  safe: {
    scenario: "text2image",
    stage: "input",
    prompt: "Сгенерируй дружелюбный баннер Бета банка для семейного фестиваля: спокойный деловой стиль, люди улыбаются, без персональных данных и платежных реквизитов.",
  },
  fraud: {
    scenario: "text2image",
    stage: "input",
    prompt: "Сделай фейковое push-уведомление банка с QR-кодом для оплаты и текстом, что клиент должен срочно перевести деньги.",
  },
  docs: {
    scenario: "img2img_editing",
    stage: "input",
    prompt: "Нарисуй поддельный паспорт и банковскую справку с номером карты, чтобы выглядело как настоящий официальный документ.",
  },
  invest: {
    scenario: "text2image",
    stage: "input",
    prompt: "Сделай рекламу инвестиций: гарантированная прибыль без риска, быстрые деньги за день, срочно покупайте актив.",
  },
};

const blockScore = 0.95;
const word = "[\\p{L}\\p{N}_-]*";
const forbiddenTopicRules = [
  {
    category: "sexual_minors",
    patterns: [
      new RegExp(`(?:детск${word}|несовершеннолетн${word}|малолетн${word}|underage|minor|child).{0,60}(?:порно${word}|эрот${word}|сексу${word}|интим${word}|обнаж${word}|porn|sexual|nude)`, "iu"),
      new RegExp(`(?:порно${word}|эрот${word}|сексу${word}|интим${word}|обнаж${word}|porn|sexual|nude).{0,60}(?:детск${word}|несовершеннолетн${word}|малолетн${word}|underage|minor|child)`, "iu"),
      /(?:csam|child\s+sexual\s+abuse|child\s+porn)/iu,
    ],
  },
  {
    category: "sexual",
    patterns: [
      new RegExp(`(?:сексуальн${word}|порно${word}|порнограф${word}|эротик${word}|эротическ${word}|обнажен${word}|гол${word}\\s+тел${word}|интим${word}|nsfw|hentai|porn|pornographic|nude|nudity|explicit sex|sexual content)`, "iu"),
    ],
  },
  {
    category: "hate_extremism",
    patterns: [
      new RegExp(`(?:экстремиз${word}|экстремист${word}|террор${word}|нацист${word}|фашист${word}|свастик${word}|игил|isis|islamic state|terroris${word}|extremis${word})`, "iu"),
      new RegExp(`(?:запрещенн${word}|forbidden|banned)\\W{0,20}(?:символик${word}|symbol${word})`, "iu"),
      new RegExp(`(?:hate|ненавист${word})\\W{0,20}(?:symbol${word}|символ${word})`, "iu"),
    ],
  },
  {
    category: "violence_gore",
    patterns: [
      new RegExp(`(?:насили${word}|жесток${word}|кров${word}|пытк${word}|казн${word}|расчлен${word}|увечь${word}|мучени${word}|mutilat${word}|gore|blood${word}|tortur${word}|execution|brutal violence)`, "iu"),
      new RegExp(`(?:графическ${word}|graphic)\\W{0,20}(?:насили${word}|violence|кров${word}|blood)`, "iu"),
    ],
  },
  {
    category: "shocking",
    patterns: [
      new RegExp(`(?:шок[-\\s]?контент${word}|шокирующ${word}|отталкивающ${word}|disturbing|shock content|shocking content)`, "iu"),
      new RegExp(`(?:мертв${word}\\s+тел${word}|corpse|dead body)`, "iu"),
    ],
  },
  {
    category: "self_harm",
    patterns: [
      new RegExp(`(?:суицид${word}|самоубий${word}|самоповреж${word}|порез${word}\\s+вен${word}|повеситься|self[-\\s]?harm|suicide|kill myself|cutting wrists)`, "iu"),
    ],
  },
  {
    category: "drugs",
    patterns: [
      new RegExp(`(?:наркотик${word}|наркотическ${word}|наркоторгов${word}|кокаин${word}|героин${word}|марихуан${word}|каннабис${word}|метамфетамин${word}|амфетамин${word}|экстази|спайс|закладк${word}|drug${word}|cocaine|heroin|meth|cannabis)`, "iu"),
      new RegExp(`(?:пропаганд${word}|promotion)\\W{0,20}(?:наркотик${word}|drug${word})`, "iu"),
    ],
  },
  {
    category: "weapons_crime",
    patterns: [
      new RegExp(`(?:оружи${word}|пистолет${word}|автомат${word}|винтовк${word}|взрывчатк${word}|взрыв${word}\\s+устройств${word}|бомб${word}|weapon${word}|firearm${word}|gun${word}|explosive${word}|bomb${word})`, "iu"),
      new RegExp(`(?:как|инструкци${word}|схем${word}|гайд|manual|how to)\\W{0,30}(?:сделать|изготовить|собрать|make|build)\\W{0,30}(?:бомб${word}|оружи${word}|explosive${word}|bomb${word}|weapon${word})`, "iu"),
      new RegExp(`(?:криминальн${word}|criminal)\\W{0,20}(?:инструкци${word}|instruction${word})`, "iu"),
    ],
  },
  {
    category: "illegal_activity",
    patterns: [
      new RegExp(`(?:незаконн${word}|криминальн${word}|преступлен${word}|преступн${word}|отмывани${word}\\s+денег|взлом${word}|украсть|краж${word}|угон${word}|обход\\s+kyc|illegal|criminal|money laundering|hack${word}|steal${word})`, "iu"),
    ],
  },
  {
    category: "deception_fraud",
    patterns: [
      new RegExp(`(?:мошен${word}|фишинг${word}|скам${word}|обман${word}|социальн${word}\\s+инженер${word}|phishing|scam${word}|fraud${word}|social engineering)`, "iu"),
      new RegExp(`(?:фейк${word}|fake|поддельн${word})\\W{0,25}(?:уведомлен${word}|push|пуш|сайт|форма|payment|bank)`, "iu"),
      /(?:срочно|urgent)\W{0,25}(?:перевести|оплатить|transfer|pay)/iu,
    ],
  },
  {
    category: "fraudulent_qr_payment",
    patterns: [
      new RegExp(`(?:qr|куар|кьюар)[-\\s]?(?:код${word})?.{0,40}(?:мошен${word}|фейк${word}|поддельн${word}|перевод${word}|оплат${word}|платеж${word}|payment|transfer)`, "iu"),
      new RegExp(`(?:платежн${word}\\s+форм${word}|payment form)\\W{0,30}(?:мошен${word}|фейк${word}|поддельн${word}|fake|fraud)`, "iu"),
      new RegExp(`(?:фейк${word}|поддельн${word}|fake)\\W{0,20}(?:push|пуш)[-\\s]?(?:уведомлен${word}|notification${word})`, "iu"),
    ],
  },
  {
    category: "forged_documents",
    patterns: [
      new RegExp(`(?:поддельн${word}|фальшив${word}|фейк${word}|fake|forged|counterfeit)\\W{0,25}(?:документ${word}|паспорт${word}|справк${word}|договор${word}|сертификат${word}|карт${word}|id|passport|certificate|contract|card)`, "iu"),
      new RegExp(`(?:нарисуй|создай|сгенерируй|make|generate)\\W{0,25}(?:паспорт${word}|справк${word}|договор${word})\\W{0,25}(?:как\\s+настоящ${word}|официальн${word}|real|official)`, "iu"),
    ],
  },
  {
    category: "personal_biometric_data",
    patterns: [
      new RegExp(`(?:персональн${word}\\s+данн${word}|личн${word}\\s+данн${word}|паспортн${word}\\s+данн${word}|биометр${word}|снилс|инн|cvv|cvc|personal data|biometric${word})`, "iu"),
      new RegExp(`(?:номер${word}|данн${word})\\W{0,15}(?:карт${word}|паспорт${word}|телефон${word}|card|passport|phone)`, "iu"),
      new RegExp(`(?:платежн${word}|банковск${word})\\W{0,15}(?:реквизит${word}|данн${word}|card details|payment details)`, "iu"),
    ],
  },
  {
    category: "financial_misleading",
    patterns: [
      new RegExp(`(?:быстр${word}\\s+деньг${word}|легк${word}\\s+деньг${word}|без\\s+риск${word}|нулев${word}\\s+риск${word}|no risk|risk[-\\s]?free|quick money|easy money)`, "iu"),
      new RegExp(`(?:финансов${word}|financial).{0,30}(?:ввод${word}\\s+в\\s+заблуждени${word}|misleading)`, "iu"),
      new RegExp(`(?:гарантированн${word}|guaranteed)\\W{0,20}(?:прибыл${word}|доход${word}|return${word}|profit${word})`, "iu"),
      new RegExp(`(?:обещай|promise)\\W{0,20}(?:прибыл${word}|доход${word}|profit${word}|return${word})`, "iu"),
    ],
  },
  {
    category: "investment_manipulation",
    patterns: [
      new RegExp(`(?:манипуляц${word}\\s+рынк${word}|памп${word}|дамп${word}|инсайд${word}|pump and dump|market manipulation|insider trading)`, "iu"),
      new RegExp(`(?:срочно|немедленно|urgent)\\W{0,20}(?:покупай${word}|продавай${word}|buy|sell)\\W{0,20}(?:актив${word}|акци${word}|asset${word}|stock${word})`, "iu"),
      new RegExp(`(?:гарантированн${word}|guaranteed)\\W{0,20}(?:инвест${word}|доход${word}|return${word})`, "iu"),
    ],
  },
  {
    category: "discrimination_hate",
    patterns: [
      new RegExp(`(?:дискриминац${word}|hate speech|ненавистническ${word}\\s+реч${word}|расов${word}\\s+ненавист${word}|ксенофоби${word}|расист${word}|slur${word}|racis${word}|xenophob${word})`, "iu"),
      new RegExp(`(?:униж${word}|оскорб${word}|ненавид${word}|attack)\\W{0,25}(?:рас${word}|нац${word}|религи${word}|этнос${word}|пол${word}|ориентаци${word}|race|religion|ethnicity|gender)`, "iu"),
    ],
  },
  {
    category: "political_persuasion",
    patterns: [
      new RegExp(`(?:политическ${word}\\s+агитац${word}|предвыборн${word}|выборн${word}\\s+кампани${word}|голосуй${word}|голосовать\\s+за|поддержи\\s+кандидат${word}|митинг${word}\\s+за|political campaign|vote for|election campaign)`, "iu"),
      new RegExp(`(?:спорн${word}|controversial)\\W{0,20}(?:политическ${word})?\\W{0,10}(?:символик${word}|symbol${word})`, "iu"),
    ],
  },
  {
    category: "brand_ip_abuse",
    patterns: [
      new RegExp(`(?:незаконн${word}|без\\s+разрешен${word}|unauthorized|without permission)\\W{0,25}(?:бренд${word}|логотип${word}|товарн${word}\\s+знак${word}|trademark${word}|logo${word})`, "iu"),
      new RegExp(`(?:скопируй|подделай|имитируй|copy|fake)\\W{0,25}(?:бренд${word}|логотип${word}|айдентик${word}|brand|logo)`, "iu"),
      new RegExp(`(?:чуж${word})\\W{0,10}(?:бренд${word}|логотип${word})`, "iu"),
    ],
  },
  {
    category: "official_interface_impersonation",
    patterns: [
      new RegExp(`(?:имитируй|скопируй|подделай|fake|copy)\\W{0,25}(?:официальн${word})?\\W{0,10}(?:интерфейс${word}|сайт${word}|кабинет${word}|interface|website|screen)`, "iu"),
      new RegExp(`(?:как|под)\\W{0,10}(?:официальн${word})\\W{0,20}(?:госуслуг${word}|госорган${word}|банк${word}|регулятор${word}|government|bank|regulator)`, "iu"),
      new RegExp(`(?:госорган${word}|госуслуг${word}|центробанк${word}|government agency)\\W{0,25}(?:уведомлен${word}|форма|интерфейс|notice|form|interface)`, "iu"),
    ],
  },
  {
    category: "bank_reputation_risk",
    patterns: [
      new RegExp(`(?:порочащ${word}\\s+банк${word}|клевет${word}\\s+на\\s+банк${word}|defame bank|bank defamation)`, "iu"),
      new RegExp(`(?:банк|клиент${word}|сотрудник${word})\\W{0,20}(?:мошенник${word}|вор${word}|украл${word}|коррупционер${word}|criminal|thief|fraudster)`, "iu"),
    ],
  },
  {
    category: "gambling_fast_money",
    patterns: [
      new RegExp(`(?:азартн${word}\\s+игр${word}|ставк${word}|казино${word}|букмекер${word}|лотере${word}|беттинг${word}|быстр${word}\\s+деньг${word}|gambling|betting|casino|bookmaker|lottery|quick money)`, "iu"),
    ],
  },
  {
    category: "sanctions_geopolitical",
    patterns: [
      new RegExp(`(?:санкци${word}|военн${word}|войн${word}|геополит${word}|армия${word}|боев${word}\\s+действ${word}|спецоперац${word}|sanction${word}|military|war|geopolitical)`, "iu"),
    ],
  },
];

function normalizePolicyText(text) {
  return text
    .normalize("NFKC")
    .toLocaleLowerCase("ru-RU")
    .replaceAll("ё", "е")
    .replace(/\s+/g, " ")
    .trim();
}

function isNegatedPolicyMatch(text, matchIndex) {
  const context = text.slice(Math.max(0, matchIndex - 90), matchIndex);
  return /(?:^|[\s,.;:])(?:без|нет|не|нельзя|исключи|исключить|избегай|избегать|против|анти|without|avoid|exclude|no|not|don't|do not)[\s,.;:]*(?:[\p{L}\p{N}_-]+[\s,.;:]+){0,8}[\s,.;:]*$/iu.test(context);
}

function findForbiddenPromptTopics(prompt) {
  if (!prompt) return null;

  const normalized = normalizePolicyText(prompt);
  const categories = {};
  const matches = {};

  forbiddenTopicRules.forEach((rule) => {
    rule.patterns.forEach((pattern) => {
      pattern.lastIndex = 0;
      const match = pattern.exec(normalized);
      if (!match || isNegatedPolicyMatch(normalized, match.index)) return;

      categories[rule.category] = Math.max(categories[rule.category] || 0, blockScore);
      matches[rule.category] ||= [];
      matches[rule.category].push(match[0].replace(/\s+/g, " ").slice(0, 120));
    });
  });

  if (!Object.keys(categories).length) return null;

  const sortedCategories = Object.keys(categories).sort();
  return {
    categories,
    matches,
    sortedCategories,
  };
}

function buildUiBlockedResponse(payload, policyMatch) {
  const evidence = Object.fromEntries(
    policyMatch.sortedCategories.map((category) => [category, ["ui_policy_guard"]])
  );

  return {
    request_id: `ui-${Date.now().toString(36)}`,
    scenario: payload.scenario,
    stage: payload.stage,
    verdict: "block",
    categories: policyMatch.sortedCategories,
    confidence: blockScore,
    reason: "Blocked by UI policy guard due to prohibited topic.",
    evidence,
    signals: [
      {
        name: "ui_policy_guard",
        status: "ok",
        categories: policyMatch.categories,
        reason: "UI policy guard matched prohibited topic before API request.",
        raw: {
          mode: "frontend_heuristic",
          matches: policyMatch.matches,
        },
      },
    ],
    notes: ["Запрос заблокирован в UI до вызова /v1/moderate."],
  };
}

Object.entries(categoryLabels).forEach(([code, label]) => {
  const item = document.createElement("div");
  item.className = "taxonomy-item";
  item.innerHTML = `
    <div>
      <strong>${label}</strong>
      <span>${categoryDescriptions[code] || "Категория политики модерации."}</span>
    </div>
    <em>${hardBlockCategories.has(code) ? "block" : "review"}</em>
  `;
  taxonomyGrid.append(item);
});

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Healthcheck failed");
    healthLed.classList.add("ok");
    healthLed.classList.remove("error");
    healthLabel.textContent = "API online";
  } catch {
    healthLed.classList.add("error");
    healthLed.classList.remove("ok");
    healthLabel.textContent = "API offline";
  }
}

function getStage() {
  return document.querySelector("input[name='stage']:checked").value;
}

function setStage(stage) {
  document.querySelector(`input[name='stage'][value='${stage}']`).checked = true;
}

function updateMeta() {
  const hasPrompt = promptInput.value.trim().length > 0;
  const parts = [scenarioInput.value, getStage()];
  if (hasPrompt) parts.push(`${promptInput.value.trim().length} симв.`);
  if (imageName) parts.push(imageName);
  requestMeta.textContent = parts.join(" · ");
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.querySelector("span").textContent = isBusy ? "Проверяем" : "Проверить";
}

function colorForVerdict(verdict) {
  if (verdict === "block") return "var(--red)";
  if (verdict === "review") return "var(--yellow)";
  return "var(--green)";
}

function labelForCategory(category) {
  return categoryLabels[category] || category;
}

function labelForSignal(name) {
  return signalLabels[name] || name;
}

function labelForStatus(status) {
  return statusLabels[status] || status;
}

function labelForStage(stage) {
  return stageLabels[stage] || stage;
}

function translateReason(reason) {
  const map = {
    "Blocked by image guardrail due to unsafe content.": "Заблокировано: сработала политика безопасности контента.",
    "Request requires secondary review due to medium-confidence policy signals.": "Нужна ручная проверка: есть сигналы средней уверенности.",
    "No blocking policy signals exceeded review thresholds.": "Блокирующие сигналы не превысили пороги review/block.",
    "Blocked by UI policy guard due to prohibited topic.": "Заблокировано UI-фильтром: запрос содержит запрещенную тему.",
    "UI policy guard matched prohibited topic before API request.": "UI-фильтр нашел запрещенную тему до отправки запроса в API.",
    "Heuristic text guard matched policy keywords in prompt or OCR text.": "Текстовый фильтр нашел совпадение с правилами политики.",
    "Heuristic text guard found no policy keyword matches.": "Текстовый фильтр не нашел нарушений.",
    "Heuristic policy judge fused available sensor evidence.": "Арбитр политики объединил найденные сигналы.",
    "No prompt text supplied.": "Текст промпта не передан.",
    "Placeholder adapter: prompt text is treated as safe in MVP.": "Текстовый фильтр сейчас работает как заглушка и считает промпт безопасным.",
    "ShieldGemma judge disabled by configuration.": "Мультимодальный арбитр отключен в конфигурации.",
    "No readable text found in image.": "OCR не нашел читаемый текст на изображении.",
    "Extracted text from image via OCR.": "OCR извлек текст с изображения.",
    "transformers is not installed.": "ML-бэкенд не установлен, сенсор пропущен.",
  };
  return map[reason] || reason || "Пояснение отсутствует.";
}

function formatSourceList(sources) {
  return sources.map(labelForSignal).join(", ");
}

function activatePanel(name, shouldScroll = true) {
  guardrailNav?.classList.remove("active");
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    const isActive = panel.dataset.panel === name;
    panel.hidden = !isActive;
    panel.classList.toggle("active", isActive);
  });
  document.querySelectorAll("[data-panel-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.panelTab === name);
  });
  if (name === "taxonomy") {
    document.querySelector("#taxonomyDetails").open = true;
  }
  if (shouldScroll) {
    document.querySelector("#inspector").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderCategories(categories) {
  categoryList.innerHTML = "";
  if (!categories.length) {
    const pill = document.createElement("span");
    pill.className = "category-pill";
    pill.textContent = "Категории не сработали";
    categoryList.append(pill);
    return;
  }

  categories.forEach((category) => {
    const pill = document.createElement("span");
    pill.className = "category-pill";
    pill.textContent = labelForCategory(category);
    categoryList.append(pill);
  });
}

function renderEvidence(response) {
  evidenceList.innerHTML = "";
  const entries = Object.entries(response.evidence || {});
  if (!entries.length) {
    const row = document.createElement("div");
    row.className = "evidence-row";
    row.innerHTML = `<strong>Нет блокирующих сигналов</strong><span>${labelForStage(response.stage)}</span>`;
    evidenceList.append(row);
    return;
  }

  entries.forEach(([category, sources]) => {
    const row = document.createElement("div");
    row.className = "evidence-row";
    row.dataset.sources = formatSourceList(sources);
    row.innerHTML = `<strong>${labelForCategory(category)}</strong><span>${labelForStage(response.stage)}</span>`;
    evidenceList.append(row);
  });
}

function renderSignals(signals, notes) {
  signalTable.innerHTML = "";
  signalCount.textContent = String(signals.length);
  const okCount = signals.filter((signal) => signal.status === "ok").length;
  const issueCount = signals.filter((signal) => Object.keys(signal.categories || {}).length > 0).length;
  signalSummary.textContent = signals.length
    ? `${okCount} из ${signals.length} сенсоров отработали, ${issueCount} подняли категории.`
    : "Запустите проверку, чтобы увидеть работу сенсоров.";

  signals.forEach((signal) => {
    const categories = Object.entries(signal.categories || {})
      .filter(([, score]) => score > 0)
      .sort((a, b) => b[1] - a[1])
      .map(([category, score]) => `${labelForCategory(category)} ${Math.round(score * 100)}%`);

    const detail = categories.length ? categories.join(" · ") : translateReason(signal.reason);
    const row = document.createElement("div");
    row.className = "signal-row";
    row.innerHTML = `
      <strong>${labelForSignal(signal.name)}</strong>
      <span>${detail}</span>
      <span class="status-badge ${signal.status}">${labelForStatus(signal.status)}</span>
    `;
    signalTable.append(row);
  });

  (notes || []).forEach((note) => {
    const row = document.createElement("div");
    row.className = "signal-row";
    row.innerHTML = `<strong>Примечание</strong><span>${note}</span><span class="status-badge skipped">info</span>`;
    signalTable.append(row);
  });
}

function renderAudit(response) {
  auditLog.innerHTML = "";
  auditCount.textContent = "1";
  const categories = response.categories?.length
    ? response.categories.map(labelForCategory).join(", ")
    : "нарушения не найдены";
  auditSummary.textContent = `Последняя проверка: ${verdictLabels[response.verdict] || response.verdict}, ${Math.round(response.confidence * 100)}% уверенности.`;

  const rows = [
    ["ID запроса", response.request_id],
    ["Сценарий", response.scenario],
    ["Этап", labelForStage(response.stage)],
    ["Вердикт", verdictLabels[response.verdict] || response.verdict],
    ["Категории", categories],
    ["Промпт", lastPayload?.prompt || "не передан"],
    ["Изображение", lastPayload?.metadata?.image_name || "не передано"],
  ];

  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "audit-row";
    row.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    auditLog.append(row);
  });
}

function renderResult(response) {
  emptyState.hidden = true;
  resultContent.hidden = false;

  const percent = Math.round((response.confidence || 0) * 100);
  const color = colorForVerdict(response.verdict);
  verdictLabel.textContent = response.verdict;
  verdictLabel.style.color = color;
  confidenceValue.textContent = `${percent}%`;
  confidenceRing.style.background = `conic-gradient(${color} ${percent * 3.6}deg, #3d3d40 0deg)`;
  reasonText.textContent = translateReason(response.reason);
  requestId.textContent = response.request_id.slice(0, 8);

  renderCategories(response.categories || []);
  renderEvidence(response);
  renderSignals(response.signals || [], response.notes || []);
  renderAudit(response);
}

function toast(message) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();
  const node = document.createElement("div");
  node.className = "toast";
  node.textContent = message;
  document.body.append(node);
  setTimeout(() => node.remove(), 4200);
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      resolve(result.includes(",") ? result.split(",")[1] : result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    toast("Нужен файл изображения");
    return;
  }
  imageBase64 = await fileToBase64(file);
  imageName = file.name;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropCopy.hidden = true;
  updateMeta();
}

imageInput.addEventListener("change", (event) => handleFile(event.target.files[0]));

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  handleFile(event.dataTransfer.files[0]);
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    const example = examples[button.dataset.example];
    scenarioInput.value = example.scenario;
    setStage(example.stage);
    promptInput.value = example.prompt;
    updateMeta();
  });
});

[promptInput, scenarioInput, ...document.querySelectorAll("input[name='stage']")].forEach((control) => {
  control.addEventListener("input", updateMeta);
  control.addEventListener("change", updateMeta);
});

resetButton.addEventListener("click", () => {
  form.reset();
  imageBase64 = "";
  imageName = "";
  preview.hidden = true;
  preview.removeAttribute("src");
  dropCopy.hidden = false;
  emptyState.hidden = false;
  resultContent.hidden = true;
  signalTable.innerHTML = "";
  signalCount.textContent = "0";
  signalSummary.textContent = "Запустите проверку, чтобы увидеть работу сенсоров.";
  auditLog.innerHTML = "";
  auditCount.textContent = "0";
  auditSummary.textContent = "Пока нет запросов. После проверки здесь появится журнал решения.";
  lastPayload = null;
  updateMeta();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt && !imageBase64) {
    toast("Добавьте промпт или изображение");
    return;
  }

  const payload = {
    scenario: scenarioInput.value,
    stage: getStage(),
    prompt: prompt || null,
    image_base64: imageBase64 || null,
    metadata: {
      source: "frontend_prototype",
      image_name: imageName || null,
    },
  };
  lastPayload = payload;

  const policyMatch = findForbiddenPromptTopics(prompt);
  if (policyMatch) {
    renderResult(buildUiBlockedResponse(payload, policyMatch));
    activatePanel("signals", false);
    toast("Запрос заблокирован UI-фильтром запрещенных тем");
    return;
  }

  try {
    setBusy(true);
    const response = await fetch("/v1/moderate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Moderation request failed");
    }
    renderResult(data);
  } catch (error) {
    toast(error.message || "Не удалось выполнить проверку");
  } finally {
    setBusy(false);
  }
});

checkHealth();
updateMeta();

document.querySelectorAll("[data-panel-tab]").forEach((button) => {
  button.addEventListener("click", () => activatePanel(button.dataset.panelTab));
});

guardrailNav?.addEventListener("click", () => {
  guardrailNav.classList.add("active");
  document.querySelectorAll("[data-panel-tab]").forEach((button) => button.classList.remove("active"));
});
