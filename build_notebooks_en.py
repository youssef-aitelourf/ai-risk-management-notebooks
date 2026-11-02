"""Generate English versions of the AI risk management notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import build_notebooks as fr

OUT = Path("notebooks_en")


def markdown(text: str) -> dict:
    return fr.md(text)


LAB1_MARKDOWN = [
    """# Lab 1 - Identifying risks in an AI system

**Objective.** Build a small income-prediction model using the Adult dataset, then assess data, performance, and fairness risks. This notebook follows the original lab flow while adding the controls required for a reproducible and verifiable result.

**Teaching note.** The `sex` and `race` columns are used only to measure differences between groups. They must not be used to make an automated decision about an individual. A fairness metric alone does not prove that a system is fair: it starts a business, legal, and technical investigation.

**Prerequisite.** First run `pip install -r requirements.txt` from the project root. Every cell labelled **Test** must finish without an error.

## How to read this notebook

Every section follows the same cycle: **objective → code → test → interpretation**. Code cells perform the operation; **Test** cells use `assert` and immediately stop the notebook if a result is inconsistent. An `OK` message means that the stated technical check passed - not that the system is automatically ready for deployment.

Run the cells from top to bottom so that variables created in one step are available to the next. Tables, metrics, and figures must then be interpreted, rather than simply copied into a report.""",
    """## 1. Load and explore the data

The Adult dataset has no header row. We therefore provide the column list and convert `?` to a missing value (`NaN`). This conversion matters: otherwise an absence becomes a normal category and can silently influence the model.

**What the code does, line by line:** `url` identifies the source; `columns` names each position in the file; `read_csv` creates a Pandas table; `header=None` confirms there is no title row; `na_values="?"` converts missing-value markers; and `skipinitialspace=True` removes spaces after commas. `head()` shows only five rows, so the next Test cell validates the dataset more thoroughly.""",
    """**Interpretation.** Missing values are a data risk: their frequency can differ between groups and distort representation. The target distribution reveals an imbalance, so high accuracy could still conceal weak detection of the `>50K` class.

## 2. Prepare features without data leakage

Data leakage occurs when a transformation learns statistics from the test set. The `Pipeline` below fits imputation and encoding only on training data. Sensitive attributes are retained separately for the fairness diagnostic, but excluded from the training features in this demonstration.

**Process:** first copy the selected columns so the source table is unchanged; convert the target to `0` / `1`; use `train_test_split` to reserve 20% of records for evaluation; and apply `stratify=y` to preserve a comparable high-income proportion in both datasets. Numeric variables receive their median and categorical variables their most common value, then are encoded. Since both transformations are inside the pipeline, they never learn from the test data.""",
    """## 3. Train and evaluate the model

The random forest is used as a baseline. `random_state` makes the experiment repeatable. We inspect the classification report, AUC, and confusion matrix because no single metric captures every risk.

**How to read the output:** the report provides precision, recall, and F1 by class. Recall for `>50K` answers: “among truly positive cases, how many are detected?” The confusion matrix shows false positives and false negatives, whose cost depends on the use case. AUC measures how well examples are ranked across thresholds; it says nothing by itself about fairness or compliance.""",
    """## 4. Diagnose a selection-rate gap by group

The **selection rate** is the share of positive predictions. The parity ratio compares that rate with the reference group. A ratio below 0.80 is often used as an alert signal (the four-fifths rule), but it is neither a universal standard nor a legal verdict.

**Reading the table:** `observations` is the sample size for a group; a small group makes comparisons fragile. `selection_rate` is the predicted positive rate, while `actual_positive_rate` describes the historical label - which can itself contain bias. The ratio is calculated against the highest observed rate: `1` means equality with the computed reference group. A gap calls for investigation of data, criteria, and context, never an automatic conclusion.""",
    """## 5. Create a risk register

Priority is calculated here as `severity × likelihood` (both from 1 to 5). It is a prioritisation tool, not a scientific measurement. Every risk needs an owner, a control, and evidence of follow-up.

**Register steps:** each row identifies one risk; severity represents its potential consequence and likelihood its estimated probability; `priorite` orders subjects to address. The `controle` column describes a preventive or detective action, while `responsable` assigns accountability. The exported CSV can form the basis of a governance review if assessments are dated and evidence is attached.""",
    """## Conclusion and reflection questions

1. Which risk should be addressed first, and what evidence would show that its control works?
2. Beyond the code, which social or historical factors could explain an observed gap?
3. Who can challenge a decision, and within what timeframe?

Before deployment, complete the analysis with an impact assessment, legal validation, and review involving affected people.""",
]

LAB2_MARKDOWN = [
    """# Lab 2 - Auditing and fixing an AI compliance plan

**Objective.** Detect gaps in a fictional compliance plan, link each gap to a corrective action, and generate a traceable Markdown plan.

The control below is deliberately simple and rule-based. In a real setting, rules must be validated by legal, security, data, and business teams; they do not replace regulatory analysis.

## Working method

The notebook starts from a flawed plan, applies transparent detection rules, then creates a corrective document. For every finding, check three things: does the finding match the text? does the correction address the real cause? what evidence will demonstrate that the control was applied? The tests validate the program's behaviour, not an organisation's effective legal compliance.""",
    """## 1. Define a deliberately incomplete plan

Each item is a statement to audit. The examples use the lab's themes: consent, bias testing, and logging. All data are fictional.

**Why start with this format?** A list makes each statement individually testable. In a real audit, it would come from procedures, contracts, impact assessments, and technical evidence. The three sentences are deliberately negative: they are the inputs that the mechanism must detect.""",
    """## 2. Detect non-compliance using explicit rules

A rule must be readable and testable. The function returns a list of findings instead of a simple Boolean, making the audit explainable and traceable.

**Function flow:** `RULES` links every topic to a search pattern, a domain, a severity, and a correction. `audit_plan` loops through each statement and then each rule; when a pattern matches, it creates a finding dictionary. The output therefore retains both the original text and the recommendation. Regular expressions suit this example; a real audit would use controlled vocabulary and human review of false positives and false negatives.""",
    """## 3. Generate the corrected plan

The document links a finding to its domain, severity, and recommended control. Its date is generated automatically so that an example date is not mistaken for real information.

**Production steps:** `render_plan` converts the finding list into Markdown; each action receives a title and structured fields; `write_text` saves the result in UTF-8 to preserve accents; and `read_text` displays it for immediate review. The Test cell then opens the actual saved file and verifies its heading, findings, and corrections - it does not merely test the in-memory Python object.""",
    """## 4. Limits and next audit steps

The tests confirm that the detection mechanism and document are consistent. They do not show that the controls are applied in practice. The next step is to assign a due date, collect evidence, control access to logs, and have the plan approved by the appropriate owners.""",
]

LAB3_MARKDOWN = [
    """# Lab 3 - Building a complete AI risk strategy

**Scenario.** SmartHire is a fictional CV-screening tool. It has a gender-related ranking gap, limited explainability, and no logging or monitoring.

**Objective.** Produce a strategy structured across four domains, export it to YAML, and verify that the documentation and visualisation are created.

## How to use this notebook

A risk strategy is not a static list: it connects a scenario, controls, owners, and evidence. This notebook shows how to structure and share that list. Run cells in order: the strategy is defined, validated, exported, reloaded, and then visualised. **Test** cells confirm that each step produced its expected result.""",
    """## 1. Define the scope and controls

Controls are organised by domain so that AI risk is not reduced to a purely technical issue. This list is a starting point: every control needs an owner, date, acceptance criterion, and evidence.

**Reading the structure:** the dictionary uses a domain as its key and a control list as its value. `Technique` covers model behaviour; `Opérationnel` covers how the system runs over time; `Éthique` covers effects on people; and `Juridique` covers applicable obligations. The final loop prints the items before export. The test requires four domains and at least three non-empty controls in each: this is a minimum completeness check.""",
    """## 2. Export readable YAML documentation

YAML is readable by people and usable by software. `safe_dump` prevents Python-specific types from being emitted; `allow_unicode=True` preserves French accents. Reloading after writing is critical: it verifies the delivered artefact, not only the in-memory object.

**Flow:** `document` separates metadata from the business strategy; `safe_dump` converts the dictionary to YAML; `write_text` saves the file; and the Test cell uses `safe_load` to reload it and compare the reloaded strategy with the original. If they differ, execution stops, exposing a serialisation or path problem.""",
    """## 3. Visualise control coverage

The chart counts controls; it does not measure their effectiveness. A domain with more controls is not necessarily better covered: evidence quality and observed effectiveness remain decisive.

**How the chart is made:** `domains` preserves the domain order, `counts` calculates the length of each list, `bar` draws one bar per domain, and the loop places the count above each bar. `savefig` creates an image that can be reused in a report. The test compares displayed counts with the source lists and confirms the image exists and is non-empty. Interpret differences afterwards: a small number may signal a gap or simply broader controls, so effectiveness must be examined.""",
    """## Final reflection

What evidence would you request to judge whether a control is effective? Consider metrics, logs, human reviews, and candidate feedback. A strategy becomes operational only when its controls are measured, followed up, and improved over time.""",
]


def translate_code(source: str) -> str:
    # The compliance lab contains matching input phrases, regular expressions,
    # and assertions. Translate these interdependent cells as complete units,
    # rather than applying fragile word-by-word substitutions.
    if source.lstrip().startswith("broken_checks = ["):
        return '''broken_checks = [
    "Data is collected without user consent.",
    "Bias testing is skipped because of time constraints.",
    "No predictions are logged.",
]
for item in broken_checks:
    print(f"[!] {item}")
'''
    if "scénario couvre les trois familles" in source:
        return '''# Test - the scenario covers the three expected gap categories.
assert len(broken_checks) == 3
text = " ".join(broken_checks).lower()
assert all(token in text for token in ("consent", "bias", "logged"))
print("OK - audit scenario initialised.")
'''
    if source.lstrip().startswith("RULES = {"):
        return '''RULES = {
    "consent": {"pattern": r"without user consent", "domain": "Legal", "severity": "High", "correction": "Obtain and retain evidence of valid consent before collection."},
    "fairness": {"pattern": r"bias testing.*skipped", "domain": "Ethical", "severity": "High", "correction": "Plan fairness metrics, thresholds, and an independent review before production."},
    "traceability": {"pattern": r"no predictions are logged", "domain": "Operational", "severity": "High", "correction": "Log versions, decisions, and audit-relevant events using data minimisation."},
}

def audit_plan(items, rules=RULES):
    findings = []
    for item in items:
        for rule_id, rule in rules.items():
            if re.search(rule["pattern"], item, flags=re.IGNORECASE):
                findings.append({"rule": rule_id, "finding": item, **{key: value for key, value in rule.items() if key != "pattern"}})
    return findings

findings = audit_plan(broken_checks)
findings
'''
    if "chaque lacune déclenche exactement" in source:
        return '''# Test - every gap triggers exactly one finding with a correction.
assert len(findings) == len(broken_checks)
assert {finding["rule"] for finding in findings} == set(RULES)
assert all(finding["correction"] and finding["domain"] for finding in findings)
print("OK - all expected rules were detected.")
'''
    if source.lstrip().startswith("def render_plan("):
        return '''def render_plan(findings, owner="AI Compliance Team"):
    lines = ["# Revised AI Compliance Plan", "", "## Metadata", f"- **Owner:** {owner}", f"- **Generation date:** {date.today().isoformat()}", "- **Status:** Pending validation by the relevant functions", ""]
    for number, finding in enumerate(findings, start=1):
        lines += [f"## Action {number} - {finding['rule'].title()}", f"- **Finding:** {finding['finding']}", f"- **Domain:** {finding['domain']}", f"- **Severity:** {finding['severity']}", f"- **Recommended control:** {finding['correction']}", "- **Expected evidence:** approved procedure, control result, and named owner.", ""]
    return "\\n".join(lines)

plan = render_plan(findings)
output = ARTIFACTS / "lab2_revised_compliance_plan.md"
output.write_text(plan, encoding="utf-8")
print(output.read_text(encoding="utf-8"))
'''
    if "l'artefact est créé" in source:
        return '''# Test - the artefact is created, non-empty, and contains every finding and correction.
assert output.exists() and output.stat().st_size > 300
saved = output.read_text(encoding="utf-8")
assert saved.startswith("# Revised AI Compliance Plan")
for finding in findings:
    assert finding["finding"] in saved
    assert finding["correction"] in saved
print("OK - complete, traceable Markdown plan.")
'''
    replacements = {
        'PROJECT_ROOT / "notebooks" / "artifacts"': 'PROJECT_ROOT / "notebooks_en" / "artifacts"',
        '# Test - ': '# Test - ',
        'Test - intégrité minimale de la source chargée.': 'Test - minimum integrity of the loaded source.',
        'Test - séparation exhaustive, sans recouvrement d\'indices, et cible stratifiée.': 'Test - complete split, no index overlap, and stratified target.',
        'Test - le pipeline est entraîné et retourne des prédictions binaires alignées.': 'Test - trained pipeline returns aligned binary predictions.',
        'Test - les taux et ratios sont bornés ; tous les groupes observés sont reportés.': 'Test - rates and ratios are bounded; every observed group is reported.',
        'Test - registre complet et formule de priorité correcte.': 'Test - complete register and correct priority formula.',
        'Test - le scénario couvre les trois familles de lacunes attendues.': 'Test - scenario covers the three expected gap categories.',
        'Test - chaque lacune déclenche exactement un constat, avec une correction.': 'Test - every gap triggers exactly one finding with a correction.',
        'Test - l\'artefact est créé, non vide, et contient chaque constat et correction.': 'Test - the artefact is created, non-empty, and contains every finding and correction.',
        'Test - stratégie équilibrée : quatre domaines, trois contrôles non vides dans chacun.': 'Test - balanced strategy: four domains and three non-empty controls in each.',
        'Test - relecture YAML et égalité du contenu métier.': 'Test - YAML reload and business-content equality.',
        'Test - les comptes reflètent exactement la stratégie et l\'image a été écrite.': 'Test - counts match the strategy and the image was written.',
        'OK - ': 'OK - ',
        'lignes et': 'rows and',
        'colonnes chargées.': 'columns loaded.',
        'séparation reproductible et stratifiée.': 'reproducible, stratified split.',
        'modèle entraîné et prédictions valides.': 'trained model and valid predictions.',
        'diagnostic d’équité calculé pour les groupes observés.': 'fairness diagnostic computed for observed groups.',
        "diagnostic d'équité calculé pour les groupes observés.": 'fairness diagnostic computed for observed groups.',
        'registre exporté et validé.': 'risk register exported and validated.',
        'scénario d’audit initialisé.': 'audit scenario initialised.',
        'toutes les règles attendues ont été détectées.': 'all expected rules were detected.',
        'plan Markdown complet et traçable.': 'complete, traceable Markdown plan.',
        'stratégie structurée et complète.': 'structured, complete strategy.',
        'export YAML relu et validé.': 'YAML export reloaded and validated.',
        'visualisation créée et validée.': 'visualisation created and validated.',
        'diagnostic d’équité calculé pour les groupes observés.': 'fairness diagnostic computed for observed groups.',
        'Valeurs manquantes et biais de représentation': 'Missing values and representation bias',
        'Écart de taux de sélection entre groupes': 'Selection-rate gap between groups',
        'Dérive des données et des performances': 'Data and performance drift',
        'Décision insuffisamment explicable': 'Insufficiently explainable decision',
        'Données': 'Data',
        'Éthique': 'Ethical',
        'Opérationnel': 'Operational',
        'Gouvernance': 'Governance',
        'Imputation dans pipeline; suivi des taux par groupe': 'Pipeline imputation; monitor rates by group',
        'Audit d’équité, revue humaine, critères documentés': 'Fairness audit, human review, documented criteria',
        "Audit d'équité, revue humaine, critères documentés": 'Fairness audit, human review, documented criteria',
        'Monitoring, seuils d’alerte, revue périodique': 'Monitoring, alert thresholds, periodic review',
        'Monitoring, seuils d’alerte, revue périodique': 'Monitoring, alert thresholds, periodic review',
        'Motifs compréhensibles, procédure de recours': 'Understandable reasons, appeal procedure',
        'gravite': 'severity',
        'vraisemblance': 'likelihood',
        'priorite': 'priority',
        'categorie': 'category',
        'controle': 'control',
        'responsable': 'owner',
        'Les données sont collectées sans consentement de l’utilisateur.': 'Data is collected without user consent.',
        "Les données sont collectées sans consentement de l'utilisateur.": 'Data is collected without user consent.',
        'Les tests de biais sont ignorés faute de temps.': 'Bias testing is skipped because of time constraints.',
        'Aucune prédiction n’est journalisée.': 'No predictions are logged.',
        "Aucune prédiction n'est journalisée.": 'No predictions are logged.',
        'consentement': 'consent',
        'equite': 'fairness',
        'tracabilite': 'traceability',
        'sans consentement': 'without consent',
        'tests? de biais.*ignor': 'bias testing.*skipped',
        'aucune prédiction n’est journalisée': 'no predictions are logged',
        "aucune prédiction n'est journalisée": 'no predictions are logged',
        '("consentement", "biais", "journalisée")': '("consent", "bias", "logged")',
        'domaine': 'domain',
        'correction': 'correction',
        'constat': 'finding',
        'regle': 'rule',
        'Élevée': 'High',
        'Obtenir et conserver une preuve de consentement valide avant la collecte.': 'Obtain and retain evidence of valid consent before collection.',
        'Planifier des métriques d’équité, des seuils et une revue indépendante avant mise en production.': 'Plan fairness metrics, thresholds, and an independent review before production.',
        'Journaliser de manière minimisée les versions, décisions et événements utiles à l’audit.': 'Log versions, decisions, and audit-relevant events using data minimisation.',
        "Journaliser de manière minimisée les versions, décisions et événements utiles à l'audit.": 'Log versions, decisions, and audit-relevant events using data minimisation.',
        'Équipe conformité IA': 'AI Compliance Team',
        '# Plan de conformité IA révisé': '# Revised AI Compliance Plan',
        '## Métadonnées': '## Metadata',
        'Responsable': 'Owner',
        'Date de génération': 'Generation date',
        'Statut': 'Status',
        'À valider par les fonctions compétentes': 'Pending validation by the relevant functions',
        'Action': 'Action',
        'Constat': 'Finding',
        'Domaine': 'Domain',
        'Gravité': 'Severity',
        'Contrôle recommandé': 'Recommended control',
        'Preuve attendue': 'Expected evidence',
        'procédure approuvée, résultat de contrôle et responsable désigné.': 'approved procedure, control result, and named owner.',
        'Technique': 'Technical',
        'Mesurer les écarts d’équité avant chaque version': 'Measure fairness gaps before every release',
        "Mesurer les écarts d'équité avant chaque version": 'Measure fairness gaps before every release',
        'Surveiller dérive et performance après déploiement': 'Monitor drift and performance after deployment',
        'Fournir des raisons de classement compréhensibles': 'Provide understandable ranking reasons',
        'Versionner modèle, données et paramètres': 'Version the model, data, and parameters',
        'Conserver la lignée des données et les changements': 'Maintain data lineage and change tracking',
        'Tester un plan de réponse aux incidents IA': 'Test an AI incident-response plan',
        'Auditer les biais sur des populations pertinentes': 'Audit bias across relevant populations',
        'Définir des critères d’équité avec les parties prenantes': 'Define fairness criteria with stakeholders',
        "Définir des critères d'équité avec les parties prenantes": 'Define fairness criteria with stakeholders',
        'Prévoir une revue humaine et une voie de recours': 'Provide human review and an appeal route',
        'Juridique': 'Legal',
        'Documenter la base légale et le consentement applicable': 'Document the legal basis and applicable consent',
        'Évaluer les décisions automatisées à effet significatif': 'Assess automated decisions with significant effects',
        'Valider les obligations de transparence et de non-discrimination': 'Validate transparency and non-discrimination obligations',
        'contrôles': 'controls',
        'projet': 'project',
        'stratégie de risque IA': 'AI risk strategy',
        'statut': 'status',
        'Brouillon à valider': 'Draft pending validation',
        'Écart de classement, explicabilité insuffisante et absence de surveillance.': 'Ranking gap, insufficient explainability, and no monitoring.',
        'strategie_risque': 'risk_strategy',
        'Couverture de la stratégie de risque par domaine': 'AI risk strategy coverage by domain',
        'Nombre de contrôles': 'Number of controls',
    }
    # Translate complete sentences before individual keys such as
    # "consentement"; otherwise a partial replacement prevents the
    # corresponding full-sentence rule from ever matching.
    for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(old, new)
    return source


def localise(cells: list[dict], translations: list[str]) -> list[dict]:
    result = []
    position = 0
    for cell in cells:
        if cell["cell_type"] == "markdown":
            result.append(markdown(translations[position]))
            position += 1
        else:
            cell = dict(cell)
            cell["source"] = translate_code("".join(cell["source"])).splitlines(keepends=True)
            result.append(cell)
    assert position == len(translations)
    return result


def main() -> None:
    OUT.mkdir(exist_ok=True)
    notebooks = [
        ("01_lab_ai_system_risks.ipynb", fr.lab1, LAB1_MARKDOWN),
        ("02_lab_ai_compliance_audit.ipynb", fr.lab2, LAB2_MARKDOWN),
        ("03_lab_ai_risk_strategy.ipynb", fr.lab3, LAB3_MARKDOWN),
    ]
    for filename, cells, translations in notebooks:
        content = fr.notebook(localise(cells, translations))
        (OUT / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
