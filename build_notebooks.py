"""Generate the French, executable Jupyter notebooks for the AI risk labs."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from textwrap import dedent

OUT = Path("notebooks")
ARTIFACTS = OUT / "artifacts"


def md(text: str) -> dict:
    # Les chaînes de ce générateur sont indentées dans le code Python.
    # On retire l'indentation résiduelle ligne par ligne afin qu'elle ne soit
    # jamais interprétée comme un bloc de code Markdown dans Jupyter.
    normalized = "\n".join(line.lstrip() for line in dedent(text).strip().splitlines())
    source = normalized.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "id": hashlib.sha1("".join(source).encode()).hexdigest()[:8], "source": source}


def code(text: str) -> dict:
    source = dedent(text).strip().splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "id": hashlib.sha1("".join(source).encode()).hexdigest()[:8], "source": source}


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


lab1 = [
    md("""
    # Lab 1 - Identifier les risques d'un système d'IA

    **Objectif.** Construire un petit modèle de prédiction de revenu à partir du jeu de données Adult, puis évaluer ses risques de données, de performance et d'équité. Ce notebook reprend le fil du lab, mais ajoute les contrôles nécessaires pour que le résultat soit vérifiable et réutilisable.

    **Attention pédagogique.** Les colonnes `sex` et `race` sont utilisées ici uniquement pour mesurer des écarts entre groupes. Elles ne doivent pas être employées pour décider automatiquement du sort d'une personne. Une métrique d'équité ne prouve pas, à elle seule, qu'un système est équitable : elle déclenche une investigation métier, juridique et technique.

    **Pré-requis.** Exécuter d'abord `pip install -r requirements.txt` à la racine du projet. Toutes les cellules marquées **Test** doivent finir sans erreur.

    ## Comment lire ce notebook

    Chaque partie suit le même cycle : **objectif → code → test → interprétation**. Les cellules de code réalisent l'opération ; les cellules **Test** emploient `assert` et stoppent immédiatement le notebook si le résultat est incohérent. Un message `OK` signifie que le contrôle décrit dans la cellule a réussi, pas que le système est automatiquement prêt à être déployé.

    L'ordre est important : exécuter les cellules de haut en bas afin que les variables créées à une étape soient disponibles à l'étape suivante. Les sorties (tableaux, métriques, figures) doivent ensuite être commentées, et non simplement copiées dans un rapport.
    """),
    code("""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier

    PROJECT_ROOT = next((path for path in (Path.cwd(), *Path.cwd().parents) if (path / "requirements.txt").exists()), Path.cwd())
    ARTIFACTS = PROJECT_ROOT / "notebooks" / "artifacts"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RANDOM_STATE = 42
    pd.set_option("display.max_columns", 30)
    """),
    md("""
    ## 1. Charger et explorer les données

    Le jeu Adult n'a pas d'en-tête. On fournit donc la liste des colonnes et on convertit `?` en valeur manquante (`NaN`). Cette conversion est importante : sinon une absence devient une modalité ordinaire et peut influencer silencieusement le modèle.

    **Ce que fait le code, ligne par ligne :** `url` indique l'emplacement de la source ; `columns` donne un nom à chaque position du fichier ; `read_csv` crée le tableau Pandas ; `header=None` confirme l'absence de titre dans le fichier ; `na_values="?"` transforme les marqueurs d'absence ; `skipinitialspace=True` élimine les espaces après les virgules. `head()` affiche seulement cinq lignes : ce n'est pas une validation complète, d'où la cellule de test suivante.
    """),
    code("""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    data = pd.read_csv(url, header=None, names=columns, na_values="?", skipinitialspace=True)
    data.head()
    """),
    code("""
    # Test - intégrité minimale de la source chargée.
    assert data.shape[0] > 30_000, "Le jeu Adult attendu contient plus de 30 000 lignes."
    assert list(data.columns) == columns
    assert set(data["income"].dropna().unique()) == {"<=50K", ">50K"}
    print(f"OK - {len(data):,} lignes et {len(columns)} colonnes chargées.")
    """),
    code("""
    missing = data.isna().sum().sort_values(ascending=False)
    display(missing[missing.gt(0)].to_frame("valeurs_manquantes"))
    display(data["income"].value_counts(normalize=True).rename("proportion"))
    """),
    md("""
    **Interprétation.** Les valeurs manquantes sont un risque de données : leur fréquence peut différer selon les groupes et altérer la représentation. La distribution de la cible signale ici un déséquilibre : une exactitude élevée pourrait masquer une mauvaise détection de la classe `>50K`.

    ## 2. Préparer les variables sans fuite de données

    La fuite de données apparaît lorsqu'une transformation apprend des statistiques sur le jeu de test. La `Pipeline` ci-dessous ajuste imputation et encodage uniquement sur l'entraînement. Les attributs sensibles sont conservés séparément pour le diagnostic d'équité, mais exclus des variables d'entraînement dans cette démonstration.

    **Déroulé :** on copie d'abord les colonnes retenues pour ne pas modifier la table source ; la cible est convertie en `0` / `1` ; `train_test_split` réserve 20 % des données pour l'évaluation ; `stratify=y` conserve une proportion comparable de revenus élevés dans les deux jeux. Les variables numériques reçoivent leur médiane et les variables catégorielles leur modalité la plus fréquente, puis sont encodées. Ces deux transformations vivent dans le pipeline : elles n'apprennent donc jamais sur le test.
    """),
    code("""
    features = ["age", "education_num", "hours_per_week", "workclass", "occupation", "marital_status"]
    sensitive = ["sex", "race"]
    target = "income"
    model_data = data[features + sensitive + [target]].copy()
    model_data[target] = model_data[target].eq(">50K").astype(int)

    X = model_data[features]
    y = model_data[target]
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, model_data[sensitive], test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    numeric = ["age", "education_num", "hours_per_week"]
    categorical = [c for c in features if c not in numeric]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    """),
    code("""
    # Test - séparation exhaustive, sans recouvrement d'indices, et cible stratifiée.
    assert len(X_train) + len(X_test) == len(model_data)
    assert X_train.index.intersection(X_test.index).empty
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert set(groups_test.columns) == set(sensitive)
    print("OK - séparation reproductible et stratifiée.")
    """),
    md("""
    ## 3. Entraîner et évaluer le modèle

    La forêt aléatoire sert de modèle de référence. `random_state` rend l'expérience répétable. On consulte le rapport de classification, l'AUC et la matrice de confusion : aucune métrique seule ne résume tous les risques.

    **Comment lire la sortie :** le rapport fournit précision, rappel et F1 par classe. Le rappel de `>50K` répond à « parmi les cas réellement positifs, combien sont détectés ? ». La matrice de confusion montre les faux positifs et faux négatifs, dont le coût dépend du cas d'usage. L'AUC mesure la capacité à ordonner les exemples sur tous les seuils possibles ; elle n'indique ni l'équité ni la conformité.
    """),
    code("""
    model = Pipeline([
        ("prepare", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=150, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, predictions, digits=3))
    print(f"AUC ROC : {roc_auc_score(y_test, probabilities):.3f}")
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions)).plot(cmap="Blues")
    plt.title("Matrice de confusion - jeu de test")
    plt.show()
    """),
    code("""
    # Test - le pipeline est entraîné et retourne des prédictions binaires alignées.
    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert 0.0 <= roc_auc_score(y_test, probabilities) <= 1.0
    print("OK - modèle entraîné et prédictions valides.")
    """),
    md("""
    ## 4. Diagnostiquer un écart de sélection par groupe

    Le **taux de sélection** est la proportion de prédictions positives. Le ratio de parité compare ce taux à celui du groupe de référence. Un ratio inférieur à 0,80 est souvent utilisé comme signal d'alerte (règle des quatre cinquièmes), mais ce seuil n'est ni une norme universelle ni un verdict légal.

    **Lecture du tableau :** `observations` indique la taille de l'échantillon par groupe ; un groupe trop petit rend toute comparaison fragile. `selection_rate` est le taux prédit positif, tandis que `actual_positive_rate` décrit la cible historique - elle peut elle-même porter des biais. Le ratio est calculé par rapport au taux le plus élevé : `1` signifie égalité avec le groupe de référence calculé. Un écart appelle une analyse des données, des critères et du contexte, jamais une conclusion automatique.
    """),
    code("""
    audit = groups_test.copy()
    audit["prediction"] = predictions
    audit["truth"] = y_test
    fairness = audit.groupby("sex").agg(
        observations=("prediction", "size"),
        selection_rate=("prediction", "mean"),
        actual_positive_rate=("truth", "mean"),
    ).sort_index()
    reference_rate = fairness["selection_rate"].max()
    fairness["selection_parity_ratio"] = fairness["selection_rate"] / reference_rate
    display(fairness)
    """),
    code("""
    # Test - les taux et ratios sont bornés ; tous les groupes observés sont reportés.
    assert not fairness.empty
    assert fairness["observations"].gt(0).all()
    assert fairness[["selection_rate", "actual_positive_rate", "selection_parity_ratio"]].apply(lambda s: s.between(0, 1).all()).all()
    print("OK - diagnostic d'équité calculé pour les groupes observés.")
    """),
    md("""
    ## 5. Créer un registre des risques

    La priorité est ici `gravité × vraisemblance` (1 à 5). C'est un outil de tri, pas une mesure scientifique. Pour chaque risque, il faut un responsable, un contrôle et une preuve de suivi.

    **Étapes du registre :** chaque ligne identifie un risque ; la gravité représente la conséquence potentielle, la vraisemblance sa probabilité estimée ; `priorite` ordonne les sujets à traiter. La colonne `controle` décrit l'action préventive ou détective ; `responsable` rend cette action attribuable. Le fichier CSV exporté peut servir de base à une revue de gouvernance, à condition de dater les évaluations et joindre les preuves.
    """),
    code("""
    risk_register = pd.DataFrame([
        ["R-01", "Valeurs manquantes et biais de représentation", "Données", 4, 4, "Imputation dans pipeline; suivi des taux par groupe", "Data owner"],
        ["R-02", "Écart de taux de sélection entre groupes", "Éthique", 5, 4, "Audit d'équité, revue humaine, critères documentés", "Responsible AI lead"],
        ["R-03", "Dérive des données et des performances", "Opérationnel", 4, 3, "Monitoring, seuils d'alerte, revue périodique", "MLOps owner"],
        ["R-04", "Décision insuffisamment explicable", "Gouvernance", 4, 3, "Motifs compréhensibles, procédure de recours", "Product owner"],
    ], columns=["id", "description", "categorie", "gravite", "vraisemblance", "controle", "responsable"])
    risk_register["priorite"] = risk_register["gravite"] * risk_register["vraisemblance"]
    risk_register = risk_register.sort_values("priorite", ascending=False)
    display(risk_register)
    risk_register.to_csv(ARTIFACTS / "lab1_risk_register.csv", index=False)
    """),
    code("""
    # Test - registre complet et formule de priorité correcte.
    required = {"id", "description", "categorie", "gravite", "vraisemblance", "priorite", "controle", "responsable"}
    assert required.issubset(risk_register.columns)
    assert (risk_register["priorite"] == risk_register["gravite"] * risk_register["vraisemblance"]).all()
    assert (ARTIFACTS / "lab1_risk_register.csv").exists()
    print("OK - registre exporté et validé.")
    """),
    md("""
    ## Conclusion et questions de réflexion

    1. Quel risque devez-vous traiter en premier, et quelle preuve démontrerait que le contrôle fonctionne ?
    2. Quels facteurs sociaux ou historiques peuvent expliquer un écart observé, au-delà du code ?
    3. Qui peut contester une décision, et sous quel délai ?

    Avant tout déploiement, compléter l'analyse par une évaluation d'impact, une validation juridique et une revue avec les personnes concernées.
    """),
]

lab2 = [
    md("""
    # Lab 2 - Auditer et corriger un plan de conformité IA

    **Objectif.** Détecter des lacunes dans un plan de conformité fictif, relier chaque lacune à une mesure corrective et générer un plan Markdown traçable.

Le contrôle ci-dessous est volontairement simple et fondé sur des règles. Dans un contexte réel, les règles doivent être validées par les équipes juridique, sécurité, données et métier ; elles ne remplacent pas une analyse réglementaire.

    ## Méthode de travail

    Le notebook part d'un plan défaillant, applique des règles de détection transparentes, puis produit un document de correction. Pour chaque résultat, vérifiez trois éléments : le constat correspond-il au texte ? la correction traite-t-elle réellement la cause ? quelle preuve permettra de démontrer l'application du contrôle ? Les tests vérifient le comportement du programme, pas la conformité juridique effective de l'organisation.
    """),
    code("""
    from pathlib import Path
    from datetime import date
    import re

    PROJECT_ROOT = next((path for path in (Path.cwd(), *Path.cwd().parents) if (path / "requirements.txt").exists()), Path.cwd())
    ARTIFACTS = PROJECT_ROOT / "notebooks" / "artifacts"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    """),
    md("""
    ## 1. Définir un plan volontairement incomplet

Chaque élément correspond à une déclaration à auditer. Les exemples reproduisent les thèmes du lab : consentement, tests de biais et journalisation. Les données utilisées sont fictives.

    **Pourquoi commencer par ce format ?** Une liste rend les affirmations contrôlables une par une. Dans un véritable audit, elle proviendrait de procédures, contrats, analyses d'impact et preuves techniques. Ici, les trois phrases sont négatives exprès : elles constituent les entrées que le mécanisme doit repérer.
    """),
    code("""
    broken_checks = [
        "Les données sont collectées sans consentement de l'utilisateur.",
        "Les tests de biais sont ignorés faute de temps.",
        "Aucune prédiction n'est journalisée.",
    ]
    for item in broken_checks:
        print(f"[!] {item}")
    """),
    code("""
    # Test - le scénario couvre les trois familles de lacunes attendues.
    assert len(broken_checks) == 3
    text = " ".join(broken_checks).lower()
    assert all(token in text for token in ("consentement", "biais", "journalisée"))
    print("OK - scénario d'audit initialisé.")
    """),
    md("""
    ## 2. Détecter les non-conformités avec des règles explicites

Une règle doit être lisible et testable. La fonction renvoie une liste de constats plutôt qu'un simple booléen : cela rend l'audit explicable et facilite la traçabilité.

    **Déroulé de la fonction :** `RULES` associe à chaque sujet une expression recherchée, un domaine, une gravité et une correction. `audit_plan` parcourt chaque déclaration puis chaque règle ; si l'expression correspond, elle crée un dictionnaire de constat. La sortie conserve donc le texte d'origine et la recommandation associée. Les expressions régulières sont adaptées à cet exemple ; un audit réel utiliserait un vocabulaire contrôlé et une revue humaine des faux positifs ou négatifs.
    """),
    code("""
    RULES = {
        "consentement": {"pattern": r"sans consentement", "domaine": "Juridique", "gravite": "Élevée", "correction": "Obtenir et conserver une preuve de consentement valide avant la collecte."},
        "equite": {"pattern": r"tests? de biais.*ignor", "domaine": "Éthique", "gravite": "Élevée", "correction": "Planifier des métriques d'équité, des seuils et une revue indépendante avant mise en production."},
        "tracabilite": {"pattern": r"aucune prédiction n'est journalisée", "domaine": "Opérationnel", "gravite": "Élevée", "correction": "Journaliser de manière minimisée les versions, décisions et événements utiles à l'audit."},
    }

    def audit_plan(items, rules=RULES):
        findings = []
        for item in items:
            for rule_id, rule in rules.items():
                if re.search(rule["pattern"], item, flags=re.IGNORECASE):
                    findings.append({"regle": rule_id, "constat": item, **{k: v for k, v in rule.items() if k != "pattern"}})
        return findings

    findings = audit_plan(broken_checks)
    findings
    """),
    code("""
    # Test - chaque lacune déclenche exactement un constat, avec une correction.
    assert len(findings) == len(broken_checks)
    assert {f["regle"] for f in findings} == set(RULES)
    assert all(f["correction"] and f["domaine"] for f in findings)
    print("OK - toutes les règles attendues ont été détectées.")
    """),
    md("""
    ## 3. Générer le plan corrigé

Le document associe le constat, son domaine, sa gravité et le contrôle recommandé. Sa date est générée automatiquement afin d'éviter qu'une date d'exemple soit prise pour une information réelle.

    **Étapes de production :** `render_plan` transforme la liste de constats en texte Markdown ; chaque action reçoit un titre et des champs structurés ; `write_text` écrit le résultat en UTF-8 afin de préserver les accents ; `read_text` l'affiche pour une relecture immédiate. La cellule de test ouvre ensuite le fichier réellement écrit et vérifie son en-tête, les constats et les corrections - elle ne se limite donc pas à tester l'objet Python en mémoire.
    """),
    code("""
    def render_plan(findings, owner="Équipe conformité IA"):
        lines = ["# Plan de conformité IA révisé", "", "## Métadonnées", f"- **Responsable :** {owner}", f"- **Date de génération :** {date.today().isoformat()}", "- **Statut :** À valider par les fonctions compétentes", ""]
        for number, finding in enumerate(findings, start=1):
            lines += [f"## Action {number} - {finding['regle'].title()}", f"- **Constat :** {finding['constat']}", f"- **Domaine :** {finding['domaine']}", f"- **Gravité :** {finding['gravite']}", f"- **Contrôle recommandé :** {finding['correction']}", "- **Preuve attendue :** procédure approuvée, résultat de contrôle et responsable désigné.", ""]
        return "\\n".join(lines)

    plan = render_plan(findings)
    output = ARTIFACTS / "lab2_plan_conformite_revise.md"
    output.write_text(plan, encoding="utf-8")
    print(output.read_text(encoding="utf-8"))
    """),
    code("""
    # Test - l'artefact est créé, non vide, et contient chaque constat et correction.
    assert output.exists() and output.stat().st_size > 300
    saved = output.read_text(encoding="utf-8")
    assert saved.startswith("# Plan de conformité IA révisé")
    for finding in findings:
        assert finding["constat"] in saved
        assert finding["correction"] in saved
    print("OK - plan Markdown complet et traçable.")
    """),
    md("""
    ## 4. Limites et suite de l'audit

Les tests vérifient que la mécanique de détection et le document sont cohérents. Ils ne démontrent pas que les contrôles sont appliqués en pratique. La suite consiste à attribuer un échéancier, collecter les preuves, contrôler les accès aux journaux et faire valider le plan par les responsables concernés.
    """),
]

lab3 = [
    md("""
    # Lab 3 - Construire une stratégie complète de risque IA

    **Scénario.** SmartHire est un outil fictif de présélection de CV. Il présente un écart de classement selon le genre, une faible explicabilité et l'absence de journalisation et de suivi.

**Objectif.** Produire une stratégie structurée sur quatre domaines, l'exporter en YAML et vérifier que la documentation et la visualisation sont bien créées.

    ## Comment utiliser ce notebook

    Une stratégie de risque n'est pas une liste figée : elle relie un scénario, des contrôles, des responsables et des preuves. Ici, le notebook montre comment structurer et partager cette liste. Exécutez les cellules dans l'ordre : la stratégie est définie, validée, exportée, relue, puis représentée graphiquement. Les cellules **Test** confirment que chaque étape a produit le résultat attendu.
    """),
    code("""
    from pathlib import Path
    from datetime import date
    import yaml
    import matplotlib.pyplot as plt

    PROJECT_ROOT = next((path for path in (Path.cwd(), *Path.cwd().parents) if (path / "requirements.txt").exists()), Path.cwd())
    ARTIFACTS = PROJECT_ROOT / "notebooks" / "artifacts"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    """),
    md("""
    ## 1. Formaliser le périmètre et les contrôles

Les contrôles sont rangés par domaine pour éviter de réduire le risque IA à un problème purement technique. Cette liste est un point de départ : chaque contrôle doit recevoir un responsable, une date, un critère d'acceptation et une preuve.

    **Lecture de la structure :** le dictionnaire utilise un domaine comme clé et une liste de contrôles comme valeur. `Technique` couvre le comportement du modèle ; `Opérationnel` son fonctionnement dans le temps ; `Éthique` les effets sur les personnes ; `Juridique` les obligations applicables. La boucle finale affiche ces éléments pour que la stratégie reste lisible avant tout export. Le test impose quatre domaines et au moins trois contrôles non vides dans chacun : c'est un contrôle de complétude minimale.
    """),
    code("""
    risk_strategy = {
        "Technique": ["Mesurer les écarts d'équité avant chaque version", "Surveiller dérive et performance après déploiement", "Fournir des raisons de classement compréhensibles"],
        "Opérationnel": ["Versionner modèle, données et paramètres", "Conserver la lignée des données et les changements", "Tester un plan de réponse aux incidents IA"],
        "Éthique": ["Auditer les biais sur des populations pertinentes", "Définir des critères d'équité avec les parties prenantes", "Prévoir une revue humaine et une voie de recours"],
        "Juridique": ["Documenter la base légale et le consentement applicable", "Évaluer les décisions automatisées à effet significatif", "Valider les obligations de transparence et de non-discrimination"],
    }
    for domain, controls in risk_strategy.items():
        print(f"\\n{domain.upper()} ({len(controls)} contrôles)")
        for control in controls:
            print(f"- {control}")
    """),
    code("""
    # Test - stratégie équilibrée : quatre domaines, trois contrôles non vides dans chacun.
    assert set(risk_strategy) == {"Technique", "Opérationnel", "Éthique", "Juridique"}
    assert all(len(controls) >= 3 and all(isinstance(c, str) and c for c in controls) for controls in risk_strategy.values())
    print("OK - stratégie structurée et complète.")
    """),
    md("""
    ## 2. Exporter une documentation YAML

YAML est lisible par une personne et exploitable par un outil. `safe_dump` empêche l'émission de types Python spécifiques ; `allow_unicode=True` conserve les accents français. La relecture après écriture est un test crucial : elle vérifie l'artefact réellement livré, pas seulement l'objet en mémoire.

    **Déroulé :** `document` sépare les métadonnées de la stratégie métier ; `safe_dump` convertit ce dictionnaire en YAML ; `write_text` enregistre le fichier ; la cellule de test charge de nouveau le fichier avec `safe_load` et compare la stratégie relue à celle créée en mémoire. Si les deux diffèrent, l'exécution s'arrête : cela détecte un problème de sérialisation ou de chemin de fichier.
    """),
    code("""
    document = {
        "metadata": {
            "projet": "SmartHire - stratégie de risque IA",
            "date_generation": date.today().isoformat(),
            "statut": "Brouillon à valider",
            "scenario": "Écart de classement, explicabilité insuffisante et absence de surveillance.",
        },
        "strategie_risque": risk_strategy,
    }
    yaml_path = ARTIFACTS / "lab3_strategie_risque.yaml"
    yaml_path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(yaml_path.read_text(encoding="utf-8"))
    """),
    code("""
    # Test - relecture YAML et égalité du contenu métier.
    loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert loaded["strategie_risque"] == risk_strategy
    assert loaded["metadata"]["projet"].startswith("SmartHire")
    assert yaml_path.exists()
    print("OK - export YAML relu et validé.")
    """),
    md("""
    ## 3. Visualiser la couverture des contrôles

Le graphique compte les contrôles, il ne mesure pas leur efficacité. Un domaine avec davantage de contrôles n'est pas nécessairement mieux couvert ; la qualité des preuves et l'efficacité observée restent déterminantes.

    **Comment le graphique est construit :** `domains` conserve l'ordre des domaines, `counts` calcule la taille de chaque liste, `bar` trace une barre par domaine et la boucle ajoute le chiffre au-dessus de chaque barre. `savefig` produit une image réutilisable dans un rapport. Le test compare les comptes affichés aux listes d'origine et s'assure que l'image existe et n'est pas vide. Interprétez ensuite les écarts : un faible nombre de contrôles peut signaler une lacune ou, au contraire, des contrôles plus larges - il faut examiner leur efficacité.
    """),
    code("""
    domains = list(risk_strategy)
    counts = [len(risk_strategy[d]) for d in domains]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(domains, counts, color=["#2563eb", "#0f766e", "#7c3aed", "#b45309"])
    ax.set_title("Couverture de la stratégie de risque par domaine")
    ax.set_ylabel("Nombre de contrôles")
    ax.set_ylim(0, max(counts) + 1)
    ax.grid(axis="y", linestyle="--", alpha=.4)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, count + .05, str(count), ha="center")
    fig.tight_layout()
    figure_path = ARTIFACTS / "lab3_couverture_controles.png"
    fig.savefig(figure_path, dpi=150)
    plt.show()
    """),
    code("""
    # Test - les comptes reflètent exactement la stratégie et l'image a été écrite.
    assert counts == [len(v) for v in risk_strategy.values()]
    assert figure_path.exists() and figure_path.stat().st_size > 1_000
    print("OK - visualisation créée et validée.")
    """),
    md("""
    ## Réflexion finale

Quelle preuve demanderiez-vous pour juger qu'un contrôle est efficace ? Pensez aux métriques, aux journaux, aux revues humaines et aux retours des candidats. Une stratégie devient opérationnelle seulement lorsque ses contrôles sont mesurés, suivis et améliorés dans le temps.
    """),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in [("01_lab_risques_systeme_ia.ipynb", lab1), ("02_lab_audit_conformite_ia.ipynb", lab2), ("03_lab_strategie_risque_ia.ipynb", lab3)]:
        (OUT / name).write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
