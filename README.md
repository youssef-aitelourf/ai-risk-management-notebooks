# Notebooks - gestion des risques IA

Trois notebooks Jupyter, en français, adaptés des trois labs programmables du guide fourni :

- `notebooks/01_lab_risques_systeme_ia.ipynb` : données Adult, modèle de référence, évaluation et registre des risques ;
- `notebooks/02_lab_audit_conformite_ia.ipynb` : détection de lacunes de conformité et génération d'un plan Markdown ;
- `notebooks/03_lab_strategie_risque_ia.ipynb` : stratégie SmartHire, export YAML et graphique de couverture.

Les mêmes notebooks sont également disponibles en anglais dans `notebooks_en/`.

## Installation et exécution

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build_notebooks.py
.venv/bin/python build_notebooks_en.py
.venv/bin/jupyter lab
```

Chaque bloc fonctionnel contient une cellule `Test` avec des assertions. Pour une vérification automatisée intégrale, exécutez :

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/*.ipynb
```

Les fichiers produits par les notebooks sont créés dans `notebooks/artifacts/`.

## Périmètre

Ces artefacts sont pédagogiques. Ils illustrent des contrôles et ne remplacent ni analyse juridique, ni étude d'impact, ni audit de production.
