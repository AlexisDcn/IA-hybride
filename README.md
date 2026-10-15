# IA hybride – Mini-projet, sujet 4 : lésions de la peau sous contraintes logiques

**Groupe 4** : Lina Khadly, Alexis Dechenais · ISIS Castres, 5e année DSIA

Classer des images de lésions cutanées (DermaMNIST, 7 types) en imposant la cohérence avec la hiérarchie médicale
bénin / malin, mesurer l'apport de cette contrainte quand seulement 10 % des étiquettes fines sont disponibles, et
exploiter l'incertitude pour décider quand transmettre à un dermatologue.

## Contenu

| Fichier | Rôle |
|---|---|
| `S4_lesions_peau_hybride.ipynb` | Le notebook complet, **déjà exécuté**, avec les explications pas à pas et les commentaires des résultats |
| `requirements.txt` | Dépendances Python |
| `Rapport_S4_groupe4.pdf` | Le rapport (4 pages) |
| `S4_soutenance.pptx` | La présentation de soutenance, avec les notes du présentateur |
| `data/dermamnist.npz` | Les données (à télécharger, voir ci-dessous ; non versionnées) |

Plan du notebook : données et hiérarchie (sourcée) → métriques → modèle et pertes logiques → expérience principale
(100 % et 10 % des étiquettes, 3 graines) → ablations → courbe d'apprentissage (2 % à 100 %) → calibration →
cas difficiles (classes rares, conflits entre têtes, perturbations, hors distribution) → analyse d'erreurs par brique →
prédiction conforme et règle d'escalade → coût et synthèse.

## Réexécuter

1. Installer les dépendances (Python 3.10 ou plus) :
   ```bash
   pip install -r requirements.txt
   ```
2. Télécharger les données depuis Zenodo (MedMNIST v2, environ 20 Mo) :
   https://zenodo.org/records/10519652/files/dermamnist.npz?download=1
   et placer le fichier dans `data/dermamnist.npz` (MD5 : `0744692d530f8e62ec473284d019b0c7`).
   Si le fichier manque, le notebook essaie de le télécharger automatiquement avec le paquet `medmnist`.
3. Lancer le notebook depuis la racine du dépôt :
   ```bash
   jupyter nbconvert --to notebook --execute S4_lesions_peau_hybride.ipynb --output S4_lesions_peau_hybride.ipynb
   ```
   ou l'ouvrir dans Jupyter / VS Code et tout exécuter.

**Durée** : environ 35 minutes sur un CPU à 2 cœurs (33 entraînements d'environ 40 s), beaucoup moins avec un GPU.
Pour vérifier rapidement que tout s'exécute (150 pas au lieu de 1200, une seule graine, résultats non significatifs) :
```bash
RAPIDE=1 jupyter nbconvert --to notebook --execute S4_lesions_peau_hybride.ipynb --output test_rapide.ipynb
```
(sous Windows PowerShell : `$env:RAPIDE="1"` avant la commande).

La partie 8.4 (hors distribution) télécharge MNIST avec `torchvision` ; sans réseau, elle se limite au bruit uniforme.

Les graines sont fixées ; une réexécution donne des valeurs proches mais pas identiques à celles commentées
(non-déterminisme des calculs sur CPU / GPU et versions de bibliothèques).

## Approche

* **Référence non hybride (R)** : CNN à 7 classes (≈ 34 000 paramètres), entraîné sur les images dont le type est connu.
* **Multi-tâche sans règle (M)** : même CNN avec une tête « type » (7) et une tête « groupe » (2).
* **Hybride (H)** : M + deux pertes logiques issues de la hiérarchie type ⇒ groupe :
  * règle « groupe observé ⇒ le type est l'un de ses enfants » : $-\log \sum_{k \in G(g)} p_k$ ;
  * cohérence entre les têtes par *semantic loss* : $-\log \sum_k p_k \, q_{g(k)}$ (variante : t-norme de Łukasiewicz) ;
  * décodage cohérent en sortie : $\arg\max_k p_k \, q_{g(k)}$.
* **Régime peu supervisé** : 10 % des étiquettes fines (tirage stratifié), le groupe restant connu pour toutes les images.
* **Incertitude** : mise à l'échelle par température, prédiction conforme (couverture 90 %) et règle d'escalade.

Hiérarchie utilisée : akiec, bcc, mel → malin / pré-malin ; bkl, df, nv, vasc → bénin (justifiée et sourcée dans le
notebook, partie 1.3).

## Principaux résultats (régime 10 % des étiquettes fines, moyenne sur 3 graines)

| | R : référence | M : multi-tâche | H : hybride |
|---|---|---|---|
| Précision équilibrée (7 types) | **0,389** | 0,384 | 0,369 |
| Rappel du mélanome | 0,344 | **0,354** | 0,324 |
| AUC bénin / malin | 0,817 | 0,858 | **0,862** |
| Incohérence entre têtes avant décodage | (une seule tête) | 5,7 % | 3,8 % |
| ECE (décodage brut) | 0,143 | 0,087 | **0,075** |

* La hiérarchie aide nettement la **décision bénin / malin** quand les étiquettes fines sont rares (AUC ≈ 0,86 même avec
  2 % des étiquettes, contre ≈ 0,78 pour la référence) et rend les sorties **cohérentes**.
* Elle n'aide pas à reconnaître le **type exact** : la règle « type ⇒ groupe » ne dit rien à l'intérieur d'un groupe et
  pousse vers la classe dominante (nævus).
* Les limites principales sont la **perception** (mélanome confondu avec nævus à 28×28) et l'absence de **détection hors
  distribution** (les modèles sont très confiants sur des chiffres manuscrits).

Le détail, les ablations et l'analyse d'erreurs sont dans le notebook.

## Sources

Données : Tschandl et al., *Scientific Data* 2018 (HAM10000) ; Yang et al., *Scientific Data* 2023 (MedMNIST v2).
Connaissance médicale : Tschandl et al. 2018 ; de Berker et al., *British Journal of Dermatology* 2017 (kératose actinique).
Méthodes : Xu et al., ICML 2018 (semantic loss) ; Badreddine et al., *Artificial Intelligence* 2022 (Logic Tensor Networks) ;
Guo et al., ICML 2017 (calibration) ; Hendrycks et Gimpel, ICLR 2017 (hors distribution) ; Angelopoulos et Bates, 2021
(prédiction conforme). Références complètes à la fin du notebook.

## Utilisation d'un assistant de code

Ce projet a été réalisé avec l'aide de **Claude Opus 5.5** (Anthropic), utilisé pour écrire le code, exécuter les
expériences et rédiger les explications et les commentaires des résultats.
