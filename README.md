# Conformal Prediction and Pl@ntNet-CrowdSWE-v2 database

Welcome to our project on conformal prediction and the Pl@ntNet-CrowdSWE-v2 database for the academic year 2025-2026.

Name of supervisors:

- Joseph Salmon (joseph.salmon@inria.fr)
- Christophe Botella (christophe.botella@inria.fr)
- Jean-Baptiste Fermanian (jean-baptiste.fermanian@inria.fr)

The team members are:

- AGOSSOU Dossou (dossou.agossou@etu.umontpellier.fr)
- DIAGNE Moussa (moussa.diagne@etu.umontpellier.fr)
- KARIMOU Firdaousse (firdaousse.karimou@etu.umontpellier.fr)

---

Pl@ntNet is a citizen science mobile application for plant identification, used by over 20 million users worldwide. Its deep learning algorithm can identify more than 75,000 species, but most of them are rare and lack sufficient training images, leading to frequent prediction errors. This project is based on the [Pl@ntNet-CrowdSWE-v2](https://zenodo.org/records/17913995) database, containing approximately 5.5 million observations of plant species in South-Western Europe, including 21,624 expert-validated observations covering 3,082 species.

The core challenge is the **long-tail distribution**: 80% of species have fewer than 10 observations, making standard conformal prediction methods unreliable for rare species. To address this, the project is divided into several parts:

- Exploration and preprocessing of the Pl@ntNet-CrowdSWE-v2 dataset (JSON parsing, expert/non-expert splitting, stratified calibration/test partitioning)
- Validation of the conformal prediction pipeline on synthetic data (balanced and imbalanced Gaussian mixtures)
- Implementation and comparison of three conformal methods (Standard CP, Classwise CP, PAS CP) across three coverage notions (marginal, conditional, macro)
- Diagnosis and correction of the truncation bias caused by the 0.001 score threshold
- Temperature scaling optimization to reduce prediction set sizes while preserving coverage guarantees

The main finding is that **PAS CP** (Prevalence-Adjusted Softmax), combined with a temperature parameter T = 0.5, achieves the best trade-off: approximately 95% macro-coverage with an average prediction set size of about 4 species.

Here is a diagram of the architecture of our project, detailing the location of each folder and file:

```
├── CP_for_PlantNet/
│   ├── src/
│   │   ├── 01a_conversion_json_csv.py
│   │   ├── 01b_inspect.py
│   │   ├── 02_splitting_expert_nonexpert.py
│   │   ├── 03_shuffle_split_50_50.py
│   │   ├── 04_statistique_descriptive.py
│   │   ├── 05a_sanity_check_equilibre.py
│   │   ├── 05b_sanity_check_desequilibre.py
│   │   ├── 06a_marginale_coverage.py
│   │   ├── 06b_conditionnelle_coverage.py
│   │   ├── 06c_macro_coverage.py
│   │   ├── 07_biais_correction.py
│   │   └── 08_temperature_scaling.py
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── figures/
│   │   ├── fig_conditional_naive_alpha005.png
│   │   ├── fig_histogramme_prevalence.png
│   │   ├── fig_longue_traine.png
│   │   ├── fig_macro_naive.png
│   │   ├── fig_marginale_naive.png
│   │   ├── fig_sanity_check_desequilibre.png
│   │   ├── fig_sanity_check_equilibre.png
│   │   └── fig_temperature_scaling.png
│   ├── rapport/
│   │   ├── Images/
│   │   ├── Rapport.tex
│   │   └── AGOSSOU-DIAGNE-KARIMOU_Rapport.pdf
│   ├── presentation/
│   │   └── soutenance.pptx
│   ├── .gitignore
│   ├── requirements.txt
│   └── README.md
```

## References

- Angelopoulos, A. N., & Bates, S. (2021). *A gentle introduction to conformal prediction and distribution-free uncertainty quantification*. arXiv:2107.07511.
- Ding, T., Fermanian, J.-B., & Salmon, J. (2025). *Conformal Prediction for Long-Tailed Classification*. ICLR 2025. Blog: https://josephsalmon.eu/blog/long-tail/
- Lefort, T., et al. (2024). *Pl@ntNet collaborative learning: South-Western-Europe dataset*. arXiv:2406.03356.
- Dabah, L., & Tirer, T. (2024). *On Temperature Scaling and Conformal Prediction of Deep Classifiers*.
- Sadinle, M., Lei, J., & Wasserman, L. (2019). *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels*. JASA.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.