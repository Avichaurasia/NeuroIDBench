# DEEB

<div align="center">
<img src="logo.png" alt="My Logo" width="400" height="380">
</div>


[![Pandas Latest Release](https://img.shields.io/pypi/v/pandas.svg)](https://pypi.org/project/pandas/) [![MNE Latest Release - MNE](https://img.shields.io/pypi/v/mne.svg)](https://pypi.org/project/mne/) [![PyPI - Keras](https://img.shields.io/badge/Keras-latest-red.svg)](https://pypi.org/project/keras/)

<details>
<summary><strong>📘 Short Summary of Thesis</strong></summary>

Brainwaves present a compelling avenue for secure person authentication because they are inherently unobservable externally and capable of facilitating liveness detection. Harnessing brainwave’s unique and individualistic attributes, they have found extensive utility in various authentication applications. Nonetheless, the domain of brainwave authentication research has witnessed an upsurge in diverse experimental setups and the meticulous fine-tuning of parameters to optimize authentication methodologies. The substantial diversity in their methods poses a significant obstacle in assessing and measuring authentic research advancements. 

To address this multifaceted issue, this thesis introduces a versatile and robust benchmarking framework tailored explicitly for brainwave authentication systems. This framework draws upon the resources of four publicly accessible medical-grade brainwave datasets. It is worth mentioning that our study encompasses a substantial sample size consisting of 195 participants. The number of participants in our study is noteworthy, particularly when compared to the customary approach in brainwave authentication research, which typically involves a participant pool about one-fifth the size of our study.

Our extensive assessment encompassed a variety of state-of-the-art authentication algorithms, including Logistic Regression, Linear Discriminant Analysis, Support Vector Machine, Naive Bayes, K-Nearest Neighbours, Random Forest, and advanced deep learning methods like Siamese Neural Networks. Our evaluation approach incorporated both within-session (single-session) and cross-session (multi-session) analysis, covering threat cases like close-set (seen attacker) and open-set (unseen attacker) scenarios to ensure the tool’s versatility in different contexts. 

In within-session evaluation, our framework showcased outstanding performance for several classifiers, particularly Siamese Networks, which achieved an Equal Error Rate of 1.60% in the unseen attacker scenario. Additionally, our benchmarking framework’s adaptability is a notable asset, allowing researchers to tailor pre-processing, feature extraction, and authentication parameters to suit their specific requirements.
</details>

This repository serves as a comprehensive resource for brainwave-based authentication methods. It encompasses the entire implementation codebase along with a collection of illustrative examples for conducting benchmarking experiments using this powerful tool. Please note that while this repository is a valuable resource for code and methodologies, it does not include the proprietary or sensitive data utilized in our thesis. 

The thesis was written at the IT Security group at Paderborn University. It was supervised by Patricia Arias Cabarcos, who also leads the group. Further, the implementation aspects of this benchmarking tool was supervised by Matin Fallahi, a reserach associate at Kalrsruhe Insistute of Technology, Germany.   


