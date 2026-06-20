# Authorship Verification: Pseudo-Chrysostomos vs. Asterius

## Description
This project applies computational stylometry and machine learning to investigate the authorship of ancient Greek texts. Specifically, it aims to determine which texts traditionally grouped under the umbrella term "Pseudo-Chrysostomos" exhibit a stylistic fingerprint similar to the known works of the author Asterius. 

By extracting linguistic features and training a Siamese Neural Network, the project computes the stylistic distance between unidentified texts and verified samples, ranking the most likely candidates for Asterius' authorship.

## Methodology
The pipeline consists of two main phases:
1. **Feature Extraction:** Ancient Greek texts (XML/TXT) are parsed and divided into standardized samples (~1000 words). Using the specialized `grc_odycy_joint_trf` NLP model, the script extracts three types of features:
   - Most Frequent Words (MFW)
   - Part-of-Speech (POS) Trigrams (Syntax)
   - Word Affixes (Morphology)
2. **Authorship Verification:** The top features are normalized and fed into a Siamese Neural Network (via the `freestyl` library). The network is trained using a contrastive loss function to recognize the stylistic boundaries of different authors. Finally, it predicts the stylistic distance between the Pseudo-Chrysostomos texts and Asterius.

## Prerequisites & Installation
Ensure you have Python 3.8+ installed. The project relies on GPU acceleration (CUDA) if available.

### Required Libraries
Install the necessary Python packages:
```bash
pip install pandas spacy beautifulsoup4 lxml nltk freestyl