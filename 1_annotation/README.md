# Annotation

This directory contains annotation resources and agreement analysis for the "local person" identity classification project.

## Contents

- **Annotation Manual**: Detailed guidelines for annotators using the Recognition Logic (RL) framework
- **Agreement Analysis**: Inter-rater reliability metrics and agreement statistics
- **Dataset Richness**: Analysis of dialectal diversity and regional linguistic features

## Files

- `Identity Annotation Manual.pdf`: Complete annotation guidelines with RL framework definitions
- `agreements.py`: Script for calculating inter-rater agreement
- `agreement_results.md`: Markdown containing agreement statistics and reports
- `dialect_analysis.csv`: Detailed sample-by-sample analysis of regional linguistic features and dialectal expressions

## Dataset Richness and Dialectal Analysis

Despite its size, our dataset is exceptionally rich, featuring a significant number of samples from diverse dialect groups. Our analysis identified 32 samples of Sichuanese (from Chengdu and Chongqing), 24 of Cantonese (Guangzhou), 6 of Xiang (Changsha), 5 of Beijing Mandarin, 4 of Wuhan dialect, 3 of Jiao-Liao Mandarin (Qingdao), and 2 of Northeastern Mandarin (Harbin). While the dataset also includes comments from other locations (e.g., Hefei, Jinan), they did not contain distinct, analyzable dialectal features.

Our analysis identified over 70 unique regional linguistic features, which can be broken down to showcase the dataset's richness:

- **30 unique features are from Cantonese**, which involves distinct orthography and grammar (e.g., the aspect particle 咗, the verb 搵食). Our native Cantonese-speaking annotator accurately annotated these.

- **45 unique features are from other Mandarin dialects**, which differ from Standard Mandarin primarily at the lexical level. This category includes:
  - **25 Traditional Dialect Vocabularies**: Core dialectal words like 弯脚杆 (Sichuanese), 乡里别 (Xiang), and 小嫚 (Qingdao).
  - **10 Local/Cultural Keywords**: Terms deeply embedded in local culture and identity, including 三镇 (Wuhan) and 京爷 (Beijing).
  - **5 Grammatical Features**: Unique markers like the Sichuanese passive marker 遭 and the Changsha continuous aspect marker 起.
  - **5 Modern & Online Phenomena**: Contemporary innovations like the neologism 青普 (Wuhan) and online slang 被烧 (Chengdu).

The manageable number of unique features (70+ total) enabled comprehensive annotation coverage. The adequacy of our three-annotator team (North, Central, South) was ensured by a collaborative consensus model. Crucially, most identified dialects (excluding Cantonese) differ from Standard Mandarin primarily at the lexical level rather than in overall orthography, making them largely comprehensible with context [1]. Meanwhile, Cantonese expressions, which exhibit greater divergence, were accurately annotated by our native-speaking annotator. This triangulation approach, combined with an online search for contemporary slang, guaranteed that every regional expression was successfully resolved.

[1] Snow, D. (2004). Cantonese as Written Language: The Growth of a Written Chinese Vernacular. Hong Kong: Hong Kong University Press. https://doi.org/10.1515/9789882200531