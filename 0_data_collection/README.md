# Data Collection

This directory contains the social media data used for analyzing "local person" identity definition standards across Chinese cities.

## Dataset Overview

The `dataset.csv` file contains social media comments and replies collected from Chinese social media platforms, primarily focusing on discussions about local identity in various cities.

**Note**: This is the initial release of the dataset. Additional data is currently being organized and will be released in future updates.

## Data Sources

### Platform Information
- **Primary Source**: Red Note (小红书) - Chinese social media platform
- **Data Source**: 2025 Red Note WILL Business Conference (2025小红书WILL商业大会)
- **Reference**: [Red Note User Statistics Report](https://doc.weixin.qq.com/doc/w3_AWUA1QYUANgCGABwcG1QgSd7BhOsZ?dver=)
- **User Demographics**: 
  - 50% of users are from the 1995s generation
  - 35% of users are from the 2000s generation  
  - 50% of users live in tier-1 and tier-2 cities

### User Statistics
Detailed user statistics are available in the `red_note_user_statistics/` directory:
- `red_note_user_statistics_2024.pdf` (In page 10)
- `red_note_user_statistics_2025.pdf` (In page 12)

## Geographic Coverage



**Future Updates**: Additional cities and expanded datasets will be included in subsequent releases.

## Dataset Structure

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Unique identifier for each text | 1, 2, 3... |
| `text_id` | Original text identifier with city prefix | `comment_chengdu_1_1`, `reply_harbin1_1_4_7` |
| `type` | Content type: `comment` or `reply` | comment, reply |
| `text` | The actual social media text content | Full text of the comment/reply |

## Data Characteristics

- **Content Types**: Comments and replies from social media discussions
- **Language**: Primarily Chinese (Simplified)
- **Major Themes**: Local identity, urban development, cultural differences, regional stereotypes
- **Discussion Topics**: 
  - City pride and local identity
  - Urban vs. suburban distinctions
  - Cultural and dialect differences
  - Economic development and social status
  - Historical heritage and modernization
  - Inter-city comparisons and stereotypes

## Data Quality

- **Labeled Data**: Contains both manually annotated and unlabeled texts
- **Geographic Diversity**: Covers cities from different regions and development levels
- **Content Diversity**: Includes both positive and negative discussions about local identity

## Usage Notes

### Research Applications
- Analysis of local identity construction in Chinese cities
- Study of Recognition Logic framework implementation
- Cross-city comparison of identity markers
- Linguistic and cultural identity research

### Ethical Considerations
- All data is publicly available social media content
- User identities have been anonymized
- Content focuses on identity discussions rather than personal information

## File Organization

```
0_data_collection/
├── README.md                           # This file
├── dataset.csv                         # Main dataset
└── red_note_user_statistics/           # Platform statistics
    ├── red_note_user_statistics_2024.pdf
    └── red_note_user_statistics_2025.pdf
```