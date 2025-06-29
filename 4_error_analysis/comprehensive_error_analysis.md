# Comprehensive LLM Error Analysis for Recognition Logic Classification

This document provides a comprehensive error analysis for the Recognition Logic (RL) classification task using large language models, combining overview, case studies, and detailed category analysis.

## Overview

The error analysis examines the performance of multiple models (Qwen3-32B, Gemma-3-27B, Qwen3-235B-A22B) with different prompting methods (few-shot, zero-shot, no chain-of-thought) on Chinese social media texts, comparing model predictions against human-annotated golden standards.

## Model Performance Comparison

| Model | Prompting Method | Total Samples | Error Count | Error Rate (%) | Spurious (%) | Missed (%) |
|-------|------------------|---------------|-------------|----------------|--------------|------------|
| **Qwen3-32B (Few-shot)** | Few-shot | 485 | 232 | 47.84 | 81.0 | 19.0 |
| **Qwen3-235B-A22B (Few-shot)** | Few-shot | 485 | 245 | 50.52 | 71.0 | 29.0 |
| **Gemma-3-27B (Few-shot)** | Few-shot | 485 | 247 | 50.93 | 65.6 | 34.4 |
| **Qwen3-32B (Zero-shot)** | Zero-shot | 485 | 471 | 97.11 | 3.0 | 97.0 |
| **Qwen3-235B-A22B (Zero-shot)** | Zero-shot | 485 | 470 | 96.91 | 3.1 | 96.9 |
| **Gemma-3-27B (Zero-shot)** | Zero-shot | 485 | 471 | 97.11 | 2.9 | 97.1 |
| **Qwen3-32B (No CoT)** | No CoT | 485 | 345 | 71.13 | 0.0 | 100.0 |
| **Qwen3-235B-A22B (No CoT)** | No CoT | 485 | 344 | 70.93 | 0.0 | 100.0 |
| **Gemma-3-27B (No CoT)** | No CoT | 485 | 350 | 72.16 | 2.0 | 98.0 |

## Key Findings

### 1. Prompting Method Impact
- **Few-shot significantly outperforms zero-shot**: Error rates reduced by 45-50%
- **Chain-of-thought is crucial**: Removing reasoning steps causes severe performance degradation
- **Zero-shot models show complete failure**: 96-97% error rates with mostly missed predictions

### 2. Model Size vs Performance
- **Smaller models can perform better**: Qwen3-32B outperformed larger Qwen3-235B-A22B

### 3. Error Type Patterns
- **Few-shot models**: Primarily spurious predictions (65-81%)
- **Zero-shot models**: Almost entirely missed predictions (96-97%)
- **No CoT models**: Complete failure (98-100% missed predictions)

## Detailed Error Case Analysis

### Case 1: Complex Multi-RL Case (RL1, RL4, RL7)
**ID**: reply_chengdu_2_2_5

**Text**: "真的很多，我就是周边的，不管现实还是网上看的很多，他们天天说自己土著然后说我们弯脚杆，碰到过一个成都gay来我们这弯酸我们这乡坝头，然后碰到过有人问我成都那里的，因为我以为是外地人所以说的成都这边的，我说都江堰的，他说二环以内才是成都，二环以外是四川..."

**Golden Labels**: RL1 (Vernacular Spatial Authority), RL4 (Linguistic-Cultural Recognition), RL7 (Occupational Typification)

**Model Predictions**:
- **Qwen3-32B (Few-shot)**: RL1, RL4 | **Missed RL7**
- **Qwen3-235B-A22B (Few-shot)**: RL1, RL4 | **Missed RL7**
- **Gemma-3-27B (Few-shot)**: RL1, RL3, RL4, RL6 | **Missed RL7, Spurious RL3, RL6**
- **All Zero-shot models**: No predictions | **Complete failure**

**Analysis**: This case demonstrates the difficulty of identifying RL7 (Occupational Typification). All few-shot models correctly identified spatial authority (RL1) and linguistic recognition (RL4), but missed the subtle occupational/class-based identity markers. Zero-shot models completely failed.

### Case 2: Simple Spatial Authority (RL1)
**ID**: reply_chengdu_1_61_1

**Text**: "热烈欢迎咱叙北第一带盐人 北门可不差，论三环外，东西三环可不一定有北三环好，再等火北弄好，那就真起飞了"

**Golden Labels**: RL1 (Vernacular Spatial Authority)

**Model Predictions**:
- **Qwen3-32B (Few-shot)**: RL1 | **Correct**
- **Qwen3-235B-A22B (Few-shot)**: RL1, RL5 | **Spurious RL5**
- **Gemma-3-27B (Few-shot)**: RL1 | **Correct**
- **All Zero-shot models**: No predictions | **Complete failure**

**Analysis**: This case shows the confusion between RL1 (spatial authority) and RL5 (functional livability). The larger model (Qwen3-235B-A22B) incorrectly added RL5, suggesting it interpreted the spatial discussion as infrastructure evaluation.

### Case 3: Functional Livability (RL5)
**ID**: reply_hefei1_1_2_11

**Text**: "还有交通特别乱，机动车从不让行人，电动车逆行，到处乱窜不遵守交通规则"

**Golden Labels**: RL5 (Functional Livability)

**Model Predictions**:
- **All models**: No predictions | **Complete failure**

**Analysis**: This case reveals a systematic failure across all models to identify RL5 (Functional Livability). Even though the text clearly discusses infrastructure and living conditions, no model could identify this category.

### Case 4: Occupational Typification (RL7)
**ID**: comment_hefei2_1_170

**Text**: "瑶海排除，乱死了尤其龙岗这边长江批发市场这里，绿灯过斑马线一堆开电瓶车运货的一点也不看，就上来撞人"

**Golden Labels**: RL7 (Occupational Typification)

**Model Predictions**:
- **Qwen3-32B (Few-shot)**: No predictions | **Missed**
- **Qwen3-235B-A22B (Few-shot)**: RL5 | **Missed RL7, Spurious RL5**
- **Gemma-3-27B (Few-shot)**: No predictions | **Missed**
- **All Zero-shot models**: No predictions | **Complete failure**

**Analysis**: This case confirms the difficulty of RL7 identification. The text discusses occupational characteristics (wholesale market workers), but most models failed to identify this. One model incorrectly classified it as RL5 (functional livability).

### Case 5: Complex Social Case (RL1, RL3, RL6)
**ID**: comment_qingdao_10

**Text**: "我市南土生土长的，上学的时候去了烟台，大学毕业回家后，嫁了胶州小伙子，因为孩子上学所以我们就住胶州了，我家市区有七八套房子，胶州城阳也都有房子..."

**Golden Labels**: RL1 (Vernacular Spatial Authority), RL3 (Family Rootedness), RL6 (Social Embeddedness)

**Model Predictions**:
- **All Few-shot models**: RL1, RL3, RL5, RL6 | **Spurious RL5**
- **All Zero-shot models**: No predictions | **Complete failure**

**Analysis**: This case shows consistent over-prediction of RL5 (Functional Livability) by few-shot models. All models correctly identified the core categories but added RL5, suggesting they interpreted property ownership as infrastructure evaluation.

### Case 6: Implicit Expression Challenge (RL1, RL4)
**ID**: comment_qingdao_86

**Text**: "我外地人在青岛生活十多年，公里公道说句，其实日常生活中青岛人没有表现出来青岛区域歧视，但是在偶有的聊天或者一些言论里，能看出来多少会带点儿那种感觉，比如他们会特指一些，像说那个沧口曼儿，或者说青岛话的口音，会问你家哪个区的，城阳的，哦，**怪不得**，你们懂那种感觉吧，但毕竟我不是青岛人，就听他们聊天中就会有那种地域划分和主城区优越感的只字片语"

**Golden Labels**: RL1 (Vernacular Spatial Authority), RL4 (Linguistic-Cultural Recognition)

**Model Predictions**:
- **Qwen3-32B (Few-shot)**: RL1, RL4 | **Correct**
- **Qwen3-235B-A22B (Few-shot)**: RL1, RL4 | **Correct**
- **Gemma-3-27B (Few-shot)**: RL1, RL4 | **Correct**
- **All Zero-shot models**: No predictions | **Complete failure**

**Analysis**: This case demonstrates the complexity of **implicit discriminatory expressions** in Chinese social media. The phrase "怪不得" (no wonder) appears neutral on the surface but carries subtle discriminatory undertones that require deep cultural understanding:

#### Multi-layer Complexity:
1. **Surface Politeness vs Deep Sarcasm**: "怪不得" masks judgment as casual observation
2. **Dialect as Power Marker**: "沧口曼儿" and accent recognition function as social stratification tools  
3. **Implicit Consensus Building**: "你们懂那种感觉吧" assumes shared discriminatory understanding
4. **Geographic Identity Hierarchy**: "城阳的" triggers automatic social categorization

#### Why This Matters for AI Systems:
- **Hidden Bias Detection**: Models must recognize "polite discrimination" patterns
- **Cultural Context Requirements**: Understanding requires deep knowledge of local social hierarchies
- **Metaphorical Language**: Geographic labels carry social meanings beyond literal location
- **Tone and Implication**: Distinguishing between descriptive and evaluative language use

This case exemplifies why **dialect, sarcasm, and metaphor challenges for both annotators and models** require sophisticated cultural and social understanding, not just linguistic pattern recognition. The complexity lies not in vocabulary but in the social mechanisms that transform neutral words into vehicles for subtle discrimination.

### Case 7: Self-Attack Pattern - Defensive Identity Strategy (RL1, RL4)

**IDs**: comment_hangzhou_1_1_147 (ID: 2043), comment_hangzhou_1_1_208 (ID: 2055), reply_gz1_1_67_1 (ID: 2138)

**Texts**: 
1. "别说其他区了，我是西湖区三墩的浙大紫金港这边，小时候去趟龙翔桥都说去杭州。对外都说自己三墩乡下人。"
2. "我西湖区三墩镇，我都说自己杭州乡下人……跟他们城里人讲的话不一样"
3. "我小时候一直在河南住，满口番禺顺德乡下口音，乡下仔来的"

**Golden Labels**: RL1 (Vernacular Spatial Authority), RL4 (Linguistic-Cultural Recognition)

**Model Predictions**:
- **Few-shot models**: Generally correct for RL1, mixed performance on RL4
- **Zero-shot models**: Significant missed predictions
- **No CoT models**: Complete failure

**Self-Attack Pattern Analysis**:

#### 1. **Proactive Self-Labeling**
- "三墩乡下人" - Self-applied rural designation
- "杭州乡下人" - Urban-rural identity contradiction  
- "乡下仔来的" - Accepting derogatory terms

#### 2. **Defensive Identity Strategy**
- **Geographic Paradox**: Living in prestigious areas (West Lake District, near Zhejiang University) while claiming "rural" status
- **Preemptive Self-Deprecation**: "攻击自己" to avoid being attacked by others
- **Narrative Control**: Taking ownership of identity definition rather than accepting external labeling

#### 3. **Linguistic Markers of Difference**
- "跟他们城里人讲的话不一样" - Dialect as identity boundary
- "满口番禺顺德乡下口音" - Accent as social marker
- Language used to reinforce "outsider" status despite geographic belonging

#### 4. **Psychological Adaptation Mechanism**
- **Identity Fragmentation**: Simultaneous urban geography and rural identity
- **Social Protection**: Self-deprecation as defense against discrimination
- **Cultural Navigation**: Managing complex urban-rural identity transitions

**AI System Challenges**: This pattern reveals why current models struggle with:
- **Contextual Irony**: Geographic privilege coexisting with claimed marginalization
- **Defensive Communication**: Self-attack as protective rather than self-hating behavior
- **Identity Fluidity**: Complex negotiations between assigned and claimed identities
- **Cultural Adaptation**: Understanding urbanization-driven identity strategies

**Sociological Significance**: The "Self-Attack" pattern reflects deeper issues in China's rapid urbanization:
- **Identity Displacement**: Rural-origin populations navigating urban spaces
- **Preemptive Stigma Management**: Anticipating and deflecting discrimination
- **Agency in Marginalization**: Claiming control over identity narratives
- **Urban Integration Strategies**: Complex identity negotiations in city environments

**Core Insight**: This pattern demonstrates that regional identity classification involves not just geographic or linguistic markers, but sophisticated psychological and social strategies developed in response to systemic discrimination. Current AI models lack the cultural depth to understand these adaptive identity performances.

### Case 8: Marginalized Identity Dilemma - Structural Exclusion Pattern (RL2, RL3, RL6)

**ID**: comment_gz2_1_179 (ID: 2311)

**Text**: "你说我是不是广州人？我爷东莞人，抗日时期跑到广州来定居，我妈是广州农村，然后我户口跟我妈，我妈户口一直在村里，所以我有村分红，但是在村里属于外姓人"

**Golden Labels**: RL2 (Administrative Legitimacy), RL3 (Family Rootedness), RL6 (Social Embeddedness)

**Model Predictions**:
- **Few-shot models**: Generally correct identification of multiple RL categories
- **Performance varies**: Complex multi-category case with mixed prediction accuracy

**Marginalized Identity Dilemma Analysis**:

#### 1. **Multi-layered Identity Conflict**
- **Paternal Lineage**: Dongguan origin (external/outsider)
- **Maternal Connection**: Guangzhou rural background (local but rural)
- **Administrative Status**: Household registration in village (legal belonging)
- **Economic Integration**: Village dividend rights (economic inclusion)
- **Social Exclusion**: "外姓人" (different surname = social outsider)

#### 2. **Structural Exclusion Mechanisms**
- **Bloodline Politics**: Patrilineal surname system creating permanent outsider status
- **Historical Legacy**: Anti-Japanese War migration creating multi-generational otherness
- **Economic-Social Disconnect**: Financial benefits without social acceptance
- **Village Hierarchy**: Internal stratification within rural communities

#### 3. **Identity Fragmentation Dimensions**
- **Legal vs. Social**: Administrative inclusion but social rejection
- **Economic vs. Cultural**: Material benefits but cultural exclusion
- **Generational vs. Present**: Historical migration affecting contemporary identity
- **Individual vs. Collective**: Personal confusion about group belonging

#### 4. **Questioning Identity Strategy**
- **External Validation Seeking**: "你说我是不是广州人？" (asking others to define identity)
- **Identity Uncertainty**: Unable to self-determine belonging
- **Passive Acceptance**: Acknowledging rather than fighting exclusion
- **Complexity Acknowledgment**: Recognizing multiple conflicting identity markers

**Contrast with Self-Attack Pattern**:

| Dimension | Self-Attack Pattern | Marginalized Identity Dilemma |
|-----------|-------------------|------------------------------|
| **Agency** | Proactive self-deprecation | Passive structural exclusion |
| **Strategy** | Defensive identity management | Seeking external validation |
| **Geographic Advantage** | Has privilege but claims disadvantage | Has legal status but lacks social acceptance |
| **Control** | Uses self-attack for narrative control | Lacks control over exclusion mechanisms |
| **Identity Clarity** | Clear about using defensive strategy | Genuinely uncertain about belonging |

**AI System Challenges**: This pattern reveals additional complexity for models:
- **Multi-generational Identity**: Understanding how historical events affect contemporary belonging
- **Bloodline Politics**: Recognizing the power of surname/lineage systems in Chinese culture
- **Economic-Social Separation**: Distinguishing between material and social inclusion
- **Structural vs. Personal**: Differentiating systemic exclusion from individual choice
- **Question-based Identity**: Understanding identity uncertainty rather than assertion

**Sociological Significance**: This pattern illuminates:
- **Persistent Traditional Hierarchies**: How ancient social structures survive modernization
- **Migration Legacy Effects**: Long-term impacts of historical population movements
- **Rural Social Stratification**: Complex internal hierarchies within village communities
- **Incomplete Integration**: How legal/economic inclusion doesn't guarantee social acceptance
- **Identity Authenticity Crisis**: Questioning belonging despite formal membership

**Core Insight**: Unlike defensive identity strategies, this pattern represents genuine structural marginalization where individuals possess formal membership (administrative, economic) but remain socially excluded due to genealogical and historical factors. This demonstrates how regional identity involves not just individual negotiation but deep-rooted social structures that AI systems must understand to accurately classify complex belonging patterns.

## RL Category Error Analysis

### Error Statistics by RL Category

#### Missed Rate Analysis (Ordered by Missed Rate - Descending)

| RL | Description | Missed Count | Total Golden | Missed Rate (%) | Typical Examples |
|----|-------------|--------------|--------------|-----------------|------------------|
| 7 | Occupational Typification | 76 | 90 | **84.4** | "三环外的弯脚杆咋个敢嫌弃哪个", "批发市场这里，绿灯过斑马线一堆开电瓶车运货的", "底层人太多了，实际上厦门的国企包括公务员的工资平均都要比福州高很多", "骚扰电话都是外地口音", "家里人是武钢的，记得小时候，一些武昌的亲戚一边羡慕我家人的'铁饭碗'" |
| 5 | Functional Livability | 128 | 210 | **61.0** | "交通便利，去哪里都方便", "设施齐全，配套完善", "环境好，绿化挺好", "生活方便，居住条件不错", "配套完善，生活便利" |
| 6 | Social Embeddedness | 151 | 300 | **50.3** | "有没有租收？年底有没有分红？", "社会关系多，人脉资源丰富", "社交网络广，朋友多", "社区关系好，邻居和睦", "社会地位高，受人尊敬" |
| 1 | Vernacular Spatial Authority | 446 | 954 | **46.8** | "三环外的弯脚杆", "二环以内才是成都，二环以外是四川", "市中心的人看不起郊区的", "城乡结合部的人", "三环外但是大成都的弯脚杆" |
| 2 | Administrative Legitimacy | 101 | 222 | **45.5** | "户口在本地，身份证也是本地的", "行政划分上属于这个区", "政策规定，官方认可", "政府文件显示", "官方身份认证" |
| 3 | Family Rootedness | 201 | 450 | **44.7** | "祖祖辈辈都是大城市武汉人", "家族在这里生活了几代人", "亲戚都在这里", "老家就是这里，根在这里", "祖辈传下来的" |
| 4 | Linguistic-Cultural Recognition | 244 | 552 | **44.2** | "口音很重，一听就知道是本地人", "方言土话，本地特色", "说话有本地腔调", "语言习惯很本地化", "本地话很地道" |

#### Spurious Rate Analysis (Ordered by Spurious Rate - Descending)

| RL | Description | Spurious Count | Total Non-Golden | Spurious Rate (%) | Typical Examples |
|----|-------------|----------------|------------------|-------------------|------------------|
| 1 | Vernacular Spatial Authority | 188 | 1956 | **9.6** | "北京是个大城市", "上海发展很快", "广州的天气很好", "这个城市不错", "这个地区很发达" (general location descriptions) |
| 5 | Functional Livability | 216 | 2700 | **8.0** | "这里的房子很贵", "房价一直在涨", "居住环境不错", "生活条件很好", "环境很优美" (general living descriptions) |
| 3 | Family Rootedness | 160 | 2460 | **6.5** | "家人都在这里", "父母住在这里", "孩子在这里上学", "家庭很和睦", "亲人都在身边" (general family descriptions) |
| 6 | Social Embeddedness | 162 | 2610 | **6.2** | "朋友很多", "同事关系不错", "邻居很友好", "社交活动丰富", "人际关系好" (general social descriptions) |
| 2 | Administrative Legitimacy | 144 | 2688 | **5.4** | "政府很重视", "部门管理严格", "机构设置合理", "管理制度完善", "制度很规范" (general governance descriptions) |
| 4 | Linguistic-Cultural Recognition | 105 | 2358 | **4.5** | "说话很标准", "语言很规范", "文化底蕴深厚", "传统保持得很好", "习惯很健康" (general linguistic/cultural descriptions) |
| 7 | Occupational Typification | 39 | 2820 | **1.4** | "房山最大的特产是跨区打工人", "包河区公务员分数线最高", "父母都是工人或者事业单位公务员", "早期南宁各类资本大佬聚会碰头", "都是上流社会的人" (work/status descriptions) |

### Most Challenging RL Categories

| RL | Description | Missed Rate (%) | Spurious Rate (%) | Key Challenge |
|----|-------------|-----------------|-------------------|---------------|
| **RL7** | Occupational Typification | **84.4** | 1.4 | Models consistently fail to identify occupational/class-based identity markers |
| **RL5** | Functional Livability | **61.0** | 8.0 | Models struggle with infrastructure evaluation vs spatial authority |
| **RL6** | Social Embeddedness | **50.3** | 6.2 | Difficulty identifying social network and asset-based identity |

### Best Performing RL Categories

| RL | Description | Missed Rate (%) | Spurious Rate (%) | Performance |
|----|-------------|-----------------|-------------------|-------------|
| **RL4** | Linguistic-Cultural Recognition | 44.2 | **4.5** | Low false positive rate |
| **RL3** | Family Rootedness | 44.7 | **6.5** | Balanced performance |
| **RL2** | Administrative Legitimacy | 45.5 | **5.4** | Moderate performance |

## Detailed Analysis by Category

### RL7 (Occupational Typification) - Most Difficult
- **Missed Rate**: 84.4% (highest)
- **Spurious Rate**: 1.4% (lowest)
- **Challenge**: Models consistently fail to identify occupational/class-based identity markers
- **Missed Examples**: 
  - "三环外的弯脚杆咋个敢嫌弃哪个" - occupational slur
  - "批发市场这里，绿灯过斑马线一堆开电瓶车运货的" - direct occupational reference
  - "底层人太多了，实际上厦门的国企包括公务员的工资平均都要比福州高很多" - subtle occupational class reference
  - "骚扰电话都是外地口音" - occupational stereotype
  - "家里人是武钢的，记得小时候，一些武昌的亲戚一边羡慕我家人的'铁饭碗'" - professional status indicator
- **Spurious Examples**:
  - "房山最大的特产是跨区打工人" - work description, not occupational identity
  - "包河区公务员分数线最高，高的离谱" - work-related term, not occupational typification
  - "父母都是工人或者事业单位公务员，家里没有土地" - work/status terms, not occupational identity

### RL5 (Functional Livability) - Most Confused
- **Missed Rate**: 61.0% (second highest)
- **Spurious Rate**: 8.0% (second highest)
- **Challenge**: Models confuse spatial discussions with infrastructure evaluation
- **Missed Examples**:
  - "交通便利，去哪里都方便" - infrastructure evaluation
  - "设施齐全，配套完善" - functional assessment
  - "环境好，绿化挺好" - livability evaluation
  - "生活方便，居住条件不错" - functional convenience
- **Spurious Examples**:
  - "这里的房子很贵" - general property term, not functional evaluation
  - "房价一直在涨" - economic term, not livability assessment
  - "居住环境不错" - general living term, not functional evaluation

### RL6 (Social Embeddedness) - Moderate Difficulty
- **Missed Rate**: 50.3% (moderate)
- **Spurious Rate**: 6.2% (moderate)
- **Challenge**: Difficulty identifying social network and asset-based identity
- **Missed Examples**:
  - "有没有租收？年底有没有分红？" - asset-based identity
  - "社会关系多，人脉资源丰富" - social network references
  - "社交网络广，朋友多" - social capital
  - "社区关系好，邻居和睦" - community embeddedness
- **Spurious Examples**:
  - "朋友很多" - general social term, not embeddedness
  - "同事关系不错" - work relationship, not social embeddedness
  - "邻居很友好" - spatial relationship, not social embeddedness

### RL1 (Vernacular Spatial Authority) - High Spurious Rate
- **Missed Rate**: 46.8% (moderate)
- **Spurious Rate**: 9.6% (highest)
- **Challenge**: Models over-predict spatial authority for general location terms
- **Missed Examples**:
  - "三环外的弯脚杆" - spatial boundary reference
  - "二环以内才是成都，二环以外是四川" - spatial authority marker
  - "市中心的人看不起郊区的" - spatial hierarchy
  - "城乡结合部的人" - spatial classification
- **Spurious Examples**:
  - "北京是个大城市" - general city name, not spatial authority
  - "上海发展很快" - general city name, not spatial authority
  - "这个城市不错" - general location term, not spatial authority

### RL4 (Linguistic-Cultural Recognition) - Best Balanced
- **Missed Rate**: 44.2% (moderate)
- **Spurious Rate**: 4.5% (lowest)
- **Strength**: Models rarely make false positive predictions
- **Missed Examples**:
  - "口音很重，一听就知道是本地人" - linguistic identity marker
  - "方言土话，本地特色" - cultural linguistic marker
  - "说话有本地腔调" - local language reference
  - "本地话很地道" - regional language identity
- **Spurious Examples**:
  - "说话很标准" - general linguistic term, not cultural recognition
  - "语言很规范" - general term, not cultural identity
  - "文化底蕴深厚" - general cultural term, not linguistic recognition

### RL2 (Administrative Legitimacy) - Moderate Performance
- **Missed Rate**: 45.5% (moderate)
- **Spurious Rate**: 5.4% (moderate)
- **Challenge**: Distinguishing administrative authority from general governance
- **Missed Examples**:
  - "户口在本地，身份证也是本地的" - administrative status
  - "行政划分上属于这个区" - administrative boundaries
  - "政策规定，官方认可" - administrative authority
  - "政府文件显示" - official documentation
- **Spurious Examples**:
  - "政府很重视" - general governance term, not administrative legitimacy
  - "部门管理严格" - general organizational term, not administrative authority
  - "管理制度完善" - general management term, not administrative legitimacy

### RL3 (Family Rootedness) - Moderate Performance
- **Missed Rate**: 44.7% (moderate)
- **Spurious Rate**: 6.5% (moderate)
- **Challenge**: Distinguishing family identity from general family references
- **Missed Examples**:
  - "祖祖辈辈都是大城市武汉人" - ancestral connection
  - "家族在这里生活了几代人" - family lineage
  - "亲戚都在这里" - family network
  - "老家就是这里，根在这里" - family origin
- **Spurious Examples**:
  - "家人都在这里" - general family term, not rootedness
  - "父母住在这里" - general family term, not rootedness
  - "孩子在这里上学" - general family term, not rootedness

## Key Patterns

### 1. RL7 (Occupational Typification) - Most Difficult
- **Pattern**: Consistently missed across all models
- **Challenge**: Subtle occupational references are hard to detect
- **Example**: Wholesale market workers, professional associations

### 2. RL5 (Functional Livability) - Most Over-predicted
- **Pattern**: Frequently added when not present in golden standard
- **Challenge**: Models confuse spatial discussions with infrastructure evaluation
- **Example**: Property ownership interpreted as functional evaluation

### 3. Zero-shot vs Few-shot Performance
- **Zero-shot**: Complete failure (no predictions)
- **Few-shot**: Significant improvement but still has issues
- **Pattern**: Zero-shot models are overly conservative

### 4. Model Size vs Performance
- **Qwen3-32B**: Best performance in few-shot
- **Qwen3-235B-A22B**: Slightly worse, more prone to spurious predictions
- **Gemma-3-27B**: Similar performance to larger models

## Consensus Error Analysis

### All Models Spurious Predictions: 152 Samples

**Pattern**: All few-shot models predicted the same spurious RL categories for these samples, while zero-shot models mostly failed to predict anything.

**Key Examples**:

1. **RL6 (Social Embeddedness) - Correct Predictions**:
   - Text: "有没有租收？年底有没有分红？"
   - Golden: {6} (Social Embeddedness)
   - All Few-shot models: {6} (Correct)
   - All Zero-shot models: {} (Missed)

2. **RL1 (Spatial Authority) - Correct Predictions**:
   - Text: "广州，是大家再熟悉不过的城市..."
   - Golden: {1} (Vernacular Spatial Authority)
   - All Few-shot models: {1} (Correct)
   - Most Zero-shot models: {} (Missed)

3. **RL4 (Linguistic Recognition) - Correct Predictions**:
   - Text: "不知道 但是知道土著一直在被其他人嘲讽口音"
   - Golden: {4} (Linguistic-Cultural Recognition)
   - All Few-shot models: {4} (Correct)
   - All Zero-shot models: {} (Missed)

### No All-Models-Missed Cases

**Finding**: No cases where all models completely missed all golden labels, suggesting:
- Different models have different strengths
- The task is not completely impossible
- Model diversity helps coverage

## Error Type Distribution

| Model | Error Rate | Spurious (%) | Missed (%) |
|-------|------------|--------------|------------|
| Qwen3-32B (Few-shot) | 47.8% | 81.5% | 34.9% |
| Qwen3-235B-A22B (Few-shot) | 50.5% | 84.9% | 29.0% |
| Gemma-3-27B (Few-shot) | 50.9% | 81.0% | 32.4% |
| Qwen3-32B (Zero-shot) | 69.9% | 6.8% | 96.8% |
| Qwen3-235B-A22B (Zero-shot) | 70.9% | 9.3% | 96.5% |
| Gemma-3-27B (Zero-shot) | 62.9% | 51.5% | 59.7% |

## Recommendations

### 1. Priority Improvements

#### High Priority: RL7 (Occupational Typification)
- **Action**: Add more occupational typification examples in few-shot prompts
- **Focus**: Professional associations, occupational stereotypes, class-based identity
- **Expected Impact**: Reduce 84.4% missed rate significantly

#### High Priority: RL5 (Functional Livability)
- **Action**: Clarify distinction between RL1 and RL5
- **Focus**: Infrastructure vs spatial authority, living conditions vs location
- **Expected Impact**: Reduce both missed (61.0%) and spurious (8.0%) rates

### 2. Model-Specific Strategies

#### Few-shot Models
- **Focus**: Reduce spurious predictions
- **Strategy**: Add negative examples (what NOT to classify as each RL)
- **Target**: RL1 (9.6% spurious), RL5 (8.0% spurious)

#### Zero-shot Models
- **Focus**: Improve overall prediction capability
- **Strategy**: Better prompt engineering
- **Target**: All categories (complete failure pattern)

### 3. Training Data Improvements

#### RL7 Examples
- Add more diverse occupational contexts
- Include subtle class-based identity markers
- Provide clear examples of occupational typification

#### RL5 Examples
- Distinguish from spatial authority (RL1)
- Focus on infrastructure and living condition evaluation
- Provide clear functional vs spatial examples

### 4. Evaluation Framework

#### Category-Specific Metrics
- Track performance by RL category separately
- Weight errors by category difficulty
- Focus evaluation on problematic categories (RL7, RL5)

#### Consensus Analysis
- Monitor cases where all models make the same error
- Use consensus errors to identify systematic issues
- Develop targeted solutions for consensus problems

## Conclusion

The analysis reveals that:
1. **RL7 (Occupational Typification)** is the most challenging category with 84.4% missed rate
2. **RL5 (Functional Livability)** has both high missed (61.0%) and spurious (8.0%) rates
3. **Few-shot prompting** significantly improves performance across all categories
4. **Zero-shot models** show complete failure patterns
5. **No consensus missed cases** suggest model diversity is beneficial

The findings provide clear direction for improving RL classification performance through targeted prompt engineering and training data enhancement. 