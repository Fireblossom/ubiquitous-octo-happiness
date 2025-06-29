# RL7 (Occupational Typification) Detailed Analysis

This document provides comprehensive analysis of RL7 (Occupational Typification) errors, including missed predictions, spurious predictions, and occupational stereotype analysis.

## Overview

- **Total RL7 golden samples**: 90
- **Total RL7 missed samples**: 76 (84.4% missed rate)
- **Total RL7 spurious samples**: 25 (1.4% spurious rate)
- **Most challenging RL category**: Highest missed rate among all categories

## Missed Analysis

### Missed Patterns

| Pattern | Count | Percentage |
|---------|-------|------------|
| All models missed RL7 | 8 | 53.3% |
| Some models missed RL7 | 7 | 46.7% |
| Few-shot models missed RL7 | 13 | 86.7% |
| Zero-shot models missed RL7 | 15 | 100% |

### Key Findings

1. **Zero-shot models completely fail**: All zero-shot models missed RL7 in all 15 cases
2. **Few-shot models struggle**: 13 out of 15 cases (86.7%) were missed by at least some few-shot models
3. **Complete failure cases**: 8 cases (53.3%) where all models missed RL7
4. **Partial success cases**: 7 cases (46.7%) where at least one model correctly identified RL7

### Sample Categories

#### 1. Wholesale Market (1 sample - 6.7%)

**Example**: "瑶海排除，乱死了尤其龙岗这边长江批发市场这里，绿灯过斑马线一堆开电瓶车运货的一点也不看，就上来撞人..."

**Characteristics**:
- Direct reference to wholesale market workers
- Occupational behavior description
- All models missed this obvious RL7 case

#### 2. Occupational Stereotype (3 samples - 20%)

**Examples**:
1. "厦门工资总体不高是因为底层人太多了，实际上厦门的国企包括公务员的工资平均都要比福州高很多"
2. "基本上现在接到那些骚扰电话都是外地口音哎，不是歧视外地人，只是觉得如果他们回老家，做点其他工作"
3. "丰台：有点偏，外来打工人租房子或者住员工宿舍的地方"

**Characteristics**:
- References to occupational hierarchies
- Class-based occupational distinctions
- Subtle occupational stereotypes

#### 3. Industry Related (1 sample - 6.7%)

**Example**: "青秀区我有两套了，这边这套是我们的单位集资房，我们有股份，可以得分红，而且是按照职称高低才能买房子的大小"

**Characteristics**:
- Direct reference to workplace hierarchy
- Occupational status indicators
- Professional ranking systems

#### 4. Occupational Slurs and Stereotypes (10 samples - 66.6%)

**Examples**:
1. "他们爱喊三环外的弯脚杆" (occupational slur)
2. "三环外的弯脚杆咋个敢嫌弃哪个" (occupational slur)
3. "家里人是武钢的，记得小时候，一些武昌的亲戚一边羡慕我家人的'铁饭碗'"
4. "成都土著的优越感是最强的...有资格被看不起的，首先你要是三环外但是大成都的弯脚杆和211"

**Characteristics**:
- Occupational slurs and stereotypes
- Professional identity markers
- Class-based occupational distinctions
- Subtle occupational references

### Detailed Case Analysis

#### Case 1: Complete Failure - All Models Missed

**Text**: "怎么说呢？成都人其实真的不排外，也不歧视，但是有一点就是他们爱喊三环外的弯脚杆，老一辈传下来的吧！"

**Golden Labels**: {1, 4, 7}
**All Models**: {1, 4} (missed RL7)

**Analysis**: The term "弯脚杆" is an occupational slur referring to rural/working-class people, clearly indicating RL7. All models missed this obvious occupational typification.

#### Case 2: Partial Success - Some Models Got It

**Text**: "家里人是武钢的，记得小时候，一些武昌的亲戚一边羡慕我家人的'铁饭碗'一边嘲笑我是乡里伢"

**Golden Labels**: {1, 7}
**Few-shot models**: {1, 7} (correct)
**Zero-shot models**: {} (missed all)

**Analysis**: "铁饭碗" (iron rice bowl) is a clear occupational status marker. Few-shot models correctly identified this, but zero-shot models completely failed.

#### Case 3: Subtle Occupational Reference

**Text**: "其实没必要比，厦门本地人只要在厦门有房子很多都想要读完书回厦门工作的，厦门工资总体不高是因为底层人太多了"

**Golden Labels**: {5, 6, 7}
**Most models**: {5, 6} (missed RL7)
**Qwen3-235B-A22B (Few-shot)**: {5, 6, 7} (correct)

**Analysis**: "底层人" (bottom-tier people) is a subtle occupational class reference. Only one model correctly identified this RL7 marker.

## Spurious Analysis

### Overview

- **Total RL7 spurious samples**: 25
- **Categories**: 4 main categories with different error patterns
- **Most common**: Other category (64%) and Work Related (28%)

### Category 1: Work Related (7 samples - 28%)

#### Typical Example: 跨区打工人
**Text**: "房山柿子不行，房山最大的特产是跨区打工人..."  
**Golden Labels**: {} (No RL7)  
**Models that incorrectly predicted RL7**: Qwen3-32B, Qwen3-235B-A22B, Gemma-3-27B (Few-shot)

**Why this is a false positive**:
- "跨区打工人" refers to people who work across districts
- This is a general work description, not occupational typification
- Models incorrectly interpret work-related terms as occupational identity markers

**Key Pattern**: Models over-interpret work references as occupational typification

**Other examples in this category**:
- "工作都是互联网圈子" (work in internet industry)
- "公务员分数线" (civil service exam scores)
- "工人或者事业单位公务员" (workers or civil servants)

### Category 2: Social Class (1 sample - 4%)

#### Typical Example: 老市民心理落差
**Text**: "是老市民的心理落差……祖祖辈辈都是大城市武汉人， 混了几辈人还没到大城市上层，眼看这些祖辈都在乡下耕田的外地人来大武汉跟自己平起平坐，甚至混得比自己还好，失落……唯有排外能强挽一下老市民的尊严..."  
**Golden Labels**: {3} (Family Rootedness)  
**Models that incorrectly predicted RL7**: Qwen3-32B (Few-shot)

**Why this is a false positive**:
- "大城市上层" refers to social class hierarchy
- "平起平坐" refers to social equality
- This is about social class, not occupational identity
- Models confuse social hierarchy with occupational typification

**Key Pattern**: Models confuse social class references with occupational typification

### Category 3: Economic Terms (1 sample - 4%)

#### Typical Example: 祠堂分红
**Text**: "其实是反过来的，村民以前在老广眼里是乡下人，只不过郊区城镇化之后村民平均生活水平比之前的老城区市民高，现在基本在广州会粤语都当你广州人，只不过有些人挑拨离间，广州人又要有祠堂分红又要在老三区，能满足这俩点的全广州找不出20万个，那还得了..."  
**Golden Labels**: {4} (Linguistic-Cultural Recognition)  
**Models that incorrectly predicted RL7**: Qwen3-235B-A22B (Few-shot)

**Why this is a false positive**:
- "祠堂分红" refers to economic benefits from ancestral halls
- "生活水平" refers to living standards
- These are economic terms, not occupational identity markers
- Models interpret economic references as occupational typification

**Key Pattern**: Models interpret economic terms as occupational markers

### Category 4: Other (16 samples - 64%)

#### Typical Example 1: 拆迁户
**Text**: "不会吧渝中区的拆迁户跑李家沱去干撒子，..."  
**Golden Labels**: {} (No RL7)  
**Models that incorrectly predicted RL7**: Gemma-3-27B (Zero-shot)

**Why this is a false positive**:
- "拆迁户" refers to people whose homes were demolished for development
- This is a residential status, not an occupational identity
- Models incorrectly associate residential status with occupational typification

#### Typical Example 2: 农转非
**Text**: "哈哈哈哈 我土著我倒是还好，但我身边的土著亲戚朋友 确实看不起外地的 而且三环外的都不行 我亲戚还说我男朋友是农转非..."  
**Golden Labels**: {1, 2} (Spatial Authority, Administrative Legitimacy)  
**Models that incorrectly predicted RL7**: Qwen3-235B-A22B, Gemma-3-27B (Few-shot), Gemma-3-27B (Zero-shot)

**Why this is a false positive**:
- "农转非" refers to agricultural to non-agricultural household registration change
- This is an administrative status change, not an occupational identity
- Models confuse administrative status with occupational typification

#### Typical Example 3: 城里人 vs 乡里别
**Text**: "哈是乡里别，哪有城里人..."  
**Golden Labels**: {1} (Spatial Authority)  
**Models that incorrectly predicted RL7**: Qwen3-32B (Few-shot)

**Why this is a false positive**:
- "城里人" and "乡里别" refer to urban vs rural identity
- This is spatial/geographic identity, not occupational identity
- Models confuse spatial identity with occupational typification

#### Typical Example 4: 上流社会
**Text**: "历下:设施齐全去哪里都方便 市中:万达大观园老商埠偶尔逛逛 天桥:小时候买衣服长大了搞装修偶尔去逛农贸市场 高新:都是上流社会的人 绿化挺好 槐荫:大剧院西客站宜家（？忘了是不是在槐荫了） 历城:没去过但是知道在市里 章丘 长清平阴:已经不在市里了..."  
**Golden Labels**: {1, 5} (Spatial Authority, Functional Livability)  
**Models that incorrectly predicted RL7**: Qwen3-235B-A22B (Few-shot)

**Why this is a false positive**:
- "上流社会" refers to upper class social status
- "搞装修" refers to renovation work
- These are social class and work activities, not occupational identity markers
- Models confuse social class and work activities with occupational typification

**Key Pattern**: Models make spurious connections between various terms and occupational identity

### Model-Specific Patterns

#### Qwen3-235B-A22B (Most Prone to Spurious RL7)
**Typical Error**: Over-predicts RL7 in various contexts
- Work-related terms: "公务员", "跨区打工人"
- Social terms: "上流社会", "农转非"
- Economic terms: "祠堂分红"

#### Qwen3-32B (Moderate Spurious RL7)
**Typical Error**: Predicts RL7 for spatial identity terms
- "乡里别", "城里人"
- Social class references: "大城市上层"

#### Gemma-3-27B (Variable Spurious RL7)
**Typical Error**: Inconsistent patterns
- Sometimes predicts RL7 for residential status: "拆迁户"
- Sometimes predicts RL7 for administrative status: "农转非"

#### Zero-shot Models (Rare Spurious RL7)
**Typical Error**: Very few spurious predictions
- Only 5 out of 25 cases involve zero-shot models
- Most spurious predictions come from few-shot models

## Occupational Stereotype Analysis

### Overview

- **Total Occupational Stereotype samples**: 3
- **Total predictions**: 18 (6 models × 3 samples)
- **Overall accuracy**: 22.2% (4/18 correct)
- **Few-shot accuracy**: 44.4% (4/9 correct)
- **Zero-shot accuracy**: 0.0% (0/9 correct)

### Sample-by-Sample Analysis

#### Sample 1: "底层人" (Bottom-tier People)

**Text**: "其实没必要比，厦门本地人只要在厦门有房子很多都想要读完书回厦门工作的，厦门工资总体不高是因为底层人太多了，实际上厦门的国企包括公务员的工资平均都要比福州高很多..."

**Golden Labels**: {5, 6, 7}
**Occupational Keywords**: ['底层人', '国企', '公务员', '工资']

**Model Predictions**:
- **Qwen3-32B (Few-shot)**: {5, 6} ✗ (missed RL7)
- **Qwen3-235B-A22B (Few-shot)**: {5, 6, 7} ✓ (correct)
- **Gemma-3-27B (Few-shot)**: {5, 6} ✗ (missed RL7)
- **All Zero-shot models**: {} or {5, 6} ✗ (missed RL7)

**Analysis**: Only Qwen3-235B-A22B (Few-shot) correctly identified the occupational class reference "底层人" (bottom-tier people) as RL7. The term indicates occupational hierarchy and class-based identity.

#### Sample 2: "骚扰电话" (Harassment Calls)

**Text**: "就是。。。基本上现在接到那些骚扰电话都是外地口音哎，不是歧视外地人，只是觉得如果他们回老家，做点其他工作，也比鼓捣留在成都打骚扰电话好嘛"

**Golden Labels**: {4, 7}
**Occupational Keywords**: ['骚扰电话', '其他工作', '打工']

**Model Predictions**:
- **Qwen3-32B (Few-shot)**: {4, 7} ✓ (correct)
- **Qwen3-235B-A22B (Few-shot)**: {4} ✗ (missed RL7)
- **Gemma-3-27B (Few-shot)**: {4} ✗ (missed RL7)
- **All Zero-shot models**: {} or {4} ✗ (missed RL7)

**Analysis**: Only Qwen3-32B (Few-shot) correctly identified the occupational stereotype about harassment call workers. The text implies a negative occupational stereotype about certain types of work.

#### Sample 3: "外来打工人" (Migrant Workers)

**Text**: "丰台：有点偏，外来打工人租房子或者住员工宿舍的地方，有些地方还没开发出来，稍微有点沧桑"

**Golden Labels**: {1, 5, 7}
**Occupational Keywords**: ['外来打工人', '员工宿舍', '程序员聚集地']

**Model Predictions**:
- **Qwen3-32B (Few-shot)**: {1} ✗ (missed RL7)
- **Qwen3-235B-A22B (Few-shot)**: {1, 5, 7} ✓ (correct)
- **Gemma-3-27B (Few-shot)**: {1, 7} ✓ (correct)
- **All Zero-shot models**: {} or {1} ✗ (missed RL7)

**Analysis**: Two few-shot models (Qwen3-235B-A22B and Gemma-3-27B) correctly identified the occupational reference to "外来打工人" (migrant workers) as RL7. This sample had the best performance among the three.

### Model Performance Comparison

#### Overall Performance by Model

| Model | Correct/Total | Accuracy |
|-------|---------------|----------|
| **Qwen3-235B-A22B (Few-shot)** | 2/3 | **66.7%** |
| **Qwen3-32B (Few-shot)** | 1/3 | 33.3% |
| **Gemma-3-27B (Few-shot)** | 1/3 | 33.3% |
| **All Zero-shot models** | 0/3 | 0.0% |

#### Performance by Prompting Method

| Method | Correct/Total | Accuracy |
|--------|---------------|----------|
| **Few-shot** | 4/9 | 44.4% |
| **Zero-shot** | 0/9 | 0.0% |

### Occupational Stereotype Characteristics

#### 1. Subtle Occupational References
- **"底层人"**: Indirect reference to occupational class hierarchy
- **"外来打工人"**: Direct occupational category reference
- **"骚扰电话"**: Occupational stereotype about specific work type

#### 2. Context-Dependent Recognition
- Models need to understand the **occupational implications** of terms
- **Cultural context** is important for recognizing occupational stereotypes
- **Subtle class distinctions** are harder to identify than explicit references

#### 3. Keyword Patterns
- **Direct occupational terms**: "外来打工人", "打工"
- **Class-based terms**: "底层人", "国企", "公务员"
- **Occupational stereotypes**: "骚扰电话", "员工宿舍"

## Key Challenges for RL7 Identification

### 1. Subtle Occupational References
- **Problem**: Models miss indirect occupational class markers
- **Examples**: "底层人", "弯脚杆", "铁饭碗"
- **Solution**: Add more examples of subtle occupational references

### 2. Occupational Slurs and Stereotypes
- **Problem**: Models don't recognize occupational slurs as RL7
- **Examples**: "弯脚杆", "乡里伢"
- **Solution**: Include occupational slurs in training examples

### 3. Professional Status Indicators
- **Problem**: Models miss professional hierarchy references
- **Examples**: "职称高低", "铁饭碗", "底层人"
- **Solution**: Add examples of professional status markers

### 4. Industry-Specific References
- **Problem**: Models miss industry-related occupational markers
- **Examples**: "武钢的", "批发市场", "打工"
- **Solution**: Include more industry-specific examples

## Key Insights

### 1. Definition Confusion
Models struggle to distinguish between:
- **Occupational typification** (RL7): Identity based on professional/occupational associations
- **Work activities**: General work descriptions
- **Social class**: Economic/social hierarchy
- **Spatial identity**: Urban/rural distinctions
- **Administrative status**: Official classifications

### 2. Trigger Words
Common terms that trigger false positives:
- **Work-related**: "工作", "公务员", "打工人", "老板"
- **Social**: "上流社会", "城里人", "乡里别"
- **Status**: "拆迁户", "农转非"
- **Economic**: "分红", "生活水平"

### 3. Model Behavior
- **Few-shot models**: More prone to spurious predictions (23/25 cases)
- **Zero-shot models**: Rarely make spurious predictions (5/25 cases)
- **Larger models**: Not necessarily better at avoiding false positives

## Recommendations for Improvement

### 1. Prompt Engineering

#### Add RL7-Specific Examples
- Include occupational slurs and stereotypes
- Add professional status indicators
- Include industry-specific references
- Provide examples of subtle occupational class markers

#### Add Occupational Stereotype Examples
- Include examples with "底层人", "外来打工人", "打工仔"
- Add occupational class hierarchy examples
- Provide examples of occupational stereotypes

#### Negative Examples
- Clarify what is NOT RL7
- Distinguish RL7 from other RL categories
- Provide counter-examples
- Clarify what is NOT occupational typification
- Distinguish from other RL categories
- Provide counter-examples

### 2. Training Data Enhancement

#### Occupational Categories to Include
- **Occupational slurs**: "弯脚杆", "乡里伢", "打工仔"
- **Professional status**: "铁饭碗", "职称", "底层人"
- **Industry markers**: "武钢的", "批发市场", "单位"
- **Class distinctions**: "富人", "穷人", "有钱人"

#### Occupational Stereotype Categories
- **Class-based terms**: "底层人", "富人", "穷人"
- **Occupational categories**: "外来打工人", "打工仔", "白领"
- **Industry stereotypes**: "骚扰电话", "程序员", "公务员"
- **Workplace references**: "员工宿舍", "单位", "公司"

#### Context Variations
- Different cities and regions
- Various occupational sectors
- Different social contexts
- Subtle vs. explicit references

### 3. Model-Specific Strategies

#### Few-shot Models
- **Focus**: Improve RL7 identification accuracy
- **Strategy**: Add more RL7 examples in prompts
- **Target**: Reduce missed rate from 86.7% to <50%
- **Focus**: Improve occupational stereotype recognition
- **Strategy**: Add more occupational stereotype examples
- **Target**: Increase accuracy from 44.4% to >70%

#### Zero-shot Models
- **Focus**: Enable basic RL7 recognition
- **Strategy**: Better prompt engineering
- **Target**: Achieve some RL7 identification capability
- **Focus**: Enable basic occupational stereotype recognition
- **Strategy**: Better prompt engineering
- **Target**: Achieve some occupational stereotype identification capability

## Conclusion

RL7 (Occupational Typification) is the most challenging RL category with an 84.4% missed rate. The main challenges are:

1. **Subtle occupational references** that models fail to recognize
2. **Occupational slurs and stereotypes** that are not identified as RL7
3. **Professional status indicators** that are missed
4. **Industry-specific references** that require domain knowledge

Occupational stereotypes are particularly challenging for models to identify, with an overall accuracy of only 22.2%. Key findings:

1. **Zero-shot models completely fail** (0% accuracy)
2. **Few-shot models show limited capability** (44.4% accuracy)
3. **Qwen3-235B-A22B (Few-shot) performs best** (66.7% accuracy)
4. **Subtle occupational references** are harder to identify
5. **Cultural context** is crucial for recognition

The analysis shows that few-shot prompting helps significantly (some models correctly identify RL7 in 46.7% of cases), but there's still substantial room for improvement through better prompt engineering and training data enhancement.

The findings suggest that occupational stereotypes require significant improvement through better prompt engineering and training data enhancement, particularly for recognizing subtle occupational class references and cultural context. 