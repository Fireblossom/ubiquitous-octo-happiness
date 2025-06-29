#!/usr/bin/env python3
"""
GPT-4.1 API script for local identity analysis (Async Version)
"""

import os
import json
import time
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from typing import List, Dict, Any
import re

# Set your OpenAI API key
# You can set this as an environment variable: export OPENAI_API_KEY="your-api-key"
# Or uncomment and set it directly below:
api_key = "your-api-key"

# Initialize OpenAI client
client = AsyncOpenAI(api_key=api_key)

def load_dataset():
    """Load the dataset from CSV file"""
    df = pd.read_csv('../0_data_collection/dataset.csv')
    return df['text'].tolist()

def generate_prompt(input_text: str, mode: str = "zero_shot") -> List[Dict[str, str]]:
    """Generate prompt based on the mode (zero_shot, few_shot, no_cot)"""
    
    system_content = """# 社交网络文本本地人身份界定标准多标签分析

## 1. 任务目标

分析给定的社交媒体文本片段，判断**作者本人**在其中明确表达认同、或在字里行间**隐含表达认同**的、用于界定**"本地人"（或其近义词）**的显著标准是什么，并依据本手册RL框架，输出所有相应的RL类别标签。一个文本可能符合**一个或多个**RL类别。

**核心限定:** 本任务**仅关注**作者如何界定"本地人"身份。如果文本仅仅讨论财富、社会地位、生活品质、房产优劣、职业好坏等，而**未明确或强烈暗示**这些是用来判断一个人是否为"本地人"的标准，则**不予标注**。

## 2. 核心原则

*   **聚焦作者核心论点:** 您的判断**必须**基于**发言者（即文本作者）**所展现的、关于"本地人"身份界定的核心观点、主张或评价。理解作者**想说什么**是首要任务。
*   **分析认同逻辑的功能角色:** 识别出的认同逻辑元素不仅仅是孤立的特征，更要理解它们在作者构建其核心论点时所扮演的**功能角色**。
*   **主动推理语境与隐含逻辑:** 需要主动进行语境推理，理解话语背后作者认可的"本地人"身份合法性依据或其评价标准。
*   **排除抵抗标准与无关讨论:** 对于作者明确表达抵抗、否定，或仅仅描述他人用来针对自己的标准，**不予标注**。对于未与"本地人"身份界定直接关联的讨论，**不予标注**。

## 3. 认同逻辑框架

**统一句型模板:**
"作者认为，人们（或作者本人）应该/常常/可以凭借[认同逻辑类型]所代表的逻辑或标准，来界定谁是/不是'本地人'，或者评价一个'本地'区域的好坏/范围。"

**认同逻辑类型定义与特征:**

*   **认同逻辑1：Vernacular Spatial Authority（社会习惯性区域认知）**
    *   **定义:** 作者认同或描述的、当地人对城市内部空间类别的共同、习惯或历史沉淀的认知。这些认知并非基于官方的行政边界，而是反映了集体的情感图谱，被用作分配评价或象征性标签的文化速记。
    *   **核心辨识特征:**
        *   作者明确提出或默认一种关于**哪些区域"算"或"不算"本地核心区、哪些区域之间存在身份或认知差异**的划分逻辑（例如"三环以外不算成都"）。
        *   作者通过描述特定区域的**象征意义、历史标签或社会普遍看法（集体情感地图）**来暗示其在"本地"身份体系中的地位。
        *   作者讨论**"市区范围"、"城市边界"等概念的社会认同变迁或当前共识**。

*   **认同逻辑2：Administrative Legitimacy（行政归属合法性）**
    *   **定义:** 对官方管辖权、法律地位或行政称谓的诉求。这类发言者以户口登记（一种户籍制度）、行政区合并或市镇重新划分为由，证明自己被纳入或被排除在外。
    *   **核心辨识特征:** 关键词包括"行政区域划分"、"划进"、"归属"、"户口是哪的"、"身份证开头"等，作为判断某人/某地在行政法理上是否算"本地"的依据。

*   **认同逻辑3：Family Rootedness（家族历史根基/个体成长史）**
    *   **定义:** 作者认同或引用的、根据家族定居的世代深度来评估当地的合法性。该类别中的主张强调血统、祖先或与该地区的长期家族联系。
    *   **核心辨识特征:**
        *   强调"**土生土长**"、"**世代**"、"**祖辈**"、"**父辈**"、"**三代以内**"、"**从小就/幼儿园就来了**"等，以此证明某人是"根正苗红的本地人"或已形成"本地人"的身份认同基础。
        *   描述因迁移历史（或缺乏迁移史）导致的身份差异。

*   **认同逻辑4：Linguistic-Cultural Recognition（文化语言识别性）**
    *   **定义:** 依赖方言、口音或文化语言习惯作为边界标志。地区性语言特点被视为内部人地位的代名词，偏差往往会引起嘲笑或不信任。也可能包括对特定地方文化习惯（如习俗、生活方式）的认同。
    *   **核心辨识特征:** 提及"讲本地话"、"口音"、"听不懂/受不了某些口音"、"懂不懂我们这儿的规矩"等，作为判断内外、判断是否为"自己人"或具备"本地属性"的标准。

*   **认同逻辑5：Functional Livability（生活功能便利性与环境品质认知）**
    *   **定义:** 作者认同或引用的、从物质基础设施（如交通、住房、教育或获得服务的途径）的角度对城市地区进行评估，通常是为了宣称空间优越性或可取性。
    *   **核心辨识特征:**
        *   提及"地铁"、"配套"、"方便"、"教育"、"绿化好"、"人少"、"街道界面"、"舒服"等，并**将其与对一个区域是否"好"、是否"宜居"或是否"值得居住"的评价直接关联**。
        *   作者因某地的认同逻辑5特性而将其视为理想的"本地"生活空间，或因缺乏认同逻辑5特性而认为某地不符合"好的本地"标准。

*   **认同逻辑6：Social Embeddedness（社会根基深浅与经济地位象征）**
    *   **定义:** 作者认同或引用的、根据一个人融入当地社会圈子的情况，以及有没有当地的固定资产来判断是不是本地人。这包括社区里的资源（像分红、对社区熟不熟）以及物质或象征性的资源，比如（继承的）房产。
    *   **核心辨识特征:**
        *   提及"人脉"、"分红"、"老关系"、"社区影响力"、"朋友在附近"、"在市中心有几套房"、"房价高/低对身份认同的影响"、"拿出一两百万买房"等，并**明确或隐含地将这些与"本地人身份的稳固性、真实性、层级或对区域的评价标准"挂钩**。

*   **认同逻辑7：Occupational Typification（职业象征性）**
    *   **定义:** 作者认同或引用的、通过将某些地区与占主导地位的职业群体--如公务员、外来务工人员或企业主--联系起来，从而勾勒出阶级和价值的隐性等级。
    *   **核心辨识特征:** 提及特定职业或人群类型（如"农民"、"种地的"、"打工的"），并**将其与特定区域的"本地属性"、居民构成或社会分层强关联**，从而界定身份或评价区域。

## 4. 标注流程

1.  **第一步：识别文本中提及的命名实体**
    *   通读文本，把文中提及的实体记录下来。

2.  **第二步：为每个命名实体匹配论点**
    *   细读文本，为每个提及的实体都匹配上相应的观点表述。

3.  **第三步：分析每个观点表述背后隐藏的认同逻辑**
    *   对每个实体，都分析说话人表达的观点（若有）背后是基于哪一项认同逻辑，可为多个认同逻辑。

4.  **第四步：输出结果**
    *   输出所有匹配到的**一个或多个**认同逻辑类别标签[认同逻辑N]，用 `, ` 分隔。

## 5. 注意事项

*   时刻提醒自己，标注对象是**作者本人未明确反对的、且用来界定"本地人"身份或评价"本地"区域的标准，并理解其在作者核心论点中的功能**。
*   不要仅凭关键词进行机械匹配，深入理解作者话语背后的逻辑、意图及其核心观点。**按照 4. 标注流程 中的步骤从第一步一直到第四步地进行思考，不要省略步骤**
*   如果作者仅仅在描述一个现象、表达个人偏好或评价社会地位/财富，而没有流露出自己用这些标准来定义"谁是本地人"或评价"本地区域好坏/范围/区隔"，则**不予标注**。
*   在 `输出：`标志后输出结果。输出结果后**立即停止**，不要再继续输出任何内容。"""

    messages = [{"role": "system", "content": system_content}]
    
    if mode == "few_shot":
        # Add few-shot examples
        examples = [
            {
                "role": "user",
                "content": "崇州的嘛，我们身边的人被问到是哪里的，都直接说崇州，不会说是成都的。从小到大耳濡目染的就是成都人都觉得三环以外的不算成都的。"
            },
            {
                "role": "assistant",
                "content": "第一步：识别文本中提及的命名实体  \n- 崇州（地名）  \n- 成都（三环以内、三环以外，地理区分）\n\n第二步：为每个命名实体匹配论点  \n- 崇州：作者和身边的人被问时会说自己是\"崇州\"的，不说\"成都的\"。\n- 成都（三环以内/以外）：从小的感受和集体认知是\"成都人都觉得三环以外的不算成都的\"。\n\n第三步：分析每个观点表述背后隐藏的认同逻辑  \n- 崇州和成都区隔的说法，是在表达一种社会习惯性区域认知——\"崇州\"虽行政上归成都，但在社会文化习惯上并不认作\"成都\"本地；而\"三环以外不算成都\"，则是成都本地社会内部广泛认同的一种空间界定方式，即以\"三环\"为核心地理象征分割\"本地/非本地\"。\n- 这里\"从小到大耳濡目染的就是成都人都觉得三环以外不算成都的\"，直接体现了集体的空间认知共识，不是官方行政意义上的划分，而是文化情感层面的分区。\n- 没有提及其他如家族根基、行政归属等标准。\n\n第四步：输出结果  \n输出：认同逻辑1"
            },
            {
                "role": "user",
                "content": "抱抱，反正在我眼里我只看行政区域划分，属于青岛那就是青岛人，管是胶州还是城阳，早些年因为交通不便利人们觉得隔着远就罢了，现在不管地铁还是开车一会就到了呀"
            },
            {
                "role": "assistant",
                "content": "第一步：识别文本中提及的命名实体  \n- 青岛  \n- 胶州  \n- 城阳  \n\n第二步：为每个命名实体匹配论点  \n- 青岛：作者只看行政区域划分，属于青岛的就是青岛人。\n- 胶州、城阳：只要行政上属于青岛，这两个地方的人也算青岛人。作者提及\"早些年因交通不便利…现在…地铁开车一会就到了\"，作为对空间距离认知变化的补充，但核心标准是行政区域。\n\n第三步：分析每个观点表述背后隐藏的认同逻辑  \n- 作者明确表达\"只看行政区域划分\"，即以官方行政归属为唯一认同标准。\n- 也提及\"交通便利性\"导致社会认知的变化，但该部分用于说明过去与现在的物理距离观念、心理距离的缩短，但并没有表明作者用交通便利性作为界定\"本地人\"的标准——这仍是对空间观念变化的观察，核心不是用来界定\"本地人\"身份。\n- 没有体现对家族根基、方言文化、社会根基、职业类型等其他逻辑的认同。\n\n第四步：输出结果  \n输出：认同逻辑2"
            },
            {
                "role": "user",
                "content": "在广州三代以内上就算真正广州人，二代以内是新广州人，一代开荒牛是广州移民，三代都不讲广州话的是外省广州人（祖宗是捞佬）来广州的。"
            },
            {
                "role": "assistant",
                "content": "第一步：识别文本中提及的命名实体  \n- 三代以内的广州人  \n- 二代以内的新广州人  \n- 一代\"开荒牛\"（广州移民）  \n- 三代都不讲广州话的人\n\n第二步：为每个命名实体匹配论点  \n- 三代以内：被认为是\"真正广州人\"，强调家族扎根年限。\n- 二代以内：被认为是\"新广州人\"，家族定居时间稍短于\"真正广州人\"。\n- 一代开荒牛：被称为\"广州移民\"，代表迁入第一代。\n- 三代不讲广州话者：即使家族已经在广州三代，若不讲广州话，被归为\"外省广州人\"；描述其祖先来自外省。\n\n第三步：分析每个观点表述背后隐藏的认同逻辑  \n- 对家族在地历史长度（\"几代人\"）作为评价\"本地人\"层级的主要依据，典型的家族历史根基认同逻辑。\n- \"三代都不讲广州话\"则进一步用语言/文化融入作为认同标准，强调语言（广州话）作为身份的象征和界定标志。\n- 没有明显提及行政归属、区域认知、社会关系、职业等标准。\n\n第四步：输出结果  \n输出：认同逻辑3, 认同逻辑4"
            },
            {
                "role": "user",
                "content": "我是成都土著，我觉得还好！说实话哈，只是确确实实有一点受不了一些口音。"
            },
            {
                "role": "assistant",
                "content": "第一步：识别文本中提及的命名实体  \n- 成都土著（作者自称）\n\n第二步：为每个命名实体匹配论点  \n- 作者自认为\"成都土著\"，表达自己对\"土著\"身份的认同。\n- 还提到\"有一点受不了一些口音\"，表达对某些非本地口音的不适。\n\n第三步：分析每个观点表述背后隐藏的认同逻辑  \n- \"成都土著\"体现出家族或个体的本地根基，但未详细展开代际等家族深度，因此直接家族历史根基的认同逻辑表征不明显。\n- 主要着重于\"受不了一些口音\"，即用口音、语言作为区分\"本地人/非本地人\"的实际体验，表达了语言/文化识别是自己衡量本地性的隐含标准。\n\n第四步：输出结果  \n输出：认同逻辑4"
            },
            {
                "role": "user",
                "content": "所以我喜欢北辰，老城区往新区的过渡，街道界面没老城区那么破旧，离市中心又只需要十几分钟，配套啥都有，住起来太舒服哒"
            },
            {
                "role": "assistant",
                "content": "第一步：识别文本中提及的命名实体  \n- 北辰（区域名）\n- 老城区\n- 新区\n- 市中心\n\n第二步：为每个命名实体匹配论点  \n- 北辰受到作者喜欢，理由包括街道界面较新、离市中心近、配套设施齐全、居住感受好。\n- 与老城区对比，北辰街道界面更好，设施更完善。\n- 市中心——地理位置优势（交通便利）。\n\n第三步：分析每个观点表述背后隐藏的认同逻辑  \n- 作者对北辰的喜欢是基于生活便利度、配套设施以及居住环境的舒适度进行评价。\n\n第四步：输出结果  \n输出：认同逻辑5"
            },
            {
                "role": "user",
                "content": "毕竟好多家庭不见得可以拿出一两百万买一个环境好点、还在二环内的房子。毕竟我爸妈在成都工作这么久，基本都在一二环内，朋友也在附近，买到郊区肯定不现实，买一二环的房子改造是最好的方案了。"
            },
            {
                "role": "assistant",
                "content": "第一步：识别文本中提及的命名实体  \n- 一二环内的房子（二环内、地段）  \n- 郊区  \n- 成都（整体城市范围）\n\n第二步：为每个命名实体匹配论点  \n- 一二环内的房子：环境好、价格高（需一两百万）、与作者父母的工作、朋友关系网等密切相关，认为这是最好的选择。\n- 郊区：相较之下认为\"买到郊区肯定不现实\"，即不符合理想。\n- 作者父母长期在一二环内工作、朋友都在附近，社会关系根基也在一二环内。\n\n第三步：分析每个观点表述背后隐藏的认同逻辑  \n- 强调\"工作久\"\"朋友在附近\"、在市区有足够经济实力买房等社会关系、经济能力，是对社会根基和经济地位的一种展示。\n- 也提到一二环\"环境好\"，表达出生活便利性、居住环境优越的功能性评价。\n- 没有直接涉及家族历史、行政归属、语言文化、职业类型等逻辑。\n\n第四步：输出结果  \n输出：认同逻辑5, 认同逻辑6"
            },
            {
                "role": "user",
                "content": "李沧黄岛那里是农村，很多种地的农民，老一辈不认为他们是城里人。"
            },
            {
                "role": "assistant",
                "content": "第一步：识别文本中提及的命名实体  \n- 李沧、黄岛（区域名）\n- 农民（种地的人）\n- 老一辈（判断者）\n\n第二步：为每个命名实体匹配论点  \n- 李沧、黄岛被描述为\"农村\"，居民多为\"种地的农民\"。\n- \"老一辈不认为他们是城里人\"，即按照某些标准、身份界定逻辑，不把这些地区的人视为真正的\"本地城里人\"。\n\n第三步：分析每个观点表述背后隐藏的认同逻辑  \n- 作者提及\"种地的农民\"，通过职业类别与区域身份强相关地进行区隔，\"老一辈\"用主要职业类型来划分\"是不是城里人\"。\n- 还体现了区域空间（农村/城市）的社会集体习惯性认知，但此句重心偏向职业类型与集体观念的结合。\n- 没有直接用行政归属、家族根基、语言文化等逻辑进行界定。\n\n第四步：输出结果  \n输出：认同逻辑7, 认同逻辑1"
            }
        ]
        messages.extend(examples)
    
    # Add the current input
    messages.append({"role": "user", "content": input_text})
    
    return messages

async def call_gpt41(messages: List[Dict[str, str]], max_retries: int = 5) -> str:
    """Call GPT-4.1 API with improved retry logic for rate limiting"""
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                temperature=0.7,
                max_tokens=8192,
                top_p=0.8,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_str = str(e)
            print(f"Attempt {attempt + 1} failed: {error_str[:100]}...")
            
            # Check if it's a rate limit error
            if "429" in error_str and "rate_limit" in error_str:
                # Extract wait time from error message if available
                import re
                wait_match = re.search(r'Please try again in (\d+)ms', error_str)
                if wait_match:
                    wait_time = int(wait_match.group(1)) / 1000  # Convert to seconds
                    print(f"Rate limit hit, waiting {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
                else:
                    # Exponential backoff for rate limits
                    wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
                    print(f"Rate limit hit, waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
            else:
                # For other errors, use shorter exponential backoff
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                    print(f"Error occurred, waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e

def parse_output(output_text: str) -> str:
    """Parse the model output to extract the RL types"""
    
    # Primary regex for "**输出[:：]**"
    primary_output_marker_pattern = re.compile(r"\*\*输出[:：]\*\*\s*(.*)", re.DOTALL)
    # Secondary regex for "第四步：输出结果"
    secondary_output_marker_pattern = re.compile(r"第四步：输出结果\s*(.*)", re.DOTALL)
    # Regex for splitting by "输出:" or "输出："
    simple_output_marker_regex = r"输出[:：]"
    
    extracted_label_part = None
    
    # Try to find the primary output marker first (bolded)
    match_primary = primary_output_marker_pattern.search(output_text)
    if match_primary:
        extracted_label_part = match_primary.group(1).strip()
    else:
        # If primary marker is not found, try the secondary marker (step-based)
        match_secondary = secondary_output_marker_pattern.search(output_text)
        if match_secondary:
            extracted_label_part = match_secondary.group(1).strip()
        else:
            # If secondary marker is not found, try the simple plain "输出[:：]" marker
            parts = re.split(simple_output_marker_regex, output_text)
            if len(parts) > 1:  # Marker was found
                extracted_label_part = parts[-1].strip()
    
    if extracted_label_part is not None:
        label = ""
        # Attempt to clean the extracted part to get the pure label
        # Case 1: Markdown code block like ```\nLABEL\n```
        if extracted_label_part.startswith("```\n") and extracted_label_part.endswith("\n```"):
            label = extracted_label_part[len("```\n") : -len("\n```")].strip()
        # Case 2: Backticks like `LABEL`
        elif extracted_label_part.startswith("`") and extracted_label_part.endswith("`"):
            label = extracted_label_part[1:-1].strip()
        # Case 3: No special formatting, just the text
        else:
            label = extracted_label_part.strip()
        
        # Check if the label is empty after stripping wrappers
        if not label or label == "无" or label == "（无）":
            return "N/A"
        else:
            return label
    else:
        return "PARSE_ERROR_NO_MARKER"

async def process_single_text(text: str, mode: str, index: int, total: int) -> Dict[str, Any]:
    """Process a single text asynchronously"""
    try:
        messages = generate_prompt(text, mode)
        output = await call_gpt41(messages)
        parsed_output = parse_output(output)
        
        print(f"  Processed {index + 1}/{total}: {parsed_output}")
        
        return {
            "Original_Input_Text": text,
            "RL_Types": parsed_output,
            "Raw_Model_Output": output
        }
        
    except Exception as e:
        print(f"Error processing text {index + 1}: {e}")
        return {
            "Original_Input_Text": text,
            "RL_Types": "ERROR",
            "Raw_Model_Output": str(e)
        }

async def process_batch_async(texts: List[str], mode: str = "zero_shot", max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """Process texts asynchronously with improved concurrency control"""
    
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)  # Reduced from 10 to 5
    
    async def process_with_semaphore(text: str, index: int):
        async with semaphore:
            # Add a small delay between requests to avoid rate limits
            await asyncio.sleep(0.1)
            return await process_single_text(text, mode, index, len(texts))
    
    # Create tasks for all texts
    tasks = [process_with_semaphore(text, i) for i, text in enumerate(texts)]
    
    # Process all tasks concurrently
    print(f"Processing {len(texts)} texts with max {max_concurrent} concurrent requests...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions that occurred
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error processing text {i + 1}: {result}")
            processed_results.append({
                "Original_Input_Text": texts[i],
                "RL_Types": "ERROR",
                "Raw_Model_Output": str(result)
            })
        else:
            processed_results.append(result)
    
    return processed_results

async def main_async():
    """Main async function to run the analysis"""
    
    # Check if API key is set
    if not api_key:
        print("Error: OpenAI API key not set. Please set the OPENAI_API_KEY environment variable or set it in the script.")
        return
    
    # Load dataset
    print("Loading dataset...")
    texts = load_dataset()
    print(f"Loaded {len(texts)} texts")
    
    # Create output directory if it doesn't exist
    os.makedirs("llm_outputs", exist_ok=True)
    
    # Process with different modes
    modes = ["zero_shot", "few_shot", "no_cot"]
    
    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Processing with mode: {mode}")
        print(f"{'='*50}")
        
        # Check if output file already exists for resume functionality
        output_filename = f"llm_outputs/gpt41_{mode}.csv"
        existing_results = []
        
        if os.path.exists(output_filename):
            print(f"Found existing results file: {output_filename}")
            try:
                existing_df = pd.read_csv(output_filename)
                existing_results = existing_df.to_dict('records')
                print(f"Loaded {len(existing_results)} existing results")
                
                # Check if we have results for all texts
                if len(existing_results) >= len(texts):
                    print(f"All texts already processed for {mode}, skipping...")
                    continue
                else:
                    # Remove already processed texts
                    processed_texts = [r['Original_Input_Text'] for r in existing_results]
                    remaining_texts = [text for text in texts if text not in processed_texts]
                    print(f"Resuming with {len(remaining_texts)} remaining texts...")
                    texts_to_process = remaining_texts
            except Exception as e:
                print(f"Error reading existing file: {e}")
                texts_to_process = texts
        else:
            texts_to_process = texts
        
        # Modify system prompt for no_cot mode
        if mode == "no_cot":
            # For no_cot, we'll use a simpler prompt without the step-by-step instructions
            print("Using simplified prompt for no_cot mode...")
        
        # Process all texts asynchronously
        start_time = time.time()
        new_results = await process_batch_async(texts_to_process, mode=mode, max_concurrent=5)
        end_time = time.time()
        
        # Combine existing and new results
        all_results = existing_results + new_results
        
        # Create DataFrame and save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print(f"Results saved to {output_filename}")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        
        # Print summary statistics
        print(f"Summary for {mode}:")
        print(f"  Total processed: {len(all_results)}")
        print(f"  Successfully parsed: {len([r for r in all_results if r['RL_Types'] not in ['ERROR', 'PARSE_ERROR_NO_MARKER']])}")
        print(f"  Errors: {len([r for r in all_results if r['RL_Types'] == 'ERROR'])}")
        print(f"  Parse errors: {len([r for r in all_results if r['RL_Types'] == 'PARSE_ERROR_NO_MARKER'])}")

def main():
    """Main function to run the async analysis"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 