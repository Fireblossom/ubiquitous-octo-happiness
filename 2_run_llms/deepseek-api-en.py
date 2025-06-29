#!/usr/bin/env python3
"""
DeepSeek API script for local identity analysis (English Version)
"""

import os
import json
import time
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from typing import List, Dict, Any
import re

# Set your DeepSeek API key
# You can set this as an environment variable: export DEEPSEEK_API_KEY="your-api-key"
# Or uncomment and set it directly below:
api_key = "your-api-key"

# Initialize DeepSeek client
# DeepSeek uses OpenAI-compatible API format
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"  # DeepSeek API endpoint
)

def load_dataset():
    """Load the dataset from CSV file"""
    df = pd.read_csv('../0_data_collection/dataset.csv')
    return df['text'].tolist()

def generate_prompt(input_text: str, mode: str = "zero_shot") -> List[Dict[str, str]]:
    """Generate prompt based on the mode (zero_shot, few_shot, no_cot)"""
    
    system_content = """## Multi-Label Analysis of "Local" Identity Definition Standards in Social Network Texts

### 1. Task Objective

Analyze the given social media text segment to identify the prominent standards that the **author themself** explicitly expresses agreement with, or **implicitly agrees with** through the subtext, for defining a **"local person" (or its synonyms)**. Based on the Recognition Logic (RL) framework in this manual, output all corresponding RL category labels. A text segment may correspond to **one or more** RL categories.

**Core Limitation:** This task **only focuses** on how the author defines "local person" identity. If the text merely discusses wealth, social status, quality of life, property advantages/disadvantages, job quality, etc., **without explicitly or strongly implying** that these are standards used to judge whether someone is a "local person," then **do not annotate**.

### 2. Core Principles
*   **Focus on the Author's Core Argument:** Your judgment **must** be based on the core viewpoint, claim, or evaluation regarding the definition of "local person" identity as presented by the **speaker (i.e., the text author)**. Understanding **what the author intends to say** is the primary task.
*   **Analyze the Functional Role of Recognition Logic:** The identified recognition logic elements are not just isolated features; understand the **functional role** they play in the author's construction of their core argument.
*   **Actively Infer Context and Implicit Logic:** Actively perform contextual reasoning to understand the basis of legitimacy for "local person" identity or the evaluation criteria endorsed by the author behind their discourse.
*   **Exclude Resisted Standards and Irrelevant Discussions:** For standards that the author explicitly expresses resistance to, denies, or merely describes as being used against them by others, **do not annotate**. For discussions not directly related to defining "local person" identity, **do not annotate**.

### 3. Recognition Logic Framework

**Uniform Sentence Template:**
The author believes that people (or the author themself) should/often/can use the logic or standards represented by [Recognition Logic Type] to define who is/is not a 'local person', or to evaluate the quality/scope of a 'local' area.

**Recognition Logic Type Definitions and Features:**

#### Recognition Logic 1 (RL1): Vernacular Spatial Authority
*   **Definition:** Shared, habitual, or historically sedimented local perceptions of intra-city spatial categories that the author endorses or describes. These perceptions are not based on official administrative boundaries but reflect collective emotional maps, used as cultural shorthand for assigning evaluative or symbolic labels.
*   **Core Identifying Features:**
    *   The author explicitly proposes or assumes a logic for classifying **which areas "count" or "do not count" as core local areas, or which areas have identity or cognitive differences** (e.g., "Outside the Third Ring Road is not considered Chengdu").
    *   The author implies the status of a specific area within the "local" identity system by describing its **symbolic meaning, historical labels, or common societal views (collective emotional map)**.
    *   The author discusses **the social recognition changes or current consensus on concepts like "urban area scope," "city boundaries," etc.**

#### Recognition Logic 2 (RL2): Administrative Legitimacy
*   **Definition:** Appeals to official jurisdiction, legal status, or administrative designation. Speakers in this category justify inclusion or exclusion based on hukou registration (a household registration system), district incorporation, or municipal redistricting.
*   **Core Identifying Features:** Keywords include "administrative division," "incorporated into," "belongs to," "where is the hukou from," "ID card prefix," etc., used as a basis for judging whether a person/place is administratively/legally "local."

#### Recognition Logic 3 (RL3): Family Rootedness (Family Historical Roots / Individual Growth History)
*   **Definition:** The author endorses or cites the evaluation of local legitimacy based on the generational depth of family settlement. Claims in this category emphasize lineage, ancestry, or long-term familial ties to the area.
*   **Core Identifying Features:**
    *   Emphasizes terms like "**born and raised locally**," "**generations**," "**ancestors**," "**parents' generation**," "**within three generations**," "**came since childhood/kindergarten**," etc., to prove someone is a "genuine local" or has formed the basis of a "local person's" identity.
    *   Describes identity differences due to migration history (or lack thereof).

#### Recognition Logic 4 (RL4): Linguistic-Cultural Recognition
*   **Definition:** Relies on dialect, accent, or cultural linguistic habits as a boundary marker. Regional speech features are treated as proxies for insider status, and deviations often provoke mockery or mistrust. May also include identification with specific local cultural habits (e.g., customs, lifestyle).
*   **Core Identifying Features:** Mentions "speaking the local dialect," "accent," "cannot understand/stand certain accents," "don't you understand our local rules," etc., as criteria for distinguishing insiders from outsiders, or judging if someone is "one of us" or possesses "local attributes."

#### Recognition Logic 5 (RL5): Functional Livability (Convenience of Living Functions and Environmental Quality Perception)
*   **Definition:** The author endorses or cites the evaluation of urban areas in terms of their material infrastructure (e.g., transit, housing, education, or access to services), often to assert spatial superiority or desirability.
*   **Core Identifying Features:**
    *   Mentions "subway," "amenities," "convenience," "education," "good greening," "few people," "streetscape," "comfortable," etc., and **directly associates these with an evaluation of whether an area is "good," "livable," or "worth living in."**
    *   The author considers an area an ideal "local" living space due to its RL5 characteristics, or believes an area does not meet the standard of a "good local" area due to a lack of RL5 characteristics.

#### Recognition Logic 6 (RL6): Social Embeddedness (Depth of Social Roots and Symbolism of Economic Status)
*   **Definition:** The author endorses or cites judging whether someone is a local based on their integration into local social circles and possession of local fixed assets. This includes community resources (like dividends, familiarity with the community) and material or symbolic resources, such as (inherited) property.
*   **Core Identifying Features:**
    *   Mentions "connections (renmai)," "dividends," "old relationships," "community influence," "friends nearby," "has several properties in the city center," "impact of high/low housing prices on identity," "spending one or two million to buy a house," etc., and **explicitly or implicitly links these to the "stability, authenticity, hierarchy of local identity, or evaluation criteria for a region."**

#### Recognition Logic 7 (RL7): Occupational Typification
*   **Definition:** The author endorses or cites framing identity by associating certain districts with dominant professional groups, such as civil servants, migrant workers, or business owners, thereby invoking implicit hierarchies of class and worth.
*   **Core Identifying Features:** Mentions specific occupations or types of people (e.g., "farmers," "those who farm," "migrant workers"), and **strongly associates them with the "local attributes," resident composition, or social stratification of a specific area**, thereby defining identity or evaluating the area.

### 4. Annotation Procedure
1.  **Step 1: Identify named entities mentioned in the text**
    *   Read through the text and record the entities mentioned.
2.  **Step 2: Match arguments to each named entity**
    *   Read the text carefully and match the corresponding viewpoint statements to each mentioned entity.
3.  **Step 3: Analyze the underlying recognition logic behind each viewpoint statement**
    *   For each entity, analyze which recognition logic(s) the speaker's expressed viewpoint (if any) is based on. It can be multiple recognition logics.
4.  **Step 4: Output the results**
    *   Output all matched **one or more** recognition logic category labels [Recognition Logic N], separated by ``, `` (a comma followed by a space).

### 5. Important Notes
*   Constantly remind yourself that the annotation target is **standards that the author themself does not explicitly oppose and uses to define "local person" identity or evaluate a "local" area, and understand their function in the author's core argument.**
*   Do not rely solely on keyword matching; deeply understand the logic, intent, and core viewpoint behind the author's discourse. **Follow the steps from Step 1 to Step 4 in 4. Annotation Procedure; do not skip steps.**
*   If the author is merely describing a phenomenon, expressing personal preference, or evaluating social status/wealth, without revealing that they use these standards to define "who is a local person" or evaluate "the quality/scope/distinction of a local area," then **do not annotate.**
*   Output the result after the `Output: ` marker. **Stop immediately** after outputting the result; do not output any further content."""

    messages = [{"role": "system", "content": system_content}]
    
    if mode == "few_shot":
        # Add few-shot examples
        examples = [
            {
                "role": "user",
                "content": "I'm from the suburbs, and when people ask where I'm from, I always say the suburb name, never the city. Growing up, everyone knew that anything beyond the third ring road doesn't really count as the city proper."
            },
            {
                "role": "assistant",
                "content": "Step 1: Identify named entities mentioned in the text\n- Suburbs (location)\n- City (third ring road, city proper, geographical distinction)\n\nStep 2: Match arguments for each named entity\n- Suburbs: Author and people around them say they're from the suburb when asked, not the city.\n- City (third ring road/city proper): The collective understanding from childhood is that 'anything beyond the third ring road doesn't really count as the city proper.'\n\nStep 3: Analyze the recognition logic hidden behind each viewpoint expression\n- The distinction between suburbs and city expresses a social customary regional cognition—suburbs, though administratively part of the city, are not culturally recognized as 'local' to the city; while 'beyond the third ring road doesn't count as city proper' is a widely recognized spatial definition method within the city's local society, using the 'third ring road' as the core geographical symbol to divide 'local/non-local.'\n- Here 'everyone knew that anything beyond the third ring road doesn't really count as the city proper' directly reflects collective spatial cognitive consensus, not official administrative division but cultural-emotional zoning.\n- No mention of other standards like family foundation, administrative affiliation, etc.\n\nStep 4: Output results\nOutput: Recognition Logic 1"
            },
            {
                "role": "user",
                "content": "I only care about administrative boundaries. If you're within the city limits, you're a city person, whether you're from the outer districts or the suburbs. Years ago people thought the distance was too far, but now with subways and cars, it only takes a few minutes to get anywhere."
            },
            {
                "role": "assistant",
                "content": "Step 1: Identify named entities mentioned in the text\n- City limits\n- Outer districts\n- Suburbs\n\nStep 2: Match arguments for each named entity\n- City limits: Author only cares about administrative boundaries, those within city limits are city people.\n- Outer districts, suburbs: As long as administratively part of the city, people from these areas also count as city people. Author mentions 'years ago people thought distance was too far... now subways and cars take only minutes' as supplement to spatial distance cognition changes, but core standard is administrative boundaries.\n\nStep 3: Analyze the recognition logic hidden behind each viewpoint expression\n- Author explicitly states 'only care about administrative boundaries,' using official administrative affiliation as the sole recognition standard.\n- Also mentions 'transportation convenience' causing social cognition changes, but this part explains past vs. present physical distance concepts, psychological distance shortening, without indicating author uses transportation convenience as standard for defining 'local people'—this remains observation of spatial concept changes, core not for defining 'local people' identity.\n- No reflection of recognition of other logics like family foundation, dialect culture, social foundation, occupational types, etc.\n\nStep 4: Output results\nOutput: Recognition Logic 2"
            },
            {
                "role": "user",
                "content": "In our city, if your family has been here for three generations, you're considered a true local. Two generations makes you a new local, first generation settlers are immigrants, and if three generations don't speak the local dialect, you're an outsider even if your family has been here that long."
            },
            {
                "role": "assistant",
                "content": "Step 1: Identify named entities mentioned in the text\n- Three generations of locals\n- Two generations of new locals\n- First generation settlers (immigrants)\n- Three generations without local dialect\n\nStep 2: Match arguments for each named entity\n- Three generations: Considered 'true locals,' emphasizing family settlement duration.\n- Two generations: Considered 'new locals,' family settlement time slightly shorter than 'true locals.'\n- First generation settlers: Called 'immigrants,' representing first generation of migration.\n- Three generations without local dialect: Even if family has been here three generations, without speaking local dialect, classified as 'outsiders.'\n\nStep 3: Analyze the recognition logic hidden behind each viewpoint expression\n- Using family's local historical length ('several generations') as main basis for evaluating 'local people' hierarchy, typical family historical foundation recognition logic.\n- 'Three generations without local dialect' further uses language/cultural integration as recognition standard, emphasizing dialect as identity symbol and definition marker.\n- No clear mention of administrative affiliation, regional cognition, social relationships, occupation, etc.\n\nStep 4: Output results\nOutput: Recognition Logic 3, Recognition Logic 4"
            },
            {
                "role": "user",
                "content": "I'm a native here and honestly, I can't stand some of these accents. It's just really grating to my ears."
            },
            {
                "role": "assistant",
                "content": "Step 1: Identify named entities mentioned in the text\n- Native (author's self-identification)\n\nStep 2: Match arguments for each named entity\n- Author identifies as 'native,' expressing recognition of 'native' identity.\n- Also mentions 'can't stand some accents,' expressing discomfort with certain non-local accents.\n\nStep 3: Analyze the recognition logic hidden behind each viewpoint expression\n- 'Native' reflects family or individual local foundation, but doesn't detail generational family depth, so direct family historical foundation recognition logic representation is not obvious.\n- Main focus on 'can't stand some accents,' using accent, language as actual experience for distinguishing 'local/non-local people,' expressing language/cultural recognition as implicit standard for measuring localness.\n\nStep 4: Output results\nOutput: Recognition Logic 4"
            },
            {
                "role": "user",
                "content": "That's why I love this neighborhood. It's the transition from old city to new development, the streets aren't as run-down as the old city, it's only 15 minutes to downtown, everything is convenient, it's just so comfortable to live here."
            },
            {
                "role": "assistant",
                "content": "Step 1: Identify named entities mentioned in the text\n- This neighborhood (area name)\n- Old city\n- New development\n- Downtown\n\nStep 2: Match arguments for each named entity\n- This neighborhood is liked by author, reasons include newer streets, close to downtown, convenient facilities, good living experience.\n- Compared to old city, this neighborhood has better streets, more complete facilities.\n- Downtown—geographical advantage (transportation convenience).\n\nStep 3: Analyze the recognition logic hidden behind each viewpoint expression\n- Author's liking for this neighborhood is based on life convenience, supporting facilities, and living environment comfort for evaluation.\n\nStep 4: Output results\nOutput: Recognition Logic 5"
            },
            {
                "role": "user",
                "content": "Not many families can afford to spend a couple hundred thousand on a house in a good environment within the second ring. My parents have worked in the city for so long, mostly within the first and second rings, friends are nearby too, buying in the suburbs just isn't realistic, renovating a house within the first and second rings is the best option."
            },
            {
                "role": "assistant",
                "content": "Step 1: Identify named entities mentioned in the text\n- Houses within first and second rings (location, area)\n- Suburbs\n- City (overall urban scope)\n\nStep 2: Match arguments for each named entity\n- Houses within first and second rings: Good environment, high price (couple hundred thousand), closely related to author's parents' work, friend network, considered best choice.\n- Suburbs: Comparatively 'buying in suburbs just isn't realistic,' not meeting ideals.\n- Author's parents long-term work within first and second rings, friends nearby, social relationship foundation also within first and second rings.\n\nStep 3: Analyze the recognition logic hidden behind each viewpoint expression\n- Emphasizing 'worked long,' 'friends nearby,' having sufficient economic ability to buy houses in urban areas, etc., showing social relationships, economic capability, display of social foundation and economic status.\n- Also mentions first and second rings 'good environment,' expressing functional evaluation of life convenience, superior living environment.\n- No direct involvement of family history, administrative affiliation, language culture, occupational types, etc.\n\nStep 4: Output results\nOutput: Recognition Logic 5, Recognition Logic 6"
            },
            {
                "role": "user",
                "content": "Those areas are rural, lots of farmers working the land, the older generation doesn't consider them real city people."
            },
            {
                "role": "assistant",
                "content": "Step 1: Identify named entities mentioned in the text\n- Those areas (area names)\n- Farmers (agricultural workers)\n- Older generation (judges)\n\nStep 2: Match arguments for each named entity\n- Those areas described as 'rural,' residents mostly 'farmers working the land.'\n- 'Older generation doesn't consider them real city people,' meaning according to certain standards, identity definition logic, these areas' people not considered true 'local city people.'\n\nStep 3: Analyze the recognition logic hidden behind each viewpoint expression\n- Author mentions 'farmers working the land,' using occupational categories to strongly correlate with regional identity for demarcation, 'older generation' uses main occupational types to divide 'whether someone is a city person.'\n- Also reflects social collective customary cognition of regional space (rural/urban), but sentence focus leans toward combination of occupational types and collective concepts.\n- No direct use of administrative affiliation, family foundation, language culture, etc., for definition.\n\nStep 4: Output results\nOutput: Recognition Logic 7, Recognition Logic 1"
            }
        ]
        messages.extend(examples)
    
    # Add the current input
    messages.append({"role": "user", "content": input_text})
    
    return messages

async def call_deepseek(messages: List[Dict[str, str]], max_retries: int = 5) -> str:
    """Call DeepSeek API with improved retry logic for rate limiting"""
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="deepseek-reasoner",  # DeepSeek's main model
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
    
    # Primary regex for "**Output[:：]**"
    primary_output_marker_pattern = re.compile(r"\*\*Output[:：]\*\*\s*(.*)", re.DOTALL)
    # Secondary regex for "Step 4: Output results"
    secondary_output_marker_pattern = re.compile(r"Step 4: Output results\s*(.*)", re.DOTALL)
    # Regex for splitting by "Output:" or "Output："
    simple_output_marker_regex = r"Output[:：]"
    
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
            # If secondary marker is not found, try the simple plain "Output[:：]" marker
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
        if not label or label == "None" or label == "(None)":
            return "N/A"
        else:
            return label
    else:
        return "PARSE_ERROR_NO_MARKER"

async def process_single_text(text: str, mode: str, index: int, total: int) -> Dict[str, Any]:
    """Process a single text asynchronously"""
    try:
        messages = generate_prompt(text, mode)
        output = await call_deepseek(messages)
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
    if not api_key or api_key == "your-deepseek-api-key-here":
        print("Error: DeepSeek API key not set. Please set the DEEPSEEK_API_KEY environment variable or set it in the script.")
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
        output_filename = f"llm_outputs/deepseek_{mode}_en.csv"
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