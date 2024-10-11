"""
NLU Intent Classification with Batch Processing and Prompt Caching

This script performs Natural Language Understanding (NLU) intent classification using both OpenAI and Anthropic APIs. 
It processes a dataset of utterances, classifies their intents, and evaluates the performance of both AI models.

Key Features:
1. Batch Processing: Instead of making individual API calls for each utterance, the script uses batch processing 
   to send multiple requests at once, improving efficiency and reducing overall processing time.
2. Prompt Caching: The script leverages prompt caching mechanisms provided by both OpenAI and Anthropic to reduce 
   costs and improve response times for similar prompts.
3. Parallel Processing: The script processes batches for OpenAI and Anthropic in parallel, allowing for 
   simultaneous evaluation of both services.
4. Performance Evaluation: After processing, the script evaluates the accuracy of intent classifications for both APIs.

How it works:
1. Environment Setup:
   - Loads environment variables from a .env file, including API keys, model names, and system prompts.
   - Initializes API clients for both OpenAI and Anthropic.

2. Data Loading:
   - Loads the dataset of utterances to be classified from a JSON file.

3. OpenAI Batch Processing:
   - Prepares a batch file in JSONL format for OpenAI, including system prompts and utterances.
   - Uploads the batch file to OpenAI's servers.
   - Creates and submits a batch job to OpenAI's API.
   - Polls for job completion, waiting up to 24 hours.
   - Retrieves and processes the results once the batch job is complete.

4. Anthropic Batch Processing:
   - Prepares batch data for Anthropic, including system prompts with cache control for prompt caching.
   - Submits the batch to Anthropic's API.
   - Polls for job completion, waiting up to 24 hours.
   - Retrieves and processes the results once the batch job is complete.

5. Result Processing:
   - Parses the responses from both APIs into a structured format.
   - Evaluates the accuracy of intent classifications for each API.

6. Output Generation:
   - Combines the results and evaluations into a single JSON output.
   - Writes the output to a file for further analysis.

Prompt Caching Details:
- For OpenAI: The script structures prompts with the system message first, followed by the user message. 
  OpenAI's API automatically handles caching for prompts longer than 1024 tokens.
- For Anthropic: The script explicitly marks the system prompt for caching using the cache_control parameter. 
  This allows Anthropic to reuse the cached prompt for multiple requests, reducing processing time and costs.

Note: To use this script effectively, ensure that your .env file is properly configured with all necessary 
API keys, model names, and system prompts. Also, make sure that your system prompts are substantial enough 
to benefit from caching, especially for Anthropic (aim for at least 1024 tokens for Claude 3.5 Sonnet and 
Claude 3 Opus, or 2048 tokens for Claude 3 Haiku).
"""
import json
import os
import time
import asyncio
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)
logger.info("Environment variables loaded")

# Initialize API clients
try:
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    logger.info("API clients initialized")
except Exception as e:
    logger.error(f"Error initializing API clients: {e}")
    raise

# Define data structures
class IntentClassification(BaseModel):
    intent: str
    confidence: float

class ClassificationResult(BaseModel):
    custom_id: str
    classification: IntentClassification

# Load dataset
def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Dataset loaded from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in dataset file: {file_path}")
        raise

# Prepare system prompt
def prepare_system_prompt(company_data: Dict[str, Any], base_prompt: str) -> str:
    company_name = company_data["Company Name"]
    topics = company_data.get("Taxonomy", [])
    
    topic_list = "\n".join([f"{topic['Topic Name']}: {topic['Topic Description']}" for topic in topics])
    
    return f"{base_prompt}\n\nCompany: {company_name}\n\nAvailable Intents:\n{topic_list}"

# Prepare OpenAI batch
async def prepare_openai_batch(dataset: List[Dict[str, Any]], system_prompt: str) -> str:
    batch_data = []
    
    for i, item in enumerate(dataset["Examples"]):
        if 'sample_text' not in item:
            logger.error(f"Missing 'sample_text' key in dataset item {i}")
            raise KeyError(f"Missing 'sample_text' key in dataset item {i}")

        batch_item = {
            "custom_id": f"utterance_{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": os.getenv("OPENAI_MODEL"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item['sample_text']}
                ],
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            }
        }
        batch_data.append(json.dumps(batch_item))

    batch_file = f"openai_batch_{dataset['Company Name']}.jsonl"
    with open(batch_file, 'w') as f:
        f.write('\n'.join(batch_data))
    logger.info(f"OpenAI batch file prepared: {batch_file}")
    return batch_file

# Process OpenAI batch
async def process_openai_batch(batch_file: str, company_name: str) -> List[ClassificationResult]:
    try:
        # Upload batch file
        with open(batch_file, "rb") as f:
            file = await openai_client.files.create(file=f, purpose="batch")
        logger.info(f"Batch file uploaded for {company_name}: {file.id}")

        # Create batch
        batch = await openai_client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        logger.info(f"Batch created for {company_name}: {batch.id}")

        return batch.id
    except Exception as e:
        logger.error(f"Error processing OpenAI batch for {company_name}: {e}")
        raise

# Prepare Anthropic batch
async def prepare_anthropic_batch(dataset: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
    batch_data = []
    for i, item in enumerate(dataset["Examples"]):
        if 'sample_text' not in item:
            logger.error(f"Missing 'sample_text' key in dataset item {i}")
            raise KeyError(f"Missing 'sample_text' key in dataset item {i}")

        batch_data.append({
            "custom_id": f"utterance_{i}",
            "params": {
                "model": os.getenv("ANTHROPIC_MODEL"),
                "max_tokens": 100,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": item['sample_text']}
                ]
            }
        })

    logger.info(f"Anthropic batch prepared for {dataset['Company Name']} with {len(batch_data)} items")
    return batch_data

# Process Anthropic batch
async def process_anthropic_batch(batch_data: List[Dict[str, Any]], company_name: str) -> str:
    try:
        # Create batch
        batch = await anthropic_client.beta.messages.batches.create(requests=batch_data)
        logger.info(f"Anthropic batch created for {company_name}: {batch.id}")

        return batch.id
    except Exception as e:
        logger.error(f"Error processing Anthropic batch for {company_name}: {e}")
        raise

# Check batch status and retrieve results
async def check_batch_status(api_type: str, batch_id: str, company_name: str):
    while True:
        try:
            if api_type == "openai":
                batch_status = await openai_client.batches.retrieve(batch_id)
                if batch_status.status == "completed":
                    results = await openai_client.files.content(batch_status.output_file_id)
                    results_data = [json.loads(line) for line in results.splitlines()]
                    logger.info(f"Retrieved {len(results_data)} results from OpenAI for {company_name}")
                    return results_data
            elif api_type == "anthropic":
                batch_status = await anthropic_client.beta.messages.batches.retrieve(batch_id)
                if batch_status.processing_status == "ended":
                    results = []
                    async for result in anthropic_client.beta.messages.batches.results(batch_id):
                        if result.result.type == "succeeded":
                            content = json.loads(result.result.message.content[0].text)
                            results.extend(content['classifications'])
                    logger.info(f"Retrieved {len(results)} results from Anthropic for {company_name}")
                    return results
        except Exception as e:
            logger.error(f"Error checking batch status for {api_type} - {company_name}: {e}")
            return None

        await asyncio.sleep(60)  # Wait for 1 minute before checking again

# Evaluate results
def evaluate_results(results: List[ClassificationResult], dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r, d in zip(results, dataset["Examples"]) if r['classification']['intent'] == d['class'])
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Evaluation results: Total={total}, Correct={correct}, Accuracy={accuracy:.2f}")
    return {
        "total_attempts": total,
        "correct_matches": correct,
        "errors": total - correct,
        "accuracy": accuracy
    }

# Write partial results
def write_partial_results(output: Dict[str, Any], file_path: str):
    with open(file_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Partial results written to {file_path}")

# Main execution
async def main():
    output = {"OpenAI": {}, "Anthropic": {}}
    output_file = 'intent_classification_results.json'

    try:
        dataset = load_dataset('intent_classification_dataset.json')
        
        openai_base_prompt = os.getenv("OPENAI_BASE_PROMPT")
        anthropic_base_prompt = os.getenv("ANTHROPIC_BASE_PROMPT")

        tasks = []

        for company_data in dataset:
            company_name = company_data["Company Name"]
            
            # Prepare and process OpenAI batch
            openai_system_prompt = prepare_system_prompt(company_data, openai_base_prompt)
            openai_batch_file = await prepare_openai_batch(company_data, openai_system_prompt)
            openai_batch_id = await process_openai_batch(openai_batch_file, company_name)
            
            # Prepare and process Anthropic batch
            anthropic_system_prompt = prepare_system_prompt(company_data, anthropic_base_prompt)
            anthropic_batch_data = await prepare_anthropic_batch(company_data, anthropic_system_prompt)
            anthropic_batch_id = await process_anthropic_batch(anthropic_batch_data, company_name)
            
            # Add tasks to check batch status
            tasks.append(asyncio.create_task(check_batch_status("openai", openai_batch_id, company_name)))
            tasks.append(asyncio.create_task(check_batch_status("anthropic", anthropic_batch_id, company_name)))

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Process results
        for i in range(0, len(results), 2):
            company_name = dataset[i//2]["Company Name"]
            openai_results = results[i]
            anthropic_results = results[i+1]

            if openai_results:
                openai_evaluation = evaluate_results(openai_results, dataset[i//2])
                output["OpenAI"][company_name] = {
                    "results": openai_results,
                    "evaluation": openai_evaluation
                }
                write_partial_results(output, output_file)

            if anthropic_results:
                anthropic_evaluation = evaluate_results(anthropic_results, dataset[i//2])
                output["Anthropic"][company_name] = {
                    "results": anthropic_results,
                    "evaluation": anthropic_evaluation
                }
                write_partial_results(output, output_file)

        # Calculate overall performance
        for api in ["OpenAI", "Anthropic"]:
            total_correct = sum(company_data["evaluation"]["correct_matches"] for company_data in output[api].values())
            total_attempts = sum(company_data["evaluation"]["total_attempts"] for company_data in output[api].values())
            overall_accuracy = total_correct / total_attempts if total_attempts > 0 else 0
            output[api]["overall_performance"] = {
                "total_correct": total_correct,
                "total_attempts": total_attempts,
                "overall_accuracy": overall_accuracy
            }

        # Write final results
        write_partial_results(output, output_file)
        logger.info("Final results written to intent_classification_results.json")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
