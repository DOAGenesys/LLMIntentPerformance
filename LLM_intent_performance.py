import json
import os
import time
import asyncio
import logging
import glob
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
class IntentPrediction(BaseModel):
    intent: str
    confidence: float

class IntentClassification(BaseModel):
    predicted_intent: IntentPrediction
    second_predicted_intent: IntentPrediction
    true_intent: str
    match: bool

class ClassificationResult(BaseModel):
    custom_id: str
    utterance: str
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

# Extract intents for a company
def extract_intents(company_data: Dict[str, Any]) -> List[str]:
    intents = list(set([example["class"] for example in company_data.get("Examples", [])]))
    logger.info(f"Extracted intents for {company_data['Company Name']}: {intents}")
    return intents

# Prepare system prompt with available intents
def prepare_system_prompt(company_data: Dict[str, Any], base_prompt: str) -> str:
    company_name = company_data["Company Name"]
    intents = extract_intents(company_data)
    
    available_intents = "\n".join([f"- {intent}" for intent in intents])
    
    system_prompt = f"{base_prompt}\n\nCompany: {company_name}\n\nAvailable Intents:\n{available_intents}"
    logger.info(f"Prepared system prompt for {company_name}. Available Intents:\n{available_intents}")
    return system_prompt

# Prepare OpenAI batch
def prepare_openai_batch(dataset: Dict[str, Any], system_prompt: str) -> str:
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
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            }
        }
        batch_data.append(json.dumps(batch_item))

    batch_file = f"openai_batch_{dataset['Company Name']}.jsonl"
    with open(batch_file, 'w') as f:
        f.write('\n'.join(batch_data))
    logger.info(f"OpenAI batch file prepared: {batch_file}")
    logger.info(f"Sample system prompt in OpenAI batch: {system_prompt[:500]}...")  # Log first 500 characters
    return batch_file

# Process OpenAI batch
async def process_openai_batch(batch_file: str, company_name: str, fake_batches: bool) -> str:
    try:
        if fake_batches:
            logger.info(f"Fake batch processing for OpenAI - {company_name}")
            return f"fake_batch_id_openai_{company_name}"

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
def prepare_anthropic_batch(dataset: Dict[str, Any], system_prompt: str) -> List[Dict[str, Any]]:
    batch_data = []
    for i, item in enumerate(dataset["Examples"]):
        if 'sample_text' not in item:
            logger.error(f"Missing 'sample_text' key in dataset item {i}")
            raise KeyError(f"Missing 'sample_text' key in dataset item {i}")

        batch_data.append({
            "custom_id": f"utterance_{i}",
            "params": {
                "model": os.getenv("ANTHROPIC_MODEL"),
                "max_tokens": 150,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": item['sample_text']}
                ]
            }
        })

    batch_file = f"anthropic_batch_{dataset['Company Name']}.json"
    with open(batch_file, 'w') as f:
        json.dump(batch_data, f, indent=2)
    logger.info(f"Anthropic batch file prepared: {batch_file}")
    logger.info(f"Sample system prompt in Anthropic batch: {system_prompt[:500]}...")  # Log first 500 characters
    return batch_data

# Process Anthropic batch
async def process_anthropic_batch(batch_data: List[Dict[str, Any]], company_name: str, fake_batches: bool) -> str:
    try:
        if fake_batches:
            logger.info(f"Fake batch processing for Anthropic - {company_name}")
            return f"fake_batch_id_anthropic_{company_name}"

        # Create batch
        batch = await anthropic_client.beta.messages.batches.create(requests=batch_data)
        logger.info(f"Anthropic batch created for {company_name}: {batch.id}")

        return batch.id
    except Exception as e:
        logger.error(f"Error processing Anthropic batch for {company_name}: {e}")
        raise

# Save batch results to a local file
async def save_batch_results(api_type: str, batch_id: str, company_name: str, content: str):
    file_name = f"{api_type}_results_{company_name}_{batch_id}.json"
    with open(file_name, 'w') as f:
        f.write(content)
    logger.info(f"Batch results saved to {file_name}")
    return file_name

# Check batch status and retrieve results
async def check_batch_status(api_type: str, batch_id: str, company_name: str, fake_batches: bool, dataset: Dict[str, Any]):
    start_time = time.time()
    while True:
        try:
            if fake_batches:
                # Simulate batch processing
                await asyncio.sleep(5)
                logger.info(f"Fake batch completed for {api_type} - {company_name}")
                return [{"custom_id": "utterance_0", "classification": {"intent": "fake_intent", "confidence": 0.9}}]

            if api_type == "openai":
                batch_status = await openai_client.batches.retrieve(batch_id)
                logger.info(f"OpenAI batch status for {company_name}: {batch_status.status}")
                logger.info(f"Progress: Total: {batch_status.request_counts.total}, Completed: {batch_status.request_counts.completed}, Failed: {batch_status.request_counts.failed}")
                
                if batch_status.status == "completed":
                    content = await openai_client.files.content(batch_status.output_file_id)
                    file_name = await save_batch_results("openai", batch_id, company_name, content.text)
                    logger.info(f"OpenAI results saved to {file_name}")
                    return file_name
            elif api_type == "anthropic":
                batch_status = await anthropic_client.beta.messages.batches.retrieve(batch_id)
                logger.info(f"Anthropic batch status for {company_name}: {batch_status.processing_status}")
                logger.info(f"Progress: Processing: {batch_status.request_counts.processing}, Succeeded: {batch_status.request_counts.succeeded}, Errored: {batch_status.request_counts.errored}")
                
                if batch_status.processing_status == "ended":
                    results = []
                    batch_results = await anthropic_client.beta.messages.batches.results(batch_id)
                    async for result in batch_results:
                        results.append(result)
                    
                    file_name = await save_batch_results("anthropic", batch_id, company_name, json.dumps(results, default=lambda x: x.__dict__))
                    logger.info(f"Anthropic results saved to {file_name}")
                    return file_name
        except Exception as e:
            logger.error(f"Error checking batch status for {api_type} - {company_name}: {e}")
            return None

        if time.time() - start_time > 86400:  # 24 hours
            logger.error(f"Batch processing timeout for {api_type} - {company_name}")
            return None

        await asyncio.sleep(60)  # Wait for 1 minute before checking again

def load_jsonl_file(file_path: str) -> List[Any]:
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_json_file(file_path: str) -> Any:
    with open(file_path, 'r') as f:
        return json.load(f)

# Process OpenAI results
def process_openai_results(file_path: str, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = load_jsonl_file(file_path)
    processed_results = []
    
    for result, example in zip(results, dataset["Examples"]):
        try:
            content = json.loads(result["response"]["body"]["choices"][0]["message"]["content"])
            classification = content['classifications'][0]['classification']
            true_intent = example['class']
            utterance = example['sample_text']
            
            processed_results.append({
                "custom_id": result["custom_id"],
                "utterance": utterance,
                "classification": {
                    "predicted_intent": {
                        "intent": classification['intent1'],
                        "confidence": classification['confidence1']
                    },
                    "second_predicted_intent": {
                        "intent": classification['intent2'],
                        "confidence": classification['confidence2']
                    },
                    "true_intent": true_intent,
                    "match": classification['intent1'].lower() == true_intent.lower()
                }
            })
        except (KeyError, json.JSONDecodeError):
            processed_results.append({
                "custom_id": result["custom_id"],
                "utterance": example['sample_text'],
                "classification": {
					"predicted_intent": {"intent": "unknown", "confidence": 0.0},
						"second_predicted_intent": {"intent": "unknown", "confidence": 0.0},
						"true_intent": example['class'],
						"match": False
                }
            })
    
    return processed_results

# Process Anthropic results
def process_anthropic_results(file_path: str, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = load_json_file(file_path)
    processed_results = []
    
    for result, example in zip(results, dataset["Examples"]):
        try:
            content = json.loads(result['result']['message']['content'][0]['text'])
            classification = content['classifications'][0]['classification']
            true_intent = example['class']
            utterance = example['sample_text']
            
            processed_results.append({
                "custom_id": result['custom_id'],
                "utterance": utterance,
                "classification": {
                    "predicted_intent": {
                        "intent": classification['intent1'],
                        "confidence": classification['confidence1']
                    },
                    "second_predicted_intent": {
                        "intent": classification['intent2'],
                        "confidence": classification['confidence2']
                    },
                    "true_intent": true_intent,
                    "match": classification['intent1'].lower() == true_intent.lower()
                }
            })
        except (KeyError, json.JSONDecodeError):
            processed_results.append({
                "custom_id": result['custom_id'],
                "utterance": example['sample_text'],
                "classification": {
                    "predicted_intent": {"intent": "unknown", "confidence": 0.0},
                    "second_predicted_intent": {"intent": "unknown", "confidence": 0.0},
                    "true_intent": example['class'],
                    "match": False
                }
            })
    
    return processed_results

# Evaluate results
def evaluate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r['classification']['match'])
    accuracy = correct / total if total > 0 else 0
    return {
        "total_attempts": total,
        "correct_matches": correct,
        "accuracy": accuracy
    }

def process_company_results(company_name: str, openai_file: str, anthropic_file: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
    openai_results = process_openai_results(openai_file, dataset)
    anthropic_results = process_anthropic_results(anthropic_file, dataset)
    
    return {
        "OpenAI": {
            "results": openai_results,
            "evaluation": evaluate_results(openai_results)
        },
        "Anthropic": {
            "results": anthropic_results,
            "evaluation": evaluate_results(anthropic_results)
        }
    }

def calculate_overall_performance(company_results: Dict[str, Any]) -> Dict[str, Any]:
    overall_performance = {"OpenAI": {}, "Anthropic": {}}
    
    for api in ["OpenAI", "Anthropic"]:
        total_correct = sum(company_data[api]["evaluation"]["correct_matches"] for company_data in company_results.values())
        total_attempts = sum(company_data[api]["evaluation"]["total_attempts"] for company_data in company_results.values())
        overall_accuracy = total_correct / total_attempts if total_attempts > 0 else 0
        
        overall_performance[api] = {
            "total_correct": total_correct,
            "total_attempts": total_attempts,
            "overall_accuracy": overall_accuracy
        }
    
    return overall_performance

def calculate_final_performance(openai_performance: Dict[str, Any], anthropic_performance: Dict[str, Any]) -> Dict[str, Any]:
    total_correct = openai_performance["total_correct"] + anthropic_performance["total_correct"]
    total_attempts = openai_performance["total_attempts"] + anthropic_performance["total_attempts"]
    overall_accuracy = total_correct / total_attempts if total_attempts > 0 else 0
    
    return {
        "total_correct": total_correct,
        "total_attempts": total_attempts,
        "overall_accuracy": overall_accuracy
    }

def save_json_file(data: Any, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {file_path}")

# New function to extract failed classifications
def extract_failed_classifications(results: Dict[str, Any]) -> Dict[str, Any]:
    failed_classifications = {"OpenAI": {}, "Anthropic": {}}
    
    for api in ["OpenAI", "Anthropic"]:
        for company, company_data in results[api].items():
            if company == f"{api.lower()}_overall_performance":
                continue
            failed_classifications[api][company] = [
                result for result in company_data["results"]
                if not result["classification"]["match"]
            ]
    
    return failed_classifications

# Main execution
async def main():
    output_file = 'intent_classification_results.json'
    fails_output_file = 'intent_classification_fails.json'
    fake_batches = os.getenv("FAKE_BATCHES", "false").lower() == "true"

    try:
        dataset = load_dataset('intent_classification_dataset.json')
        
        openai_base_prompt = os.getenv("OPENAI_BASE_PROMPT")
        anthropic_base_prompt = os.getenv("ANTHROPIC_BASE_PROMPT")

        tasks = []
        company_results = {}

        for company_data in dataset:
            company_name = company_data["Company Name"]
            logger.info(f"Processing company: {company_name}")
            
            # Prepare and process OpenAI batch
            openai_system_prompt = prepare_system_prompt(company_data, openai_base_prompt)
            openai_batch_file = prepare_openai_batch(company_data, openai_system_prompt)
            openai_batch_id = await process_openai_batch(openai_batch_file, company_name, fake_batches)
            
            # Prepare and process Anthropic batch
            anthropic_system_prompt = prepare_system_prompt(company_data, anthropic_base_prompt)
            anthropic_batch_data = prepare_anthropic_batch(company_data, anthropic_system_prompt)
            anthropic_batch_id = await process_anthropic_batch(anthropic_batch_data, company_name, fake_batches)
            
            # Add tasks to check batch status
            tasks.append(asyncio.create_task(check_batch_status("openai", openai_batch_id, company_name, fake_batches, company_data)))
            tasks.append(asyncio.create_task(check_batch_status("anthropic", anthropic_batch_id, company_name, fake_batches, company_data)))

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i in range(0, len(results), 2):
            company_name = dataset[i//2]["Company Name"]
            openai_file = results[i]
            anthropic_file = results[i+1]

            if isinstance(openai_file, Exception):
                logger.error(f"Error processing OpenAI results for {company_name}: {openai_file}")
            elif isinstance(anthropic_file, Exception):
                logger.error(f"Error processing Anthropic results for {company_name}: {anthropic_file}")
            elif openai_file and anthropic_file:
                company_results[company_name] = process_company_results(company_name, openai_file, anthropic_file, dataset[i//2])

        overall_performance = calculate_overall_performance(company_results)
        final_performance = calculate_final_performance(overall_performance["OpenAI"], overall_performance["Anthropic"])

        output = {
            "OpenAI": {**company_results, "openai_overall_performance": overall_performance["OpenAI"]},
            "Anthropic": {**company_results, "anthropic_overall_performance": overall_performance["Anthropic"]},
            "final_overall_performance": final_performance
        }

        save_json_file(output, output_file)
        logger.info("Final results written to intent_classification_results.json")

        # Extract and save failed classifications
        failed_classifications = extract_failed_classifications(output)
        save_json_file(failed_classifications, fails_output_file)
        logger.info("Failed classifications written to intent_classification_fails.json")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
