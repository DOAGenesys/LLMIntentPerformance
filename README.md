# NLU Intent Classification with Batch Processing and Prompt Caching

## Description

This project implements a Natural Language Understanding (NLU) intent classification system using both OpenAI and Anthropic APIs. The script processes a dataset of utterances, classifies their intents, and evaluates the performance of both AI models.

### Key Features

1. **Batch Processing**: Instead of making individual API calls for each utterance, the script uses batch processing to send multiple requests at once, improving efficiency and reducing overall processing time.
2. **Prompt Caching**: The script leverages prompt caching mechanisms provided by both OpenAI and Anthropic to reduce costs and improve response times for similar prompts.
3. **Parallel Processing**: The script processes batches for OpenAI and Anthropic in parallel, allowing for simultaneous evaluation of both services.
4. **Performance Evaluation**: After processing, the script evaluates the accuracy of intent classifications for both APIs.
5. **Failed Classifications Output**: The script generates an additional file containing only the failed classifications.

## How It Works

1. **Environment Setup**:
   - Loads environment variables from a .env file, including API keys, model names, and system prompts.
   - Initializes API clients for both OpenAI and Anthropic.

2. **Data Loading**:
   - Loads the dataset of utterances to be classified from a JSON file.

3. **OpenAI Batch Processing**:
   - Prepares a batch file in JSONL format for OpenAI, including system prompts and utterances.
   - Uploads the batch file to OpenAI's servers.
   - Creates and submits a batch job to OpenAI's API.
   - Polls for job completion, waiting up to 24 hours.
   - Retrieves and processes the results once the batch job is complete.

4. **Anthropic Batch Processing**:
   - Prepares batch data for Anthropic, including system prompts with cache control for prompt caching.
   - Submits the batch to Anthropic's API.
   - Polls for job completion, waiting up to 24 hours.
   - Retrieves and processes the results once the batch job is complete.

5. **Result Processing**:
   - Parses the responses from both APIs into a structured format.
   - Evaluates the accuracy of intent classifications for each API.

6. **Output Generation**:
   - Combines the results and evaluations into a single JSON output.
   - Writes the output to a file for further analysis.
   - Generates an additional file containing only the failed classifications.

## Input Dataset Format

The input dataset (`intent_classification_dataset.json`) should be a JSON file with the following structure:

```json
[
  {
    "Company Name": "string",
    "Examples": [
      {
        "sample_text": "string",
        "class": "string"
      }
    ]
  }
]
```

- `Company Name`: The name of the company or domain for which intents are being classified.
- `Examples`: An array of utterances and their true intents.
  - `sample_text`: The utterance to be classified.
  - `class`: The true intent of the utterance.

## Output Results Format

The output results (`intent_classification_results.json`) will have the following structure:

```json
{
  "OpenAI": {
    "Company Name 1": {
      "results": [
        {
          "custom_id": "string",
          "utterance": "string",
          "classification": {
            "predicted_intent": {
              "intent": "string",
              "confidence": number
            },
            "second_predicted_intent": {
              "intent": "string",
              "confidence": number
            },
            "true_intent": "string",
            "match": boolean
          }
        }
      ],
      "evaluation": {
        "total_attempts": number,
        "correct_matches": number,
        "accuracy": number
      }
    },
    "openai_overall_performance": {
      "total_correct": number,
      "total_attempts": number,
      "overall_accuracy": number
    }
  },
  "Anthropic": {
    // Similar structure to OpenAI
  },
  "final_overall_performance": {
    "total_correct": number,
    "total_attempts": number,
    "overall_accuracy": number
  }
}
```

- `custom_id`: A unique identifier for each utterance.
- `utterance`: The original utterance text.
- `classification`: Contains the predicted intents, true intent, and whether there was a match.
- `evaluation`: Provides performance metrics for each company.
- `*_overall_performance`: Aggregated performance metrics for each API and overall.

## Failed Classifications Output

The failed classifications (`intent_classification_fails.json`) will have a similar structure to the main results file, but will only include the classifications where `match` is `false`.

## Environment Variables

- `FAKE_BATCHES`: Set to "true" to simulate batch processing without making actual API calls. Default is "false".
- `OPENAI_API_KEY`: Your OpenAI API key (redacted for security).
- `OPENAI_MODEL`: The OpenAI model to use for classification. Set to "gpt-4o-mini".
- `OPENAI_BASE_PROMPT`: "You are an AI assistant specialized in Natural Language Understanding (NLU) intent classification. Your task is to analyze user utterances and determine the two most appropriate intents for each one. You will be provided with a series of utterances, and you must classify each one according to the intents available for the specific company or domain. Please follow these guidelines: 1. Analyze each utterance carefully to understand the user's intention. 2. Select the two most appropriate intents based on the context and available intent options. 3. If you're unsure about the intents, choose the ones that best match the utterance ('unknown' values or similar are not allowed, only the ones on the list). 4. Provide your classification in a structured JSON format. Your output must strictly adhere to the following JSON schema: { \"type\": \"object\", \"properties\": { \"classifications\": { \"type\": \"array\", \"items\": { \"type\": \"object\", \"properties\": { \"custom_id\": { \"type\": \"string\", \"description\": \"A unique identifier for each utterance\" }, \"classification\": { \"type\": \"object\", \"properties\": { \"intent1\": { \"type\": \"string\", \"description\": \"The name of the most likely intent\" }, \"confidence1\": { \"type\": \"number\", \"description\": \"Confidence score for intent1 between 0 and 1\" }, \"intent2\": { \"type\": \"string\", \"description\": \"The name of the second most likely intent\" }, \"confidence2\": { \"type\": \"number\", \"description\": \"Confidence score for intent2 between 0 and 1\" } }, \"required\": [\"intent1\", \"confidence1\", \"intent2\", \"confidence2\"], \"additionalProperties\": false } }, \"required\": [\"custom_id\", \"classification\"], \"additionalProperties\": false } } }, \"required\": [\"classifications\"], \"additionalProperties\": false } Remember: - Maintain consistency in your classifications. - Adhere strictly to the JSON schema provided. - Do not include any explanations or additional text outside of the JSON structure"
- `ANTHROPIC_API_KEY`: Your Anthropic API key (redacted for security).
- `ANTHROPIC_MODEL`: The Anthropic model to use for classification. Set to "claude-3-haiku-20240307".
- `ANTHROPIC_BASE_PROMPT`: "You are an AI assistant specialized in Natural Language Understanding (NLU) intent classification. Your task is to analyze user utterances and determine the two most appropriate intents for each one. You will be provided with a series of utterances, and you must classify each one according to the intents available for the specific company or domain. Please follow these guidelines: 1. Analyze each utterance carefully to understand the user's intention. 2. Select the two most appropriate intents based on the context and available intent options. 3. If you're unsure about the intents, choose the ones that best match the utterance ('unknown' values or similar are not allowed, only the ones on the list). 4. Provide your classification in a structured JSON format. Your output must strictly adhere to the following JSON structure: { \"classifications\": [ { \"custom_id\": \"string\", \"classification\": { \"intent1\": \"string\", \"confidence1\": number, \"intent2\": \"string\", \"confidence2\": number } } ] } Where: - \"custom_id\" is a unique identifier for each utterance (use the index of the utterance in the batch). - \"intent1\" and \"intent2\" are the names of the two most likely classified intents. - \"confidence1\" and \"confidence2\" are floats between 0 and 1 indicating your confidence in each classification. Remember: - Maintain consistency in your classifications. - Adhere strictly to the JSON format provided. - Do not include any explanations or additional text outside of the JSON structure"

## Prompt Caching Details

- For OpenAI: The script structures prompts with the system message first, followed by the user message. OpenAI's API automatically handles caching for prompts longer than 1024 tokens.
- For Anthropic: The script explicitly marks the system prompt for caching using the cache_control parameter. This allows Anthropic to reuse the cached prompt for multiple requests, reducing processing time and costs.

## Usage

1. Ensure your `.env` file is properly configured with all necessary API keys, model names, and system prompts.
2. Prepare your `intent_classification_dataset.json` file with the required structure.
3. Run the script:
   ```
   python nlu_intent_classification.py
   ```
4. The script will generate two output files:
   - `intent_classification_results.json`: Contains all classification results and performance metrics.
   - `intent_classification_fails.json`: Contains only the failed classifications.

## Notes

- The script is designed to handle large datasets efficiently through batch processing.
- It provides a comprehensive comparison between OpenAI and Anthropic models for NLU intent classification tasks.
- The failed classifications output can be particularly useful for error analysis and improving the classification system.
