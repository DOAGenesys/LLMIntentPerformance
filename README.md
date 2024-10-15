# LLM performance for NLU Intent Classification

## Description

This project implements a Natural Language Understanding (NLU) intent classification system using both OpenAI and Anthropic APIs. The script processes two datasets of utterances (a general dataset and a banking-specific dataset), classifies their intents, and evaluates the performance of both AI models.

### Key Features

1. **Batch Processing**: Instead of making individual API calls for each utterance, the script uses batch processing to send multiple requests at once, improving efficiency and reducing overall processing time.
2. **Prompt Caching**: The script leverages prompt caching mechanisms provided by both OpenAI and Anthropic to reduce costs and improve response times for similar prompts.
3. **Parallel Processing**: The script processes batches for OpenAI and Anthropic in parallel, allowing for simultaneous evaluation of both services.
4. **Performance Evaluation**: After processing, the script evaluates the accuracy of intent classifications for both APIs across multiple datasets.
5. **Structured Output**: The script enforces JSON/structured outputs for both OpenAI and Anthropic, which is critical for accurate intent performance evaluation.

## How It Works

1. **Environment Setup**:
   - Loads environment variables from a .env file, including API keys, model names, and system prompts.
   - Initializes API clients for both OpenAI and Anthropic.

2. **Data Loading**:
   - Loads two datasets of utterances to be classified from JSON files: a general dataset and a banking-specific dataset.

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
   - Uses robust JSON parsing to handle different response structures from OpenAI and Anthropic.
   - Implements error handling to manage unexpected JSON formats or missing data.
   - Evaluates the accuracy of intent classifications for each API.

6. **Output Generation**:
   - Combines the results and evaluations into two separate JSON outputs: one for the general dataset and one for the banking dataset.
   - Writes the outputs to files for further analysis.

## Input Dataset Format

The input datasets (`intent_classification_dataset.json` and `banking_intent_classification_dataset.json`) should be JSON files with the following structure:

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

The output results (`intent_classification_results.json` and `banking_intent_classification_results.json`) will have the following structure:

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

## Environment Variables

- `FAKE_BATCHES`: Set to "true" to simulate batch processing without making actual API calls. Default is "false".
- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_MODEL`: The OpenAI model to use for classification. Results in repo produced with model "gpt-4o-mini".
- `OPENAI_BASE_PROMPT`: "You are an AI assistant specialized in Natural Language Understanding (NLU) intent classification. Your task is to analyze user utterances and determine the two most appropriate intents for each one. You will be provided with a series of utterances, and you must classify each one according to the intents available for the specific company or domain. Please follow these guidelines: 1. Analyze each utterance carefully to understand the user's intention. 2. Select the two most appropriate intents based on the context and available intent options. 3. If you're unsure about the intents, choose the ones that best match the utterance. 4. Provide your classification in a structured JSON format. Your output must strictly adhere to the following JSON schema: { "type": "object", "properties": { "classifications": { "type": "array", "items": { "type": "object", "properties": { "custom_id": { "type": "string", "description": "A unique identifier for each utterance" }, "classification": { "type": "object", "properties": { "intent1": { "type": "string", "description": "The name of the most likely intent" }, "confidence1": { "type": "number", "description": "Confidence score for intent1 between 0 and 1" }, "intent2": { "type": "string", "description": "The name of the second most likely intent" }, "confidence2": { "type": "number", "description": "Confidence score for intent2 between 0 and 1" } }, "required": ["intent1", "confidence1", "intent2", "confidence2"], "additionalProperties": false } }, "required": ["custom_id", "classification"], "additionalProperties": false } } }, "required": ["classifications"], "additionalProperties": false } Remember: - Maintain consistency in your classifications. - Adhere strictly to the JSON schema provided. - Do not include any explanations or additional text outside of the JSON structure"
- `ANTHROPIC_API_KEY`: Your Anthropic API key.
- `ANTHROPIC_MODEL`: The Anthropic model to use for classification. Results in repo produced with model "claude-3-haiku-20240307".
- `ANTHROPIC_BASE_PROMPT`: "You are an AI assistant specialized in Natural Language Understanding (NLU) intent classification. Your task is to analyze user utterances and determine the two most appropriate intents for each one. You will be provided with a series of utterances, and you must classify each one according to the intents available for the specific company or domain. Please follow these guidelines: 1. Analyze each utterance carefully to understand the user's intention. 2. Select the two most appropriate intents based on the context and available intent options. 3. If you're unsure about the intents, choose the ones that best match the utterance. 4. Provide your classification in a structured JSON format. Your output must strictly adhere to the following JSON structure: { "classifications": [ { "custom_id": "string", "classification": { "intent1": "string", "confidence1": number, "intent2": "string", "confidence2": number } } ] } Where: - "custom_id" is a unique identifier for each utterance (use the index of the utterance in the batch). - "intent1" and "intent2" are the names of the two most likely classified intents. - "confidence1" and "confidence2" are floats between 0 and 1 indicating your confidence in each classification. Remember: - Maintain consistency in your classifications. - Adhere strictly to the JSON format provided. - Do not include any explanations or additional text outside of the JSON structure"

## Prompt Caching Details

- For OpenAI: The script structures prompts with the system message first, followed by the user message. OpenAI's API automatically handles caching for prompts longer than 1024 tokens.
- For Anthropic: The script explicitly marks the system prompt for caching using the cache_control parameter. This allows Anthropic to reuse the cached prompt for multiple requests, reducing processing time and costs.

## Usage

1. Ensure your `.env` file is properly configured with all necessary API keys, model names, and system prompts.
2. Make sure `intent_classification_dataset.json` and `banking_intent_classification_dataset.json` files are present.
3. Run the script:
   ```
   python LLM_intent_performance.py
   ```
4. The script will generate two output files:
   - `intent_classification_results.json`: Contains classification results and performance metrics for the general dataset.
   - `banking_intent_classification_results.json`: Contains classification results and performance metrics for the banking-specific dataset.

## Notes

- The script is designed to handle large datasets efficiently through batch processing.
- It provides a comprehensive comparison between OpenAI and Anthropic models for NLU intent classification tasks across different domains.
- Enforcing JSON/structured outputs for both OpenAI and Anthropic is critical for accurate intent performance evaluation. This ensures that the responses can be consistently parsed and analyzed, leading to more reliable performance metrics.

## Results Discussion

The results obtained from running this script provide valuable insights into the performance of OpenAI and Anthropic models for NLU intent classification tasks across different domains. Here's a summary of the results:

### General Dataset Results

- OpenAI Performance (using gpt-4o-mini model):
  - Total Correct: 1338
  - Total Attempts: 1554
  - Overall Accuracy: 86.10%

- Anthropic Performance (using claude-3-haiku-20240307 model):
  - Total Correct: 1237
  - Total Attempts: 1554
  - Overall Accuracy: 79.60%

- Final Overall Performance:
  - Total Correct: 2575
  - Total Attempts: 3108
  - Overall Accuracy: 82.85%

### Banking Dataset Results

- OpenAI Performance (using gpt-4o-mini model):
  - Total Correct: 2051
  - Total Attempts: 3080
  - Overall Accuracy: 66.59%

- Anthropic Performance (using claude-3-haiku-20240307 model):
  - Total Correct: 2159
  - Total Attempts: 3080
  - Overall Accuracy: 70.10%

- Final Overall Performance:
  - Total Correct: 4210
  - Total Attempts: 6160
  - Overall Accuracy: 68.34%

These results show that both models perform well in intent classification tasks, with some interesting variations across datasets:

1. In the general dataset, OpenAI's model slightly outperformed Anthropic's model.
2. In the banking-specific dataset, Anthropic's model performed better than OpenAI's model.
3. Overall accuracy was higher for the general dataset compared to the banking dataset, suggesting that banking intents might be more challenging to classify (greater number of intents to pick from, 77) or require more domain-specific training.

It's important to note that these results are specific to the models used (OpenAI's gpt-4o-mini and Anthropic's claude-3-haiku-20240307) and the particular datasets employed. Performance may vary significantly with different models, datasets, or prompt engineering strategies. Factors such as the complexity of the intents, the diversity of the utterances, and the specific domain of the data can all influence the results.

Furthermore, it's crucial to understand that the binary nature of "match" or "no match" in the evaluation may not always accurately reflect the quality of the intent classification. Some classifications marked as "no match" might still be valid or reasonable interpretations of the user's intent. Consider the following example:

```json
{
  "custom_id": "utterance_8",
  "utterance": "Hi! Can I customize a half-and-half pizza with different toppings on each side?",
  "classification": {
    "predicted_intent": {
      "intent": "Special Requests",
      "confidence": 0.9
    },
    "second_predicted_intent": {
      "intent": "Menu and Ingredients",
      "confidence": 0.8
    },
    "true_intent": "Order Placement",
    "match": false
  }
}
```

In this case, while the original intent was labeled as "Order Placement", the model's prediction of "Special Requests" is also a valid interpretation of the user's utterance. The question about customizing a pizza could reasonably be categorized as both a special request and the beginning of an order placement process. This example highlights the potential for multiple valid intents for a single utterance and the importance of considering context and nuance in intent classification tasks.

Lastly, it's worth emphasizing again the importance of enforcing structured JSON outputs in this process. By requiring both OpenAI and Anthropic models to adhere to a specific JSON format, we ensure consistency in the response structure. This consistency is crucial for accurate parsing, analysis, and comparison of the results. Without this structured output, the evaluation process would be significantly more complex and potentially less reliable. The structured output allows for automated processing of large datasets and enables direct comparisons between different models and across various intents and domains.
