import os
import json
import anthropic
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm

# Set up client for API calls
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Actual API call, with exponential backoff and 10 retries
@retry(wait=wait_exponential(multiplier=1, min=5, max=60), stop=stop_after_attempt(9999))
def make_api_call(prompt, model, max_tokens, temperature):
    message = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return message

# Function to process a single question with or without context
def process_question(question, model, max_tokens, temperature, context=None):
    if context:
        prompt = f"{context['context']}\nQuestion: {question['question']}"
        special_context = context['category']
    else:
        prompt = f"Question: {question['question']}"
        special_context = "no_prompt"
    
    response = make_api_call(prompt, model, max_tokens, temperature)
    
    return {
        "question": question['question'],
        "category": question['category'],
        "response": response.content[0].text,
        "num_prompt_tokens": response.usage.input_tokens,
        "num_response_tokens": response.usage.output_tokens,
        "special_context": special_context,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

# Main function to process all questions
def process_questions(questions_file, special_context_file, output_file, model, max_tokens, temperature):
    with open(questions_file, "r") as f:
        questions = [json.loads(line) for line in f]
    
    with open(special_context_file, "r") as f:
        special_contexts = [json.loads(line) for line in f]

    # # Process each question without context first
    # for question in tqdm(questions, desc="Processing questions without context"):
    #     result = process_question(question, model, max_tokens, temperature)
    #     with open(output_file, "a") as f:
    #         json.dump(result, f, ensure_ascii=False)
    #         f.write("\n")
    
    # Process each question with each special context
    for question in tqdm(questions, desc="Processing questions with context"):
        for special_context in tqdm(special_contexts, desc="Processing special contexts"):
            result = process_question(question, model, max_tokens, temperature, special_context)
            with open(output_file, "a") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")

if __name__ == "__main__":
    questions_file = "openended_questions.jsonl"
    special_context_file = "prompts_v3.jsonl"
    output_file = "openended_results_promptsv3.jsonl"
    model = "claude-3-opus-20240229"
    max_tokens = 1500
    temperature = 0.0
    process_questions(
        questions_file,
        special_context_file,
        output_file,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
