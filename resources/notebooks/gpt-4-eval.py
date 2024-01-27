import datasets
import openai
import pandas as pd
import time
import os
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm
import csv
import tiktoken


hf_token = os.environ.get("HUGGING_FACE_TOKEN")
openai_api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI()

evaluation_data  = datasets.load_dataset("israel/JOPUjJHxWmI5x", use_auth_token=hf_token)




def split_or_keep_text(text, max_tokens=8100, delimiter="\n"):
    tiktoken.encoding_for_model('gpt-4')
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(text)
    
    print(f"len tokens {len(tokens)}")
    if len(tokens) <= max_tokens:
        return text
    else:
        tokens = tokens[:8100]
        text = tokenizer.decode(tokens)
        return text


def query_gpt4(instruction, input_text):
    message = f"{instruction}\n '''{input_text}''' "
    # # print(message)
    # print()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
        {"role": "user", "content": split_or_keep_text(message)}])
    return response.choices[0].message.content.strip()




def evaluate(test_data, data_sources, output_dir='model_evaluation', sleep_duration=2, resume=False):
    os.makedirs(output_dir, exist_ok=True)
    start_index = 0
    data_source_name = "-".join(data_sources)
    output_filename = os.path.join(output_dir, f"gpt4_responses-{data_source_name}.csv")
    print(f"gpt4_responses-{data_source_name}.csv")
    
    if resume and os.path.exists(output_filename):
        with open(output_filename, mode='r') as file:
            reader = csv.reader(file)
            data = list(reader)
        start_index = len(data)-1
        print(f"Resuming from Test Case {start_index}")
    else:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['instruction', 'input', 'ouptput','datasource', 'response'])  # Column headers

    datasource_data = test_data.filter(lambda example: example['datasource'] in data_sources)
    resume_data = datasource_data[start_index:]

    with open(output_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        for i, (
            instruction,
            input_text,
            output_text,
            data_source
        ) in enumerate(
            tqdm(
                zip(
                    resume_data['instruction'],
                    resume_data['input'],
                    resume_data['output'],
                    resume_data['datasource']
                ),
                total=len(datasource_data['input']),
                initial=start_index
            )
        ): 
            try:
                response = query_gpt4(instruction, input_text)
                writer.writerow([instruction, input_text, output_text, data_source, response])
            except Exception as e:
                print(f"API Error for {data_source} - Test Case {start_index + i}: {str(e)}")
                break

            time.sleep(sleep_duration)  # Add a sleep to avoid API rate limits
    
    print(f"Final save: Saved responses to {output_filename}")


print(evaluation_data['test'].to_pandas().datasource.value_counts())

evaluate(evaluation_data['test'], ['masakhanews'],output_dir="masakhanews",sleep_duration=0, resume=True)
