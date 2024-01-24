from typing import Optional
from afri_rlhf.data.sources import *
from afri_rlhf.prompt.templates import get_prompt_template_by_id, parse_prompt_json
from afri_rlhf.utils.language import get_language_by_iso_code
from afri_rlhf.prompt.template_utils import get_instruction_templates_from_excel
from datasets import concatenate_datasets, DatasetDict
import string
import random
import os
import numpy as np

def generate_dataset_by_prompt(prompts, datasource_class, split, languague_iso_code="amh", **kwargs):

    language = get_language_by_iso_code(languague_iso_code)
    if prompts[0].task_type == "translation":
        kwargs.pop("source_language")
        kwargs.pop("target_language")
    source = datasource_class(language=language, split = split, prompts=prompts, **kwargs)
    dataset = source.load_dataset() #.remove_columns(["prompt_header", "datasource"])

    return dataset

def generate_dataset_by_prompt_id(split, prompt_id, datasource_class, **kwargs):
    prompt = get_prompt_template_by_id(prompt_id)
    return generate_dataset_by_prompt([prompt], datasource_class, split, languague_iso_code="amh", **kwargs)


def generate_random_id(length):
    possible_values = string.digits + string.ascii_lowercase
    return "".join([ random.choice(possible_values) for _ in range(length)])


def generate_dataset_from_instruction_templates(instruction_templates, split, datasource_class, languague_iso_code="amh", randomize_prompts= False, **kwargs):
    prompts = []
    for template in instruction_templates:
        prompt_id = "id" + generate_random_id(10)
        prompt = parse_prompt_json(prompt_id, template)
        prompts.append(prompt)
    
    if randomize_prompts:
        ds = generate_dataset_by_prompt(prompts, datasource_class, split, languague_iso_code=languague_iso_code, **kwargs)
        return ds 
    else:
        ds_list = []
        for prompt in prompts:
            ds = generate_dataset_by_prompt([prompt], datasource_class, split, languague_iso_code=languague_iso_code, **kwargs)
            ds_list.append(ds)
        return concatenate_datasets(ds_list)

def generate_dataset_from_instruction_templates_excel_sheet(
    excel_path, 
    excel_sheet_name,
    task_type, 
    datasource_class, 
    split, languague_iso_code="amh",
    randomize_prompts: bool = False, 
    num_templates_to_use: Optional[int] = None,
    max_output_size: Optional[int] = None,
    **kwargs):
    
    task_sub_type = kwargs.pop("task_sub_type", None)
    
    instruction_templates = get_instruction_templates_from_excel(excel_path, excel_sheet_name, task_type, task_sub_type = task_sub_type,  **kwargs)
    if num_templates_to_use is not None:
        instruction_templates = instruction_templates[:num_templates_to_use]
    dataset =  generate_dataset_from_instruction_templates(instruction_templates, split, datasource_class, languague_iso_code, randomize_prompts=randomize_prompts, **kwargs)

    if max_output_size is None or max_output_size >  len(dataset):
        return dataset
    
    rand_indices = np.random.choice(len(dataset), size = max_output_size, replace=False)
    return dataset.select(rand_indices)


def main():


    training_datasets = concatenate_datasets([
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Sentiment Analysis", "classification", datasource_class=AfriSentDatasource, split="train", task_sub_type="sentiment_classification", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Masakhanews", "classification", datasource_class=MasakhaNewsClassificationDatasource, split="train", task_sub_type="news_classification", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "NER", "classification", datasource_class=MasakhaNERDatasource, split="train", task_sub_type="token_classification", entities_to_extract={"DATE", "LOC", "PER", "ORG"}, empty_entities_output = "ስም አልተገኘም", use_v2=False, randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "summarization", "text_generation", datasource_class=XlsumDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Reverse  summarization", "text_generation", datasource_class=XlsumReverseDatasource, split="train", max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Masakhanews - title generation", "text_generation", datasource_class=AmharicNewsTitleGenerationDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "PoemCompletion", "text_completion", datasource_class=AmharicPoemCompletionDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "PoemGeneration", "text_generation", datasource_class=AmharicPoemGenerationDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "ZefenCompletion", "text_completion", datasource_class=AmharicZefenDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "ZefenGeneration", "text_generation", datasource_class=AmharicZefenGenerationDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Story generation", "text_generation", datasource_class=AmharicStoryGenerationDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurComplition", "text_completion", datasource_class=AmharicMezmurCompletionDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurComplition", "text_completion", datasource_class=AmharicMezmurCompletionDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurGeneration", "text_generation", datasource_class=AmharicMezmurGenerationDatasource, split="train", randomize_prompts=False, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MT", "translation", datasource_class=AmharicEnglishMTDatasource, split="train", source_language="አማርኛ", target_language="English", randomize_prompts=True, max_output_size = 10000),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MT", "translation", datasource_class=EnglishAmharicMTDatasource, split="train", source_language="English", target_language="አማርኛ", randomize_prompts=True, max_output_size = 10000)
    ])

    validation_datasets = concatenate_datasets([
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Sentiment Analysis", "classification", datasource_class=AfriSentDatasource, split="validation", task_sub_type="sentiment_classification", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Masakhanews", "classification", datasource_class=MasakhaNewsClassificationDatasource, split="validation", task_sub_type="news_classification", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "NER", "classification", datasource_class=MasakhaNERDatasource, split="validation", task_sub_type="token_classification", entities_to_extract={"DATE", "LOC", "PER", "ORG"}, empty_entities_output = "ስም አልተገኘም", use_v2=False, randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "summarization", "text_generation", datasource_class=XlsumDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Reverse  summarization", "text_generation", datasource_class=XlsumReverseDatasource, split="validation", num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Masakhanews - title generation", "text_generation", datasource_class=AmharicNewsTitleGenerationDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "PoemCompletion", "text_completion", datasource_class=AmharicPoemCompletionDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "PoemGeneration", "text_generation", datasource_class=AmharicPoemGenerationDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "ZefenCompletion", "text_completion", datasource_class=AmharicZefenDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "ZefenGeneration", "text_generation", datasource_class=AmharicZefenGenerationDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Story generation", "text_generation", datasource_class=AmharicStoryGenerationDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurComplition", "text_completion", datasource_class=AmharicMezmurCompletionDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurComplition", "text_completion", datasource_class=AmharicMezmurCompletionDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurGeneration", "text_generation", datasource_class=AmharicMezmurGenerationDatasource, split="validation", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MT", "translation", datasource_class=AmharicEnglishMTDatasource, split="validation", source_language="አማርኛ", target_language="English", randomize_prompts=True),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MT", "translation", datasource_class=EnglishAmharicMTDatasource, split="validation", source_language="English", target_language="አማርኛ", randomize_prompts=True)
    ])

    test_datasets = concatenate_datasets([
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Sentiment Analysis", "classification", datasource_class=AfriSentDatasource, split="test", task_sub_type="sentiment_classification", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Masakhanews", "classification", datasource_class=MasakhaNewsClassificationDatasource, split="test", task_sub_type="news_classification", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "NER", "classification", datasource_class=MasakhaNERDatasource, split="test", task_sub_type="token_classification", entities_to_extract={"DATE", "LOC", "PER", "ORG"}, empty_entities_output = "ስም አልተገኘም", use_v2=False, randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "summarization", "text_generation", datasource_class=XlsumDatasource, split="test", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Reverse  summarization", "text_generation", datasource_class=XlsumReverseDatasource, split="test", num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Masakhanews - title generation", "text_generation", datasource_class=AmharicNewsTitleGenerationDatasource, split="test", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "PoemCompletion", "text_completion", datasource_class=AmharicPoemCompletionDatasource, split="test", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "PoemGeneration", "text_generation", datasource_class=AmharicPoemGenerationDatasource, split="test", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "ZefenCompletion", "text_completion", datasource_class=AmharicZefenDatasource, split="test", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "ZefenGeneration", "text_generation", datasource_class=AmharicZefenGenerationDatasource, split="test", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Story generation", "text_generation", datasource_class=AmharicStoryGenerationDatasource, split="test", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurComplition", "text_completion", datasource_class=AmharicMezmurCompletionDatasource, split="test", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurGeneration", "text_generation", datasource_class=AmharicMezmurGenerationDatasource, split="test", randomize_prompts=False, num_templates_to_use=1),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MT", "translation", datasource_class=AmharicEnglishMTDatasource, split="test", source_language="አማርኛ", target_language="English", randomize_prompts=True),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MT", "translation", datasource_class=EnglishAmharicMTDatasource, split="test", source_language="English", target_language="አማርኛ", randomize_prompts=True),
    ])


    dataset = DatasetDict(dict(train=training_datasets, validation=validation_datasets, test=test_datasets))
    output_path = "../logs/afri-rlhf"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dataset.save_to_disk(output_path)
    dataset.push_to_hub("israel/JOPUjJHxWmI5x",private=True)


if __name__ == "__main__":
    main()