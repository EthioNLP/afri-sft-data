from afri_rlhf.data.sources import *
from afri_rlhf.prompt.templates import get_prompt_template_by_id, parse_prompt_json
from afri_rlhf.utils.language import get_language_by_iso_code
from afri_rlhf.prompt.template_utils import get_instruction_templates_from_excel
from datasets import concatenate_datasets, DatasetDict
import string
import random
import os

def generate_dataset_by_prompt(prompts, datasource_class, split, languague_iso_code="amh", **kwargs):

    language = get_language_by_iso_code(languague_iso_code)
    source = datasource_class(language=language, split = split, prompts=prompts, **kwargs)
    dataset = source.load_dataset(apply_formatting=True).remove_columns(["prompt_header", "datasource", "prompt"])
    return dataset

def generate_dataset_by_prompt_id(split, prompt_id, datasource_class, **kwargs):
    prompt = get_prompt_template_by_id(prompt_id)
    return generate_dataset_by_prompt(prompt, datasource_class, split, languague_iso_code="amh", **kwargs)


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
            ds = generate_dataset_by_prompt(prompt, datasource_class, split, languague_iso_code=languague_iso_code, **kwargs)
            ds_list.append(ds)
        return concatenate_datasets(ds_list)

def generate_dataset_from_instruction_templates_excel_sheet(
    excel_path, 
    excel_sheet_name,
    task_type, 
    datasource_class, 
    split, languague_iso_code="amh", randomize_prompts: bool = False, **kwargs):
    
    task_sub_type = kwargs.pop("task_sub_type", None)
    instruction_templates = get_instruction_templates_from_excel(excel_path, excel_sheet_name, task_type, task_sub_type = task_sub_type,  **kwargs)
    
    return generate_dataset_from_instruction_templates(instruction_templates, split, datasource_class, languague_iso_code, randomize_prompts=randomize_prompts, **kwargs)


def main():


    training_datasets = concatenate_datasets([
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Sentiment Analysis", "classification", datasource_class=AfriSentDatasource, split="train", task_sub_type="sentiment_classification", randomize_prompts=True),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Masakhanews", "classification", datasource_class=MasakhaNewsClassificationDatasource, split="train", task_sub_type="news_classification"),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "summarization", "text_generation", datasource_class=XlsumDatasource, split="train"),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Reverse  summarization", "text_generation", datasource_class=XlsumReverseDatasource, split="train"),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Masakhanews - title generation", "text_generation", datasource_class=MasakhaNewsHeadlineGenerationDatasource, split="train"),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "PoemComplition", "text_completion", datasource_class=AmharicPoemCompletionDatasource, split="train"),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "ZefenComplition", "text_completion", datasource_class=AmharicZefenDatasource, split="train"),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "Story generation", "text_generation", datasource_class=AmharicStoryGenerationDatasource, split="train"),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurComplition", "text_completion", datasource_class=AmharicMezmurCompletionDatasource, split="train"),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurComplition", "text_completion", datasource_class=AmharicMezmurCompletionDatasource, split="train"),
        generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MezmurGeneration", "text_generation", datasource_class=AmharicMezmurGenerationDatasource, split="train"),

        # Fix the issue with translation(Needs further check if we support the templates)
        # generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MT", "translation", datasource_class=AmharicEnglishMTDatasource, split="train"),
        # generate_dataset_from_instruction_templates_excel_sheet("../resources/Template Generation.xlsx", "MT", "translation", datasource_class=AmharicEnglishMTDatasource, split="train")
    ])

    validation_datasets = concatenate_datasets([
        generate_dataset_by_prompt_id("validation", "id001", AfriSentDatasource),
        generate_dataset_by_prompt_id("validation", "id005", MasakhaNewsClassificationDatasource),
        generate_dataset_by_prompt_id("validation", "id003", XlsumDatasource),
        generate_dataset_by_prompt_id("validation", "id006", MasakhaNERDatasource, entity_to_extract= "PER", empty_entities_output = "ስም አልተገኘም", use_v2=False),
        
    ])
    test_datasets = concatenate_datasets([
        generate_dataset_by_prompt_id("test", "id001", AfriSentDatasource),
        generate_dataset_by_prompt_id("test", "id005", MasakhaNewsClassificationDatasource),
        generate_dataset_by_prompt_id("test", "id003", XlsumDatasource),
        generate_dataset_by_prompt_id("test", "id006", MasakhaNERDatasource, entity_to_extract= "PER", empty_entities_output = "ስም አልተገኘም", use_v2=False),
        
    ])


    dataset = DatasetDict(dict(train=training_datasets, validation=validation_datasets, test=test_datasets))
    output_path = "../logs/afri-rlhf"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dataset.save_to_disk(output_path)


if __name__ == "__main__":
    main()