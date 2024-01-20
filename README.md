# RLHF Data Generation

This repository generates instruction tuning dataset from different datasets.  

```
src
    ├── afri_rlhf
    │   ├── __init__.py
    │   ├── data
    │   │   ├── __init__.py
    │   │   └── sources.py                # create Datasource class for your new dataset
    │   ├── prompt
    │   │   ├── __init__.py
    │   │   ├── templates.py               # Add template for your data 
    │   │   └── validation.py
    │   └── utils
    │       ├── __init__.py
    │       ├── language.py
    │       └── support.py
    └── create_rlhf_dataset.py
```


## How to add Datasource


Datasource classes serve as the interface for importing datasets from external sources. These specialized classes play a crucial role in loading datasets and generating instructional datasets specifically tailored for training Reinforcement Learning with Human Feedback (RLHF) models. The datasets sourced from external repositories encompass a diverse array of tasks, ranging from classification to summarization and translation.


All datasource classes within this repository should inherit from the [DatasourceBase](src/afri_rlhf/data/sources.py) class. This base class encapsulates essential functionalities required for loading datasets from external sources and facilitates the conversion of original datasets into instruction datasets. When implementing custom datasource classes, it is imperative to adhere to the following four methods defined in DatasourceBase:

Methods to Implement:
* `load_from_external(self) -> datasets.Dataset`:

This method is responsible for loading datasets from external sources, such as Hugging Face.
* `get_prompt_inputs(self, item: Dict[str, Any]) -> str`:

Utilized to retrieve inputs for instruction generation. Implement this method to define how inputs are extracted from the dataset items.
* `get_prompt_output(self, item: Dict[str, Any]) -> str`:

Used to obtain responses to instructions. Implement this method to define how outputs or responses are extracted from the dataset items.
* `get_datasource_name(self) -> str:`

Should return the dataset source name. This information is crucial for analytical purposes.


Example 
```python

class MaskahanePosDatasource(DatasourceBase):
    def load_from_external(self) -> datasets.Dataset:
        # Implementation to load dataset from external sources
        ...

    def get_prompt_inputs(self, item: Dict[str, Any]) -> str:
        # Implementation to extract inputs for instruction generation
        ...

    def get_prompt_output(self, item: Dict[str, Any]) -> str:
        # Implementation to extract responses to instructions
        ...

    def get_datasource_name(self) -> str:
        # Implementation to return the dataset source name
        ...

```

## How to add template


There are two ways to create prompt templates. The first first and easy way is to add templates description using dictionary inside [src/afri_rlhf/prompt/templates.py](src/afri_rlhf/prompt/templates.py). You can copy one of the templates that best suites your dataset sources and make any modification needed. 

The second option is storing templates inside a file and using `create_instruction_templates` method inside [src/afri_rlhf/prompt/template_utils.py](src/afri_rlhf/prompt/template_utils.py) to create list of prompt from the templates. 

Example adding POS template

```python
prompts = {
    ...


    "id010": {
        "task_type": "pos",
        "prompt_language": "amh",# This is currently not support but it should be provided to make it compatible in the future
        "header": amharic_prompt_header_with_inputs, # This is also not support, but should be provided for compatablity in the future
        "prompt_template": "with_inputs",  # This is also not support, but should be provided for compatablity in the future
        "instruction": """በዚህ አረፍተ ነገር ውስጥ እያንዳንዱን ቃል ስም፣ ግስ፣ ገላጭ፣ ወዘተ እያልክ መድብ""",
        "prompt_class": "Prompt",
    }
```


## How to create prompt object from a dictionary

Once a dictionary for the prompt template is available, the next step involves creating a prompt object. The creation of a prompt object necessitates the following:

ID: An identifier uniquely assigned to the prompt.
Prompt Class: This class abstracts the prompt template dictionary.
Contents for the Prompt Class Constructor: These contents include instructions, language, and other relevant details.

Several support prompt classes are provided:

* [Prompt](src/afri_rlhf/prompt/templates.py#L89): This serves as the base class for all prompts and can also be utilized to create various types of prompts.
* [ClassificationPrompt](src/afri_rlhf/prompt/templates.py#L107): This class is designed for creating prompts tailored to classification datasources.
* [TranslationPrompt](src/afri_rlhf/prompt/templates.py#L117): Specifically crafted for generating prompts suited for translation tasks.

```python

prompt_content = {
        "task_type": "pos",
        "prompt_language": "amh", # Currently not used! Will be used in the future
        "header": "",  # Currently not used! Will be used in the future
        "prompt_template": "with_inputs",   # Currently not used! Will be used in the future
        "instruction": """በዚህ አረፍተ ነገር ውስጥ እያንዳንዱን ቃል ስም፣ ግስ፣ ገላጭ፣ ወዘተ እያልክ መድብ""",
        "prompt_class": "Prompt",
    }

prompt = parse_prompt_json("id010", prompt_content)

```




## Generating datasets(Combining all steps)


```python

from afri_rlhf.prompt.templates import parse_prompt_json
from afri_rlhf.utils.language import get_language_by_iso_code

# Can be generated using random number generator
prompt_id = "id010"

# Prompt content.
prompt_content = {
        "task_type": "pos", # Currently not used! Will be used in the future
        "header": amharic_prompt_header_with_inputs,  # Currently not used! Will be used in the future
        "prompt_template": "with_inputs",  # Currently not used! Will be used in the future
        "instruction": """በዚህ አረፍተ ነገር ውስጥ እያንዳንዱን ቃል ስም፣ ግስ፣ ገላጭ፣ ወዘተ እያልክ መድብ""",
        "prompt_class": "Prompt",
}

prompt = parse_prompt_json("id010", prompt_content)
language = get_language_by_iso_code("amh")

source = MaskahanePosDatasource(language=language, split = split, prompt=prompt)
dataset = source.load_dataset(apply_formatting=True).remove_columns(["prompt_header", "datasource", "prompt"])


```


## Generating Instruction Dataset

Use the following command to generate instruction dataset using instruction templates inside the [template file](resources/Template Generation.xlsx).
```
cd src
python create_rlhf_dataset.py
```


