from typing import Any, Dict, List
import pandas as pd


def load_template(sheet_path: str, sheet_name: str) -> pd.DataFrame:
    """Loads excel sheet from excel file.

    Args:
        sheet_path (str): Path to the file
        sheet_name (str): Name of the sheet to load

    Returns:
        pd.DataFrame: Data frame containing templates
    """


    with open(sheet_path, "rb") as input_file:
        return pd.read_excel(input_file, sheet_name=sheet_name)  
    


def create_instruction_templates(raw_templates: List[str], task_type, **kwargs) -> List[Dict[str, Any]]:
    base_template = {
        "task_type": task_type,
        "prompt_language": "en_us",

        # TODO: Add support for customization of the header. Currently this is not being used
        "header": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
        
        # Currently this is not used.
        "prompt_template": "with_inputs", 
    }

    if task_type == "classification":
        if not "task_sub_type" in kwargs:
            raise ValueError("classification task type requires `task_sub_type`")
        base_template["task_sub_type"] = kwargs.pop("task_sub_type")
    
        base_template["prompt_class"] = "ClassificationPrompt"
    elif task_type == "translation":
        base_template["source_language"] = kwargs.pop("source_language")
        base_template["target_language"] = kwargs.pop("target_language")
        base_template["prompt_class"] = "TranslationPrompt"
    else:
        base_template["prompt_class"] = "Prompt"
    outputs = []

    for template in raw_templates:
        base_template_copy = base_template.copy()
        base_template_copy["instruction"] = template
        outputs.append(base_template_copy)
    return outputs
def get_instruction_templates_from_excel(path, sheet_name, task_type, **kwargs):
    templates_df = load_template(path, sheet_name)
   
    instruction_templates = create_instruction_templates(templates_df.instruction.tolist(), task_type, **kwargs)
    return instruction_templates

if __name__ == "__main__":
    print(get_instruction_templates_from_excel("../resources/Template Generation.xlsx", "NER", "NER"))