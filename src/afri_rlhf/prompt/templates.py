from typing import Any,Dict, Optional, Type
import sys

from .validation import validate_supported_languages, validate_supported_task_types


prompt_templates = {
    "with_inputs": """
            {prompt_header}
            ### Instruction:
            {instruction}

            ### Input:
            {input}

            ### Response: {output}
        """,
    "without_inputs": """
            {prompt_header}
            ### Instruction:
            {instruction}

            ### Response: {output}
        """
}

amharic_prompt_header_with_inputs = "ከዚህ በታች አንድን ተግባር የሚገልጽ መመሪያ ተጨማሪ አውድ ከሚሰጥ ግብአት ጋር ተጣምሮ አለ። ጥያቄውን በትክክል የሚያጠናቅቅ ምላሽ ስጥ።"
prompts = {
    "id001": {
        "task_type": "classification",
        "task_sub_type": "sentiment_classification",
        "prompt_language": "amh",
        "header": amharic_prompt_header_with_inputs,
        "prompt_template": "with_inputs",
        "instruction": """የተሰጠው ጽሑፍ አስተያየት ምን አይነት ነው? "አዎንታዊ"፣ "አሉታዊ" ወይም "ገለልተኛ" የምል ምላሽ ስጥ።""",
        "prompt_class": "ClassificationPrompt",
    },
    "id002": {
        "task_type": "translation",
        "prompt_language": "amh",
        "source_language": "English",
        "target_language": "አማርኛ",
        "header": amharic_prompt_header_with_inputs,
        "prompt_template": "with_inputs",
        "instruction": "የተሰጠውን ጽሁፍ  ከ {source_language} ቋንቋ ወደ {target_language} ቋንቋ ተርጉም።",
        "prompt_class": "TranslationPrompt"
    },
    "id003": {
        "task_type": "summarization",
        "prompt_language": "amh",
        "header": amharic_prompt_header_with_inputs,
        "prompt_template": "with_inputs",
        "instruction": "ለሚከተለውን ጽሑፍ ማጠቃለያ ስጥ።",
        "prompt_class": "Prompt"
    },

    "id004": {
        "task_type": "headline_generation",
        "prompt_language": "amh",
        "header": amharic_prompt_header_with_inputs,
        "prompt_template": "with_inputs",
        "instruction": "ለተሰጠ ዝርዝር ዜና አርዕስተ ዜና ስጥ።",
        "prompt_class": "Prompt"
    },
    "id005": {
        "task_type": "classification",
        "task_sub_type": "news_classification",
        "prompt_language": "amh",
        "header": amharic_prompt_header_with_inputs,
        "prompt_template": "with_inputs",
        "instruction": """የሚከተለውን ዜና ንባብ ወደ ትክክለኛው መድብ መድብ። ከ "ብዝነስ", "መዝናኛ", "ጤና", "ፖለቲካ", "ሀይማኖት", "ስፖርት" ወይንም "ቴክኖሎጂ" መደቦች ውስጥ አንዱን ምረጥ።""",
        "prompt_class": "ClassificationPrompt",
    },

    "id006": {
        "task_type": "classification",
        "task_sub_type": "token_classification",
        "prompt_language": "amh",
        "header": amharic_prompt_header_with_inputs,
        "prompt_template": "with_inputs",
        "instruction": """ከሚከተለው ዐረፍተ ነገር ውስጥ የሰው ስም ዝርዝር አውጣ። አንድም ስም ከለለ፣ "ስም አልተገኘም" በማለት ምላሽ ስጥ።""",
        "prompt_class": "ClassificationPrompt",
        "prompt_class": "ClassificationPrompt",
    },

}


class Prompt:
    def __init__(self, id, *, header, instruction, task_type, prompt_language, prompt_template):
        self.id = id
        self.header = header
        self.instruction = instruction
        self.task_type = task_type
        self.prompt_language = prompt_language
        self.prompt_template = prompt_template
    def __hash__(self):
        return (self.id, self.task_type, self.prompt_language, self.prompt_template)
    def __str__(self) -> str:
        return """Prompt(\nid= {self.id},\n\theader= {self.header},\n\tinstructions= {self.instructions},\n\ttask_type = {self.task_type},\n\tprompt_language = {self.prompt_language},\n\tprompt_template= {self.prompt_template})"""
    def __repr__(self):
        return str(self)
    def format(self, **kwargs):
        return prompt_templates[self.prompt_template].format(**kwargs)
    

class ClassificationPrompt(Prompt):
    def __init__(self, id, *, header, instruction, task_type, prompt_language, prompt_template, task_sub_type):
        super().__init__(id, header=header, instruction=instruction, task_type = task_type, prompt_language = prompt_language, prompt_template = prompt_template)
        self.task_sub_type = task_sub_type
    def __hash__(self):
        return super().__hash__() + (self.task_sub_type, )
    def __str__(self) -> str:
        return """Prompt(\nid= {self.id},\n\theader= {self.header},\n\tinstructions= {self.instructions},\n\ttask_type = {self.task_type},\n\tprompt_language = {self.prompt_language},\n\tprompt_template= {self.prompt_template},\n\ttask_sub_type= {self.task_sub_type})"""
    

class TranslationPrompt(Prompt):
    def __init__(self, id, *, header, instruction, task_type, prompt_language, prompt_template, source_language, target_language):
        super().__init__(id, header=header, instruction=instruction, task_type = task_type, prompt_language = prompt_language, prompt_template = prompt_template)
        self.instruction = instruction.format(source_language = source_language, target_language = target_language)
    def __hash__(self):
        return super().__hash__() + (self.task_sub_type, )
    def __str__(self) -> str:
        return """Prompt(\nid= {self.id},\n\theader= {self.header},\n\tinstructions= {self.instructions},\n\ttask_type = {self.task_type},\n\tprompt_language = {self.prompt_language},\n\tprompt_template= {self.prompt_template},\n\tsource_language= {self.source_language},\n\ttarget_language=target_language)"""
    

def get_class_with_name(classname) -> Type:
    return getattr(sys.modules[__name__], classname)

def parse_prompt_json(id: str, content: Dict[str, Any]) -> Prompt:
    content = content.copy()
    prompt_class_name = content.pop("prompt_class")
    return get_class_with_name(prompt_class_name)(id, **content)

    

validate_supported_languages(prompts)
validate_supported_task_types(prompts)

def get_prompt_template_by_id(id: str) -> Prompt:
    content = prompts[id]
    return parse_prompt_json(id, content)

def get_prompt_template_by_language_and_task(language: str, task_type: str, task_sub_type: Optional[str] = None) -> Prompt:
    for prompt_id, prompt_content in prompts.items():
        if prompt_content["prompt_language"] == language and prompt_content["task_type"] == task_type:
            if task_sub_type is not None:
                
                if "task_sub_type" in prompt_content:
                    if prompt_content["task_sub_type"] == task_sub_type:
                        return get_prompt_template_by_id(prompt_id)
                else:
                    raise ValueError("task_sub_type is not supported for {} task type".format(task_type))
            else:
                return get_prompt_template_by_id(prompt_id)
    return None
