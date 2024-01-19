from abc import abstractmethod, ABC
from typing import Any, Dict, Literal
import datasets
import pandas as pd
from datasets import Dataset
import os

from afri_rlhf.prompt.templates import Prompt
from afri_rlhf.utils.language import Language



class DatasourceBase(ABC):
    def __init__(self, *, language: Language, split: str, prompt: Prompt):
        self.split = split
        self.language = language
        self.prompt = prompt

    @abstractmethod
    def get_datasource_name(self):
        raise NotImplementedError

    @abstractmethod
    def load_from_external(self) -> datasets.Dataset:
        """Loads dataset from extranal sources such as Huggingface.

        Returns:
            datasets.Dataset: Dataset loaded from extral sources.
        """
        raise NotImplementedError
    

    def format_prompt(self, prompt_sections: Dict[str, str]) -> Dict[str, str]:
        """Creates a prompt from a record. It makes it possible to train RLHF model
        from classification datasets. Prompt sections are header, instructions, input and response. 
        The input is optional and can be omitted.

        Args:
            prompt_sections (Dict[str, Any]): Dictionary containing prompt sections

        Returns:
            Dict[str, str]: Dictionary with formatted prompt. The prompt is created from the prebuilt prompt
            templates and the current prompt sections
        """

        output = prompt_sections.copy()
        output["prompt"] = self.prompt.format(**prompt_sections)
        return output

    def get_prompt_sections(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Gets prompt sections from classification datasets. Prompt sections are header, instructions, input and response. 
        The input is optional and can be omitted.

        Args:
            item (Dict[str, Any]): Dataset record.

        Returns:
            Dict[str, str]: Prompt sections
        """
        return {
            "instruction": self.prompt.instruction,
            "input": self.get_prompt_inputs(item),
            "output": self.get_prompt_output(item),
            "prompt_header": self.prompt.header,
            "datasource": self.get_datasource_name()
        }

    @abstractmethod
    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        """Gets prompt inputs from a dataset record

        Args:
            item (Dict[str, Any]): Dataset record

        Returns:
            str: Prompt inputs
        """
        raise NotImplementedError

    @abstractmethod
    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        """Gets prompt output from a dataset record

        Args:
            item (Dict[str, Any]): Dataset record

        Returns:
            str: Prompt output
        """
        raise NotImplementedError

    def load_dataset(self, apply_formatting = False) -> datasets.Dataset:
        """Loads the dataset

        Returns:
            datasets.Datasets: Dataset after applying prompt formatting.
        """
        
        dataset = self.load_from_external()
        original_columns = dataset.column_names
        
        dataset = dataset.map(self.get_prompt_sections, batched=False).remove_columns(original_columns)
        
        if apply_formatting:
            dataset = dataset.map(self.format_prompt)
        
        return dataset
    

class ClassificationDatasourceBase:

    id_to_label: Dict[int, str] = None
    

    @property
    def label_to_id(self) -> Dict[str, int]:
        return {value:key for key, value in self.id_to_label.items()}
    
class PrivateDatasource(DatasourceBase):
    def __init__(self, *, language: Language, split: str, prompt: Prompt):
        super().__init__(language=language, split=split, prompt=prompt)
        self.hf_token =  os.environ.get("HuggigFace_TOKEN")

    def load_from_external(self) -> datasets.Dataset:
        return datasets.load_dataset("israel/AmharicQA", token=self.hf_token, split=self.split)
    
class AfriSentDatasource(DatasourceBase, ClassificationDatasourceBase):

    id_to_label: Dict[int, str] = {
        0: "አዎንታዊ",
        1: "ገለልተኛ",
        2: "አሉታዊ"
    }


    def __init__(self, *, language: str,  split: str,  prompt) -> None:
        super().__init__(language=language, split = split, prompt=prompt)

    def load_from_external(self):
        return datasets.load_dataset("shmuhammad/AfriSenti-twitter-sentiment", self.language.iso_code, split = self.split)
    
    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["tweet"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return self.id_to_label[item["label"]]
    
    def get_datasource_name(self):
        return "afrisent"
    

class MasakhaNewsDatasource(DatasourceBase, ABC):
    def load_from_external(self):
        return datasets.load_dataset("masakhane/masakhanews", self.language.iso_code)[self.split]
    
    def get_datasource_name(self):
        return "masakhanews"

class MasakhaNewsHeadlineGenerationDatasource(MasakhaNewsDatasource):

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["text"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["headline"]

class MasakhaNewsClassificationDatasource(MasakhaNewsDatasource, ClassificationDatasourceBase):
    id_to_label: Dict[int, str] = {
            0: "ብዝነስ",
            1: "መዝናኛ",
            2: "ጤና",
            3: "ፖለቲካ",
            4: "ሀይማኖት",
            5: "ስፖርት",
            6: "ቴክኖሎጂ"
        }

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["text"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return self.id_to_label[item["label"]]
    
class MasakhaNERDatasource(DatasourceBase, ClassificationDatasourceBase):
    id_to_label: Dict[int, str] = {
            0: "O",
            1: "B-DATE",
            2: "I-DATE",
            3: "B-PER",
            4: "I-PER",
            5: "B-ORG",
            6: "I-ORG",
            7: "B-LOC",
            8: "I-LOC"
        }
    

    def __init__(self, *, language: Language, split: str, prompt: Prompt, entity_to_extract: str, empty_entities_output: str, use_v2: bool = True):
        super().__init__(language=language, split=split, prompt=prompt)
        self.use_v2 = use_v2
        self.entity_to_extract = entity_to_extract.upper().replace("B-", "").replace("I-", "")
        self.empty_entities_output = empty_entities_output


    def load_from_external(self):
        if self.use_v2:
            return datasets.load_dataset("masakhane/masakhaner2", self.language.iso_code)[self.split]
        else:
            return datasets.load_dataset("masakhaner", self.language.iso_code)[self.split]
    def is_current_entity_label(self, label):
        return label.upper().replace("B-", "").replace("I-", "") == self.entity_to_extract

    def extract_named_entities(self, tokens, tag_ids):
        entity_entity_words = []
        current_entity = ""

        for word, tag_id in zip(tokens, tag_ids):
            tag = self.id_to_label[tag_id]
            if self.is_current_entity_label(tag):
                if tag.startswith("I-"):
                    current_entity += " " + word
                else:
                    if current_entity != "":
                        entity_entity_words.append(current_entity)
                    current_entity = word

            else:
                if current_entity != "":
                    entity_entity_words.append(current_entity)
                current_entity = ""
        if current_entity != "":
            entity_entity_words.append(current_entity)

        return entity_entity_words
    
    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return " ".join(item["tokens"])

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        
        assert len(item["ner_tags"]) == len(item["tokens"])

        entity_entity_words = self.extract_named_entities(item["tokens"], item["ner_tags"])
            
        entity_entity_words = ", ".join(entity_entity_words)

        if entity_entity_words == "":
            return self.empty_entities_output 
       
        return entity_entity_words
    
    def get_datasource_name(self):
        return "masakhaner"

        
CCAlignedDatasourceTypes = Literal["sentences", "documents"]
class CCAlignedDatasource(DatasourceBase):
    def __init__(self, *, language: Language, split: str, prompt: Prompt, source_type, transilate_to_english: bool = False) -> None:
        super().__init__(language=language, split = split, prompt=prompt)
        self.source_type = source_type        
        self.transilate_to_english = transilate_to_english
    
    def load_from_external(self) -> datasets.Dataset:
        language_locale_code = self.language.locale_code.replace("-", "_")
        dataset = datasets.load_dataset("ccaligned_multilingual", language_code=language_locale_code, type=self.source_type)[self.split]
        return dataset
   
    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        if self.transilate_to_english:
            source_language_code = self.language.locale_code.replace("-", "_")
        else:
            source_language_code = "en_XX"


        return item["translation"][source_language_code]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        if self.transilate_to_english:
            target_language_code = "en_XX"

        else:
            target_language_code = self.language.locale_code.replace("-", "_")

        return item["translation"][target_language_code]
    
    def get_datasource_name(self):
        return "ccaligned"

class XlsumDatasource(DatasourceBase):
    def __init__(self, *, language: Language, split: str, prompt: Prompt):
        super().__init__(language=language, split=split, prompt=prompt)

    def load_from_external(self) -> datasets.Dataset:
        language = self.language.english_name.lower()
        dataset = datasets.load_dataset("csebuetnlp/xlsum", language)[self.split]
        return dataset

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["text"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["summary"]
    
    def get_datasource_name(self):
        return "xlsum"

class QADatasource(PrivateDatasource):


    def __init__(self, *, language: str,  split: str,  prompt) -> None:
        super().__init__(language=language, split = split, prompt=prompt)

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["context"]+"\n\n"+item["question"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["answer"]
    
    def get_datasource_name(self):
        return "amharicqa"

## Summarization
# https://huggingface.co/datasets/csebuetnlp/xlsum
"""
Papers to read
* Xl summary
* https://openreview.net/pdf?id=ybc9V6Cbq2
* LLaMA
* LLAMA adapter
* Chinese LLAMA
"""