from abc import abstractmethod, ABC
from typing import Any, Dict, List, Literal, Union, Set
import datasets
import os

from afri_rlhf.prompt.templates import Prompt
from afri_rlhf.utils.language import Language
import random


class DatasourceBase(ABC):
    def __init__(self, *, language: Language, split: str, prompts: Union[Prompt, List[Prompt]]):
        self.split = split
        self.language = language
        if type(prompts) == Prompt:
            prompts = [prompts]
            
        self.prompts = prompts

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
    


    def get_prompt_sections(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Gets prompt sections from classification datasets. Prompt sections are header, instructions, input and response. 
        The input is optional and can be omitted.

        Args:
            item (Dict[str, Any]): Dataset record.

        Returns:
            Dict[str, str]: Prompt sections
        """
        if len(self.prompts) == 1:
            prompt = self.prompts[0]
        else:
            prompt = random.choice(self.prompts)

        return {
            "instruction": prompt.instruction,
            "input": self.get_prompt_inputs(item),
            "output": self.get_prompt_output(item),
            "prompt_header": prompt.header,
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

    def load_dataset(self) -> datasets.Dataset:
        """Loads the dataset

        Returns:
            datasets.Datasets: Dataset after applying prompt formatting.
        """
        
        dataset = self.load_from_external()
        original_columns = dataset.column_names
        
        dataset = dataset.map(self.get_prompt_sections, batched=False).remove_columns(original_columns)
    
        
        return dataset
    

class ClassificationDatasourceBase:

    id_to_label: Dict[int, str] = None
    

    @property
    def label_to_id(self) -> Dict[str, int]:
        return {value:key for key, value in self.id_to_label.items()}
    
class PrivateDatasource(DatasourceBase):
    def __init__(self, *, language: Language, split: str, prompts: Union[List[Prompt], Prompt]):
        super().__init__(language=language, split=split, prompts=prompts)
        self.hf_token =  os.environ.get("HuggigFace_TOKEN")
        
    @abstractmethod
    def get_dataset_location(self):
        raise NotImplementedError
    
    def load_from_external(self) -> datasets.Dataset:
        return datasets.load_dataset(self.get_dataset_location(), use_auth_token=self.hf_token, split=self.split)

    
class AfriSentDatasource(DatasourceBase, ClassificationDatasourceBase):

    id_to_label: Dict[int, str] = {
        0: "አዎንታዊ",
        1: "ገለልተኛ",
        2: "አሉታዊ"
    }
    
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
    def get_datasource_name(self):
        return "masakhanews_headline_generation"

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
        return f"ይህ ዜና {self.id_to_label[item['label']]} ነው"
    
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
    

    def __init__(self, *, language: Language, split: str, prompts:Union[Prompt, List[Prompt]], entities_to_extract: Set[str], empty_entities_output: str, use_v2: bool = True):
        super().__init__(language=language, split=split, prompts=prompts)
        self.use_v2 = use_v2
        self.entities_to_extract = set([entity_to_extract.upper().replace("B-", "").replace("I-", "") for entity_to_extract in entities_to_extract])
        self.empty_entities_output = empty_entities_output


    def load_from_external(self):
        if self.use_v2:
            return datasets.load_dataset("masakhane/masakhaner2", self.language.iso_code)[self.split]
        else:
            return datasets.load_dataset("masakhaner", self.language.iso_code)[self.split]
    def is_current_entity_label(self, label):
        return label.upper().replace("B-", "").replace("I-", "") in  self.entities_to_extract

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

class XlsumDatasource(DatasourceBase):
    
    def load_from_external(self) -> datasets.Dataset:
        language = self.language.english_name.lower()
        dataset = datasets.load_dataset("csebuetnlp/xlsum", language)[self.split]
        return dataset

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["text"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["summary"]
    
    def get_datasource_name(self):
        return "xlsum_summerization"

class QADatasource(PrivateDatasource):
    
    def get_dataset_location(self):
        return "israel/AmharicQA"
    
    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["context"]+"\n\n"+item["question"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["answer"]
    
    def get_datasource_name(self):
        return "amharicqa"


class XlsumReverseDatasource(XlsumDatasource):


    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["summary"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["text"]
    def get_datasource_name(self):
        return "xlsum_reverse_summerization"
    
    
class SpellingDatasource(PrivateDatasource):

    def get_dataset_location(self):
        return "israel/AmharicSpellCheck"

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["source"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["target"]
    
    def get_datasource_name(self):
        return "amharic_spellcheck"


    
class AmharicPoemCompletionDatasource(PrivateDatasource):
    def get_dataset_location(self):
        return "israel/AmharicPoem"

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["input"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["ouput"]
  
    def get_datasource_name(self):
        return "amharic_poem_completion"
    
class AmharicPoemGenerationDatasource(AmharicPoemCompletionDatasource):
    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return ""
    def get_datasource_name(self):
        return "amharic_poem_generation"

class AmharicZefenDatasource(AmharicPoemCompletionDatasource):

    def get_dataset_location(self):
        return "israel/AmharicZefen"

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["input"]

    def get_datasource_name(self):
        return "amharic_zefen_completion"

class AmharicZefenGenerationDatasource(AmharicZefenDatasource):

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["input"]
    
    def get_datasource_name(self):
        return "amharic_zefen_generation"


class AmharicStoryGenerationDatasource(PrivateDatasource):
    def get_dataset_location(self):
        return "israel/AmharicStoryGeneration"

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["title"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["body"]
    
    def get_datasource_name(self):
        return "amharic_story_generation"

class AmharicMezmurCompletionDatasource(PrivateDatasource):

    
    def get_dataset_location(self):
        return "israel/MezmurCompletion"

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["source"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["target"]
    
    def get_datasource_name(self):
        return "amharic_mezmur_completion"

class AmharicMezmurGenerationDatasource(PrivateDatasource):

    
    def get_dataset_location(self):
        return "israel/MezmurGeneration"

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["title"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["lyrics"]
    
    def get_datasource_name(self):
        return "amharic_mezmur_generation"
    
class AmharicEnglishMTDatasource(PrivateDatasource):
    def get_dataset_location(self):
        return "israel/AmharicMT"

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["amh"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["eng"]
    
    def get_datasource_name(self):
        return "amharic_mt amh-eng"
    
class EnglishAmharicMTDatasource(AmharicEnglishMTDatasource):
    def get_dataset_location(self):
        return "israel/AmharicMT"

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["eng"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["amh"]
    def get_datasource_name(self):
        return "amharic_mt eng-amh"

class AmharicNewsTitleGenerationDatasource(PrivateDatasource):

    def get_dataset_location(self):
        return "israel/TadesDataset"

    def get_prompt_inputs(self,  item: Dict[str, Any]) -> str:
        return item["news"]

    def get_prompt_output(self,  item: Dict[str, Any]) -> str:
        return item["topic"]
    
    def get_datasource_name(self):
        return "amharic_title_generation"