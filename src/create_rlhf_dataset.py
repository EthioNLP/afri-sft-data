from afri_rlhf.data.sources import *
from afri_rlhf.prompt.templates import get_prompt_template_by_id
from afri_rlhf.utils.language import get_language_by_iso_code
from datasets import concatenate_datasets

def get_dataset(split, prompt_id, datasource_class, **kwargs):
    prompt = get_prompt_template_by_id(prompt_id)
    language = get_language_by_iso_code("amh")
    source = datasource_class(language=language, split = split, prompt=prompt, **kwargs)
    dataset = source.load_dataset(apply_formatting=True).remove_columns(["prompt_header", "datasource", "prompt"])

    return dataset



def main():
    import json
    training_datasets = concatenate_datasets([
        get_dataset("train", "id001", AfriSentDatasource),
        get_dataset("train", "id005", MasakhaNewsClassificationDatasource),
        get_dataset("train", "id002", CCAlignedDatasource, source_type="sentences", transilate_to_english=True),
        get_dataset("train", "id003", XlsumDatasource),
        get_dataset("train", "id006", MasakhaNERDatasource, entity_to_extract= "PER", empty_entities_output = "ስም አልተገኘም", use_v2=False)
    ])

    training_dict = [item for item in training_datasets]
    with open("../logs/datasets/train.json", "w") as output_file:
        json.dump(training_dict, output_file)

    validation_datasets = concatenate_datasets([
        get_dataset("validation", "id001", AfriSentDatasource),
        get_dataset("validation", "id005", MasakhaNewsClassificationDatasource),
        get_dataset("validation", "id003", XlsumDatasource),
        get_dataset("validation", "id006", MasakhaNERDatasource, entity_to_extract= "PER", empty_entities_output = "ስም አልተገኘም", use_v2=False)
    ])

    validation_dict = [item for item in validation_datasets]
    with open("../logs/datasets/validation.json", "w") as output_file:
        json.dump(validation_dict, output_file)


    test_datasets = concatenate_datasets([
        get_dataset("test", "id001", AfriSentDatasource),
        get_dataset("test", "id005", MasakhaNewsClassificationDatasource),
        get_dataset("test", "id003", XlsumDatasource),
        get_dataset("test", "id006", MasakhaNERDatasource, entity_to_extract= "PER", empty_entities_output = "ስም አልተገኘም", use_v2=False)
    ])


    test_dict = [item for item in test_datasets]
    with open("../logs/datasets/test.json", "w") as output_file:
        json.dump(test_dict, output_file)

    


if __name__ == "__main__":
    main()