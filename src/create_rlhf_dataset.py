from afri_rlhf.data.sources import *
from afri_rlhf.prompt.templates import get_prompt_template_by_id
from afri_rlhf.utils.language import get_language_by_iso_code
import IPython
def main():
    label_mapping = {
        0: "አዎንታዊ",
        1: "ገለልተኛ",
        2: "አሉታዊ"
    }
    prompt = get_prompt_template_by_id("id001")
    language = get_language_by_iso_code("amh")
    source = AfriSentDatasource(language=language, split = "train", prompt=prompt)
    dataset = source.load_dataset()

    print("Sentiment dataset examples")
    for _, item in zip(range(5), dataset):
        print(item["prompt"])
        print("##" * 20)

    print("__" * 20)
    print("##")
    prompt = get_prompt_template_by_id("id002")
    source = CCAlignedDatasource(language=language, split="train", prompt=prompt, source_type="sentences", transilate_to_english=True)
    dataset = source.load_dataset()


    print("Transilation dataset examples")
    for _, item in zip(range(5), dataset):
        print(item["prompt"])
        print("##" * 20)

    
    IPython.embed()


if __name__ == "__main__":
    main()