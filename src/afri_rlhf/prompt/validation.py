from afri_rlhf.utils.support import supported_languages, supported_task_types


def validate_supported_languages(prompts):
    for _, prompt in prompts.items():
        assert prompt["prompt_language"] in supported_languages
def validate_supported_task_types(prompts):
    for _, prompt in prompts.items():
        assert prompt["task_type"] in supported_task_types 