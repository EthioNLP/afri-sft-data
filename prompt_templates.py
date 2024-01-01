sentiment_classification_prompt_templates = {
    "amh":"""
        የተሰጠው ጽሑፍ አስተያየት ምን አይነት ነው? "አዎንታዊ"፣ "አሉታዊ" ወይም "ገለልተኛ" የምል ምላሽ ስጥ።
        ### ጽሑፍ: {text} ### አስተያየት: {sentiment}
        """
}


machine_translation_prompt_templates = {
    "አማርኛ": """
        የተሰጠውን ጽሁፍ ### ከ{source_language} ቋንቋ ### ወደ: {target_language} ተርጉም።
        ### የመነሻ ቋንቋ ጽሁፍ: {source_text} ### ትርጉም: {translation}"""
}

summarization_prompt_templates = {
    "am":"""
        ለተሰጠው ጽሁፍ አጭር ማጠቃልያ ስጥ። 
        ### ጽሑፍ: {text} ### ማጠቃልያ: {summary}
        """
}

masakhane_ner_prompt_templates = {
    "amh": {
        ""
    }
}



masakhane_pos_prompt_templates = {
    "amh": {

    }
}