from typing import Any, Dict
from .support import supported_languages

class Language:
    def __init__(self, iso_code: str, *, iso_short_code: str, locale_code: str, english_name: str, self_name: str):
        self.iso_code = iso_code
        self.iso_short_code = iso_short_code
        self.english_name = english_name
        self.self_name = self_name
        self.locale_code = locale_code

    def __str__(self):
        return "Language(iso_code = {self.iso_code} iso_short_code= {self.iso_short_code}, locale_code = {self.locale_code}, english_name= {self.english_name}, self_name= {self.self_name})"
    def __repr__(self) -> str:
        return str(self)
    def __hash__(self) -> int:
        return (self.iso_code, self.iso_short_code, self.locale_code, self.english_name, self.self_name)


def parse_language_json(iso_code: str, **content: Dict[str, Any]) -> Language:
    return Language(iso_code, **content)

def get_language_by_iso_code(code) -> Language:
    for iso_code, language_content in supported_languages.items():
        if iso_code == code:
            return parse_language_json(iso_code, **language_content)
    return None