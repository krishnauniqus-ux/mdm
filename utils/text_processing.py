# AdvancedTitleCase class
"""Text processing utilities"""

import pandas as pd


class AdvancedTitleCase:
    SMALL_WORDS = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'en', 'for', 'from', 
                   'how', 'if', 'in', 'neither', 'nor', 'of', 'on', 'onto', 'or', 
                   'per', 'so', 'than', 'that', 'the', 'to', 'until', 'up', 'upon', 
                   'v', 'v.', 'via', 'vs', 'vs.', 'when', 'with', 'without', 'yet'}
    
    ALWAYS_CAP = {'II', 'III', 'IV', 'VI', 'VII', 'VIII', 'IX', 'XI', 'XII',
                  'ID', 'TV', 'OK', 'AI', 'ML', 'API', 'URL', 'HTTP', 'HTTPS',
                  'SQL', 'NoSQL', 'JSON', 'XML', 'HTML', 'CSS', 'UI', 'UX',
                  'CEO', 'CTO', 'CFO', 'COO', 'CMO', 'VP', 'SVP', 'EVP',
                  'PhD', 'MD', 'JD', 'MBA', 'BA', 'BS', 'MA', 'MS', 'RN', 'CPA',
                  'IBM', 'AWS', 'GCP', 'Azure', 'SaaS', 'PaaS', 'IaaS'}
    
    @classmethod
    def convert(cls, text: str, style: str = 'apa') -> str:
        if not text or pd.isna(text):
            return text
        
        words = str(text).split()
        if not words:
            return text
        
        result = []
        
        for i, word in enumerate(words):
            if '-' in word:
                parts = word.split('-')
                processed_parts = [cls._process_word(p, i, len(words), style) for p in parts]
                result.append('-'.join(processed_parts))
            else:
                result.append(cls._process_word(word, i, len(words), style))
        
        return ' '.join(result)
    
    @classmethod
    def _process_word(cls, word: str, index: int, total: int, style: str) -> str:
        word_upper = word.upper()
        word_lower = word.lower()
        
        if word_upper in cls.ALWAYS_CAP or (word.isupper() and 2 <= len(word) <= 5):
            return word_upper
        
        if index == 0 or index == total - 1:
            return word.capitalize()
        
        if style == 'apa':
            if word_lower in cls.SMALL_WORDS:
                return word_lower
            return word.capitalize()
        elif style == 'chicago':
            if word_lower in cls.SMALL_WORDS and len(word) < 4:
                return word_lower
            return word.capitalize()
        elif style == 'sentence':
            if index == 0:
                return word.capitalize()
            return word_lower
        
        return word.capitalize()