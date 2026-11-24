import re


_ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'st', 'a.m', 'p.m', 'e.g', 'i.e'}


class ClauseChunker:
    def __init__(self, first_min_chars: int, min_chars: int):
        self.first_min_chars = first_min_chars
        self.min_chars = min_chars
        self.buffer = ''
        self.emitted = False

    def push(self, text: str) -> list[str]:
        self.buffer += text
        result = []
        start = 0
        for i, char in enumerate(self.buffer):
            if char not in '.!?,;:—' or i + 1 >= len(self.buffer) or not self.buffer[i + 1].isspace():
                continue
            candidate = self.buffer[start:i + 1]
            if char == '.':
                word = candidate.rstrip()[:-1].split()[-1].lower() if candidate[:-1].split() else ''
                if word in _ABBREVIATIONS:
                    continue
            if char in ',;:—' and len(candidate.strip()) < (self.min_chars if self.emitted else self.first_min_chars):
                continue
            clean = re.sub(r'[*#`]', '', candidate).strip()
            start = i + 1
            if clean:
                result.append(clean)
                self.emitted = True
        self.buffer = self.buffer[start:]
        return result

    def flush(self) -> str | None:
        clean = re.sub(r'[*#`]', '', self.buffer).strip()
        self.buffer = ''
        if clean:
            self.emitted = True
        return clean or None
