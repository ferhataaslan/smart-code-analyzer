class FakeTokenizer:
    def encode(self, c, add_special_tokens=False):
        return [ord(c)]

class TestDataset:
    def __init__(self):
        self._max_length = 512
        self._overlap = 64
        self._tokenizer = FakeTokenizer()
        
    def _sliding_window_chunk(self, ids: list[int]) -> list[list[int]]:
        if len(ids) <= self._max_length:
            return [ids]

        if getattr(self, "_split_ids", None) is None:
            split_chars = ["}", ";", "\n"]
            split_ids = set()
            for c in split_chars:
                encoded = self._tokenizer.encode(c, add_special_tokens=False)
                if encoded:
                    split_ids.add(encoded[0])
            self._split_ids = split_ids

        chunks: list[list[int]] = []
        start = 0
        while start < len(ids):
            end = min(start + self._max_length, len(ids))

            if end < len(ids):
                best = end
                search_from = max(end - 64, start)
                for i in range(end - 1, search_from, -1):
                    if ids[i] in self._split_ids:
                        best = i + 1
                        break
                end = best

            chunks.append(ids[start:end])
            start = end - self._overlap
        return chunks

d = TestDataset()
ids = [0] * 1000 # No split tokens
chunks = d._sliding_window_chunk(ids)
print("No split token chunks:", len(chunks))

ids = [0] * 1000
ids[500] = ord(';')
chunks = d._sliding_window_chunk(ids)
print("With split token chunks:", len(chunks))
print("Test finished successfully!")
