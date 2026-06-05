from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        unique_chars = sorted(list(set(text)))
        
        # Create string-to-int mapping
        stoi = {ch: i for i, ch in enumerate(unique_chars)}
        
        # Create int-to-string reverse mapping
        itos = {i: ch for i, ch in enumerate(unique_chars)}
        
        return stoi, itos
        pass

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        return [stoi[ch] for ch in text]
        pass

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        return "".join([itos[i] for i in ids])
        pass
