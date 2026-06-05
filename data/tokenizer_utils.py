from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []
        for num in numbers:
            text = str(num)
            tokens = []
            i = 0
            while i < len(text):
                match_found = False
                for length in range(len(text) - i, 0, -1):
                    substring = text[i:i+length]
                    if substring in vocab:
                        tokens.append(substring)
                        i += length
                        match_found = True
                        break
                
                # If no match is found in the vocabulary, consume a single character
                if not match_found:
                    tokens.append(text[i])
                    i += 1
            result.append(tokens)
            
        return result
        pass

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        token_count = 0
        i = 0
        while i < len(text):
            match_found = False
            for length in range(len(text) - i, 0, -1):
                if text[i:i+length] in vocab:
                    token_count += 1
                    i += length
                    match_found = True
                    break
                    
            if not match_found:
                token_count += 1
                i += 1
                
        return token_count
        pass

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        
        if not words:
            return 0.0
            
        # We can reuse the count_tokens method we just defined above
        total_tokens = self.count_tokens(text, vocab)
        fertility = total_tokens / len(words)
        
        return round(fertility, 4)
        pass
