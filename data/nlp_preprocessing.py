import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_sentences = positive + negative
        
        # 2. Build vocabulary: collect all unique words
        unique_words = set()
        for sentence in all_sentences:
            # .update() is a slightly cleaner way to add multiple elements to a set
            unique_words.update(sentence.split())
                
        # Sort them alphabetically and assign integer IDs starting at 1
        sorted_words = sorted(list(unique_words))
        word_to_id = {word: i + 1 for i, word in enumerate(sorted_words)}
        
        # 3. Encode each sentence by replacing words with their IDs
        encoded_tensors = []
        for sentence in all_sentences:
            encoded_sentence = [float(word_to_id[word]) for word in sentence.split()]
            # Convert each encoded list into a 1D float tensor
            encoded_tensors.append(torch.tensor(encoded_sentence, dtype=torch.float32))
            
        # 4. Pad shorter sequences with 0s to make a rectangular tensor
        # Using the exact PyTorch module and arguments recommended in the hint
        padded_dataset = nn.utils.rnn.pad_sequence(
            encoded_tensors, 
            batch_first=True, 
            padding_value=0.0
        )
        
        return padded_dataset
        pass
