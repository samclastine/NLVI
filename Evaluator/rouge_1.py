from rouge_score import rouge_scorer

class rouge_1_score:
    def __init__(self, json1, json2):
        self.json1 = json1
        self.json2 = json2

    def flatten_dict(self, d, parent_key='', separator='_'):
        items = {}
        for k, v in d.items():
            new_key = parent_key + separator + k if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, new_key, separator=separator))
            else:
                items[new_key] = v
        return items

    def convert_to_string(self, value):
        if isinstance(value, list):
            return ' '.join(map(str, value))
        return str(value)

    def evaluate_rouge(self):
        # Flatten the JSON objects
        flattened_json1 = self.flatten_dict(self.json1)
        flattened_json2 = self.flatten_dict(self.json2)

        # Convert dictionaries to strings
        str1 = ' '.join(self.convert_to_string(value) for value in flattened_json1.values())
        str2 = ' '.join(self.convert_to_string(value) for value in flattened_json2.values())

        # Calculate Rouge-1 score
        scorer = rouge_scorer.RougeScorer(['rouge1'])
        rouge_scores = scorer.score(str1, str2)

        return rouge_scores['rouge1'].fmeasure