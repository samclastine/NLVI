from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Bleu_1_score:
    def __init__(self, json1, json2):
        self.json1 = json1
        self.json2 = json2

    @staticmethod
    def _flatten_json(obj, separator='_'):
        flattened = {}

        def recurse(current, key=''):
            if isinstance(current, dict):
                for k, v in current.items():
                    new_key = key + separator + k if key else k
                    recurse(v, new_key)
            elif isinstance(current, list):
                for i, v in enumerate(current):
                    new_key = key + separator + str(i) if key else str(i)
                    recurse(v, new_key)
            else:
                flattened[key] = current

        recurse(obj)
        return flattened

    def tokenize_and_flatten(self, obj):
        flattened_obj = self._flatten_json(obj)
        # Tokenize the flattened values
        tokenized_values = [str(value).split() for value in flattened_obj.values()]
        return [item for sublist in tokenized_values for item in sublist]

    def evaluate_bleu(self):
        tokenized_values1 = self.tokenize_and_flatten(self.json1)
        tokenized_values2 = self.tokenize_and_flatten(self.json2)

        # Calculate BLEU-1 score with smoothing function
        smoothing_function = SmoothingFunction().method1  # Laplace smoothing
        bleu_score = sentence_bleu([tokenized_values1], tokenized_values2, smoothing_function=smoothing_function, weights=(1.0, 0, 0, 0))

        return bleu_score
