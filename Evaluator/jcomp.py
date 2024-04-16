class JSONComparator:
    def __init__(self, json1, json2):
        self.json1 = json1
        self.json2 = json2

    @staticmethod
    def jaccard_similarity(set1, set2):
        print(set1, set2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0.0

    @staticmethod
    def _get_keys(obj):
        keys = set()
        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.add(key)
                if isinstance(value, dict) or isinstance(value, list):
                    nested_keys = JSONComparator._get_keys(value)
                    keys.update(nested_keys)
        elif isinstance(obj, list):
            for item in obj:
                nested_keys = JSONComparator._get_keys(item)
                keys.update(nested_keys)
        return keys

    @staticmethod
    def _get_string_values(obj):
        string_values = []
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, str):
                    string_values.append(value)
                elif isinstance(value, dict) or isinstance(value, list):
                    nested_values = JSONComparator._get_string_values(value)
                    string_values.extend(nested_values)
        elif isinstance(obj, list):
            for item in obj:
                nested_values = JSONComparator._get_string_values(item)
                string_values.extend(nested_values)
        return string_values

    @staticmethod
    def _get_data_types(obj):
        data_types = set()
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, int):
                    data_types.add("int")
                elif isinstance(value, float):
                    data_types.add("float")
                elif isinstance(value, str):
                    data_types.add("str")
                elif isinstance(value, bool):
                    data_types.add("bool")
                elif isinstance(value, list):
                    data_types.add("list")
                elif isinstance(value, dict):
                    data_types.add("dict")
                else:
                    data_types.add("other")
                    # You can add more data types as needed
        elif isinstance(obj, list):
            for item in obj:
                data_types.update(JSONComparator._get_data_types(item))
        return data_types

    def compare_keys(self):
        keys1 = JSONComparator._get_keys(self.json1)
        keys2 = JSONComparator._get_keys(self.json2)
        keys_similarity = JSONComparator.jaccard_similarity(keys1, keys2)
        return keys_similarity

    def compare_values(self):
        string_json1 = JSONComparator._get_string_values(self.json1)
        string_json2 = JSONComparator._get_string_values(self.json2)
        string_values1 = set(string_json1)
        string_values2 = set(string_json2)
        values_similarity = JSONComparator.jaccard_similarity(string_values1, string_values2)
        return values_similarity

    def compare_data_types(self):
        data_types_json1 = JSONComparator._get_data_types(self.json1)
        data_types_json2 = JSONComparator._get_data_types(self.json2)
        data_types_similarity = JSONComparator.jaccard_similarity(data_types_json1, data_types_json2)
        return data_types_similarity

    def evaluate_json(self):
        keys_similarity = self.compare_keys()
        values_similarity = self.compare_values()
        data_types_similarity = self.compare_data_types()
        Jcomp_Score = 0.4 * keys_similarity + 0.4 * values_similarity + 0.2 * data_types_similarity
        # evaluation = {
        #     "Keys Similarity": keys_similarity,
        #     "Values Similarity": values_similarity,
        #     "Data Types Similarity": data_types_similarity,
        #     "Jcomp Score" : Jcomp_Score
        # }
        return Jcomp_Score
