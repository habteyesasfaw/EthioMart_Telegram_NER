import unittest
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class TestNERModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load model and tokenizer for testing
        cls.model_name = "xlm-roberta-base"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = AutoModelForTokenClassification.from_pretrained(cls.model_name)
        cls.label2id = {"O": 0, "B-PRODUCT": 1, "I-PRODUCT": 2, "B-PRICE": 3, "I-PRICE": 4, "B-LOC": 5, "I-LOC": 6}
        cls.id2label = {v: k for k, v in cls.label2id.items()}

    def test_load_conll_data(self):
        """Test loading CoNLL data."""
        sentences, labels = load_conll_data("../data/merged_amharic_ner_data.conll")
        self.assertIsInstance(sentences, list)
        self.assertIsInstance(labels, list)
        self.assertGreater(len(sentences), 0, "No sentences found.")
        self.assertGreater(len(labels), 0, "No labels found.")

    def test_tokenization(self):
        """Test tokenization of input sentences."""
        sample_sentence = ["ምርቶች", "ከአዲስ", "አበባ", "በቅናሽ", "ዋጋ", "ይሰጣሉ።"]
        tokenized_inputs = self.tokenizer(sample_sentence, truncation=True, is_split_into_words=True)
        self.assertIn("input_ids", tokenized_inputs)
        self.assertIn("attention_mask", tokenized_inputs)



    def test_prediction(self):
        """Test if the model can make a prediction."""
        sample_text = "ምርቶች ከአዲስ አበባ በቅናሽ ዋጋ ይሰጣሉ።"
        inputs = self.tokenizer(sample_text, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        self.assertIsInstance(predictions, torch.Tensor)
        self.assertEqual(predictions.shape[0], 1)



if __name__ == "__main__":
    unittest.main()
