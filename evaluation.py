import re
from transformers import pipeline
from evaluate import load

# Initialize evaluation metrics
rouge_metric = load("rouge")
class Evaluation:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.correct_predictions = 0

    @staticmethod
    def extract_final_answer(answer_text):
        # Add the logic to extract the final answer from the answer text
        # Example: assuming answers are at the end of the text after '####'
        try:
            match = re.findall(r'####\s*(-?\d+)', text)
            if match:
                return int(match[-1])  # Convert the last matched number integer
            return None  # Return None if no answer was found
        except IndexError:
            return None

    def evaluate_sample(self, actual_answer, predicted_answer):
        # Extract the answers
        actual_final_answer = self.extract_final_answer(actual_answer)
        predicted_final_answer = self.extract_final_answer(predicted_answer)

        # Check if they match
        return actual_final_answer == predicted_final_answer

    def update_accuracy(self, actual_answer, predicted_answer):
        # Update correct predictions if the answer is correct
        if self.evaluate_sample(actual_answer, predicted_answer):
            self.correct_predictions += 1

    def calculate_accuracy(self, total_samples):
        return (self.correct_predictions / total_samples) * 100
    
    def evaluate_rouge(self, eval_data):
        # Evaluate model on ROUGE metrics
        generated_texts = []
        references = []

        for example in eval_data:
            input_text = example['question']
            reference_answer = example["answer"]
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            generated_texts.append(generated_text)
            references.append(reference_answer)

        # Compute ROUGE
        result = rouge_metric.compute(predictions=generated_texts, references=references)
        return result
