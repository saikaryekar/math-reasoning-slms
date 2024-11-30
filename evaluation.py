import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Initialize evaluation metrics
rouge_metric = load("rouge")

class Evaluation:
    def __init__(self, model, tokenizer, device='cuda'):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.qa_pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if device == 'cuda' else -1)
        self.correct_predictions = 0

    @staticmethod
    def extract_final_answer(answer_text):
        # Extract final answer from text using regex
        match = re.findall(r'####\s*(-?\d+)', answer_text)
        return int(match[-1]) if match else None

    def calculate_accuracy(self, actual_answers, predicted_answers):
        correct_predictions = sum(
            1 for actual, predicted in zip(actual_answers, predicted_answers)
            if self.extract_final_answer(actual) == self.extract_final_answer(predicted)
        )
        return (correct_predictions / len(actual_answers)) * 100
    
    def evaluate_rouge(self, eval_data):
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

    def breakdown_question(self, question):
        few_shot_examples = [
            "Example 1: Question: What is 5 + 7? Breakdown: We start by adding the numbers 5 and 7. The result is 12.",
            "Example 2: Question: If I have 20 apples and give away 6, how many apples do I have left? Breakdown: We subtract 6 from 20. The result is 14 apples.",
            "Example 3: Question: A movie lasts 2 hours and 30 minutes. If it starts at 3:00 PM, what time does it end? Breakdown: First, add 2 hours to 3:00 PM. This gives 5:00 PM. Then, add 30 minutes. The result is 5:30 PM.",
            "Example 4: Question: A box contains 50 red balls, 30 blue balls, and 20 green balls. What fraction of the balls are blue? Breakdown: First, calculate the total number of balls: 50 + 30 + 20 = 100. The fraction of blue balls is 30/100, which simplifies to 3/10.",
            # "Example 5: Question: A farmer has 5 cows. Each cow produces 7 liters of milk per day. How much milk does the farmer collect in a week? Breakdown: First, multiply 7 liters by 5 cows to get the daily milk production: 7 * 5 = 35 liters per day. Then, multiply 35 liters by 7 days to get the weekly total: 35 * 7 = 245 liters.",
            # "Example 6: Question: How many minutes are in 4 hours and 15 minutes? Breakdown: First, calculate the number of minutes in 4 hours: 4 * 60 = 240 minutes. Then, add the extra 15 minutes: 240 + 15 = 255 minutes.",
            # "Example 7: Question: If a train travels at 60 miles per hour for 3 hours, how far does it travel? Breakdown: Multiply the speed (60 miles per hour) by the time (3 hours): 60 * 3 = 180 miles.",
            # "Example 8: Question: A rectangle has a length of 10 cm and a width of 4 cm. What is the area of the rectangle? Breakdown: The area of a rectangle is calculated by multiplying length and width: 10 * 4 = 40 square centimeters."
        ]

        prompt = "\n".join(few_shot_examples) + f"\n\nNow, for the question: {question}, let's break it down step by step."
        return prompt
    
    def generate_answer_with_reasoning(self, broken_down_question):
        prompt = f"Given the breakdown of the question: {broken_down_question}, provide the step-by-step solution."
        prediction = self.qa_pipeline(prompt, max_length=256)[0]["generated_text"]
        return prediction

    def evaluate_with_cot(self, eval_data):
        correct_predictions = 0
        num_samples = len(eval_data)
        
        # Step 1: Convert evaluation data into a dataset object
        eval_dataset = Dataset.from_dict({
            "question": [item["question"] for item in eval_data],
            "answer": [item["answer"] for item in eval_data]
        })

        # Step 2: Use DataLoader for efficient batching
        dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

        all_predicted_answers = []
        all_actual_answers = eval_dataset["answer"]

        for batch in dataloader:
            # Move questions to GPU for faster processing
            batch_questions = batch["question"]

            for question in batch_questions:
                # Step 3: Break down the question
                broken_down_question = self.breakdown_question(question)

                # Step 4: Generate answer with reasoning based on the breakdown
                prediction = self.generate_answer_with_reasoning(broken_down_question)

                # Append the prediction to the list
                all_predicted_answers.append(prediction)
        
        # Show 3-4 example questions, answers, and predictions
        for i in range(min(4, len(all_actual_answers))):
            print(f"Question {i+1}: {eval_dataset['question'][i]}")
            print(f"Correct Answer: {all_actual_answers[i]}")
            print(f"Predicted Answer: {all_predicted_answers[i]}\n")

        # Calculate accuracy
        accuracy = self.calculate_accuracy(all_actual_answers, all_predicted_answers)
        return accuracy

    # Main function for GPU-optimized evaluation
    def evaluate_accuracy(self, eval_data, model, tokenizer, batch_size=16, device='cuda'):
        correct_predictions = 0
        num_samples = len(eval_data)
        
        # Step 1: Convert evaluation data into a dataset object
        eval_dataset = Dataset.from_dict({
            "question": [item["question"] for item in eval_data],
            "answer": [item["answer"] for item in eval_data]
        })

        # Step 2: Use DataLoader for efficient batching
        dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
        
        all_predicted_answers = []
        all_actual_answers = eval_data["answer"]

        for batch in dataloader:
            batch_questions = batch["question"]

            predictions = self.qa_pipeline(batch_questions, max_length=256)
            
            batch_predicted_answers = [pred["generated_text"] for pred in predictions]
            all_predicted_answers.extend(batch_predicted_answers)

        accuracy = self.calculate_accuracy(all_actual_answers, all_predicted_answers)
        # print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    # Function to extract embeddings from T5
    def extract_t5_embeddings(self, text):
        """
        Extract embeddings from the T5 model.
        """
        # Tokenize input text and get tensor input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.model.device)

        # Get hidden states (outputs from the last encoder layer)
        with torch.no_grad():
            outputs = self.model.encoder(input_ids=input_ids)
            # Get embeddings (hidden states of the last encoder layer)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Average across all token embeddings
        
        return embeddings
    
    # Function to calculate semantic similarity using T5 embeddings
    def calculate_semantic_similarity_t5(self, predicted_steps, ground_truth_steps):
        """
        Calculate semantic similarity using T5 embeddings.
        """
        predicted_embedding = self.extract_t5_embeddings(predicted_steps)
        ground_truth_embedding = self.extract_t5_embeddings(ground_truth_steps)
        
        # Compute cosine similarity
        similarity = cosine_similarity(predicted_embedding.cpu().numpy(), ground_truth_embedding.cpu().numpy())
        return similarity[0][0]
    
    # Function to evaluate semantic similarity on the entire dataset
    def evaluate_semantic_similarity(self, eval_data):
        """
        Evaluate semantic similarity between generated texts and reference answers for the entire dataset.
        
        Args:
            model (T5 model): Your fine-tuned T5 model.
            tokenizer (T5 tokenizer): The tokenizer for your fine-tuned T5 model.
            eval_data (list): A list of examples with 'question' and 'answer' keys.
        
        Returns:
            dict: A dictionary containing average semantic similarity score.
        """
        generated_texts = []
        references = []
        
        for example in eval_data:
            input_text = example['question']
            reference_answer = example["answer"]
            
            # Tokenize and generate text using the fine-tuned model
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            outputs =self.model.generate(**inputs)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            generated_texts.append(generated_text)
            references.append(reference_answer)

        # Compute semantic similarity for the entire dataset
        similarities = [self.calculate_semantic_similarity_t5(pred, ref) for pred, ref in zip(generated_texts, references)]

        # Calculate the overall semantic similarity metric (average in this case)
        overall_semantic_similarity = sum(similarities) / len(similarities)

        return overall_semantic_similarity