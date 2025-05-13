import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class HybridQAModel:  # Corrected class name
    def __init__(self, t5_model_name="t5-small"):
        self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    def generate_answer(self, question, context, max_len=100):
        input_text = f"question: {question} context: {context}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
       
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
            self.model.to('cuda') # Ensure model is also on GPU

        outputs = self.model.generate(input_ids, max_length=max_len)
       
        if torch.cuda.is_available():
             outputs = outputs.cpu()

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True) 