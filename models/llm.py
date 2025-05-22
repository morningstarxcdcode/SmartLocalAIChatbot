from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLLM:
    def __init__(self, model_name: str, model_path: str = None):
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading LLaMA2 model {self.model_name}...")
        # Load tokenizer and model from HuggingFace hub or local path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_length: int = 100, response_type: str = "default") -> str:
        print(f"Generating response for prompt: {prompt} with response_type: {response_type}")
        style_instructions = {
            "default": "",
            "friendly": " Respond in a friendly and warm tone.",
            "formal": " Respond in a formal and professional manner.",
            "humorous": " Respond with a touch of humor.",
            "concise": " Respond concisely and to the point.",
            "detailed": " Provide a detailed and thorough response."
        }
        instruction = style_instructions.get(response_type, "")
        full_prompt = prompt + instruction
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                num_return_sequences=1,
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the full_prompt from the generated text to get only the response
        response = generated_text[len(full_prompt):].strip()
        return response if response else "Sorry, I could not generate a response."
