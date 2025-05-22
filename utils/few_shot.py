from typing import List
from loguru import logger


class FewShotTrainer:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.max_examples = config.get("max_examples", 5)
        self.examples = []

    def add_example(self, input_text: str, output_text: str):
        if len(self.examples) >= self.max_examples:
            self.examples.pop(0)
        self.examples.append((input_text, output_text))
        logger.info(f"Added few-shot example. Total examples: {len(self.examples)}")

    def build_prompt(self, user_input: str, context: List[str]) -> str:
        """
        Build a prompt with few-shot examples and context for the LLM.
        """
        prompt = ""
        if self.enabled and self.examples:
            for inp, out in self.examples:
                prompt += f"User: {inp}\nBot: {out}\n\n"
        if context:
            prompt += "Context:\n"
            for c in context:
                prompt += f"{c}\n"
            prompt += "\n"
        prompt += f"User: {user_input}\nBot:"
        return prompt
