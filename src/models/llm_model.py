import torch
from jsonformer import Jsonformer
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM:
    def __init__(self, llm_cfg):
        self.llm_cfg = llm_cfg
        self.tokenizer = AutoTokenizer.from_pretrained(llm_cfg.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_cfg.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )

    def get_prompt(self, top_words: str, text: str) -> str:
        prompt = f"""
            These are the keywords of a group of scientific articles: {top_words}.

            This is an example of the abstract of one article in this group:
            {text}

            What are the high- and low-level topics of this group of articles?

            You must fill:
            - hight_level_topic (string)
            - low_level_topic (string)

            hight_level_topic is a general scientific discipline like Mathematics, Physics, Medicide and so on.
            low_level_topic is a specific subtopic of the defined scientific discipline.

            Return ONLY the structured data.
        """
        return prompt

    def get_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "hight_level_topic": {"type": "string"},
                "low_level_topic": {"type": "string"},
            },
            "required": ["hight_level_topic", "low_level_topic"],
        }
        return schema

    def run_model(self, top_words: str, text: str) -> str:
        prompt = self.get_prompt(top_words, text)
        schema = self.get_schema()

        generator = Jsonformer(
            model=self.model,
            tokenizer=self.tokenizer,
            json_schema=schema,
            prompt=prompt,
            max_array_length=10,
            max_number_tokens=5,
        )

        result = generator()
        return result
