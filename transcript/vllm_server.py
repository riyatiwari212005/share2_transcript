from vllm import LLM, SamplingParams
from huggingface_hub import login
from typing import List, Dict, Optional
import logging
import os
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def login_to_huggingface():
    """Login to Hugging Face Hub using the token from environment."""
    token = settings.HUGGING_FACE_TOKEN
    if not token or token == "your_huggingface_token_here":
        logger.warning(
            "No HUGGING_FACE_TOKEN set in .env file. "
            "Gemma models require authentication."
        )
        return False
    try:
        login(token=token)
        logger.info("Successfully logged in to Hugging Face Hub")
        return True
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face: {e}")
        return False


class VLLMServer:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.llm = None
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["<end_of_turn>"]
        )
        
    def load_model(self):
        login_to_huggingface()
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.llm = LLM(
                model=self.model_path,
                gpu_memory_utilization=settings.VLLM_GPU_MEMORY_UTILIZATION,
                max_model_len=settings.VLLM_MAX_MODEL_LEN,
                trust_remote_code=True
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            logger.info("Falling back to base model from Hugging Face")
            login_to_huggingface()
            self.llm = LLM(
                model=settings.MODEL_NAME,
                gpu_memory_utilization=settings.VLLM_GPU_MEMORY_UTILIZATION,
                max_model_len=settings.VLLM_MAX_MODEL_LEN,
                trust_remote_code=True
            )
    
    def format_prompt(self, user_message: str, context: str = "", system_message: str = None) -> str:
        if system_message is None:
            system_message = "You are an AI assistant trained on specific video content. Respond in the same tone and style as the training material. Only answer questions related to the topics covered in your training data."
        
        prompt = f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
        
        if context:
            prompt += f"<start_of_turn>user\nContext from training videos:\n{context}\n\nQuestion: {user_message}<end_of_turn>\n"
        else:
            prompt += f"<start_of_turn>user\n{user_message}<end_of_turn>\n"
        
        prompt += "<start_of_turn>model\n"
        
        return prompt
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stop=["<end_of_turn>", "<start_of_turn>"]
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        
        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text.strip()
        
        return ""
    
    def chat(self, user_message: str, context: str = "", temperature: float = 0.7) -> str:
        prompt = self.format_prompt(user_message, context)
        response = self.generate(prompt, temperature=temperature)
        return response

if __name__ == "__main__":
    server = VLLMServer()
    server.load_model()
    
    test_message = "Hello, can you help me?"
    response = server.chat(test_message)
    logger.info(f"Test response: {response}")
