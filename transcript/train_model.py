import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from huggingface_hub import login, hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import os
import logging
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def login_to_huggingface():
    """Login to Hugging Face Hub using the token from environment."""
    token = settings.HUGGING_FACE_TOKEN
    if not token or token == "your_huggingface_token_here":
        logger.warning(
            "No HUGGING_FACE_TOKEN set in .env file. "
            "Gemma models require authentication. "
            "Please set your token: https://huggingface.co/settings/tokens"
        )
        return False
    try:
        login(token=token)
        logger.info("Successfully logged in to Hugging Face Hub")
        return True
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face: {e}")
        return False

class GemmaFineTuner:
    def __init__(self, model_name: str = None, output_dir: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.output_dir = output_dir or settings.MODEL_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        
    def load_model_and_tokenizer(self):
        login_to_huggingface()
        logger.info(f"Loading model: {self.model_name}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_model_for_training(self):
        logger.info("Preparing model for training with LoRA")
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def format_instruction(self, example):
        if "messages" in example:
            messages = example["messages"]
            system = example.get("system", "")
            
            formatted_text = f"<start_of_turn>system\n{system}<end_of_turn>\n"
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    formatted_text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                elif role == "assistant":
                    formatted_text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            
            return {"text": formatted_text}
        
        return {"text": ""}
    
    def train(self, training_data_path: str, epochs: int = 3, batch_size: int = 4):
        logger.info(f"Loading training data from {training_data_path}")
        
        dataset = load_dataset("json", data_files=training_data_path, split="train")
        dataset = dataset.map(self.format_instruction, remove_columns=dataset.column_names)
        
        logger.info(f"Dataset size: {len(dataset)}")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            save_strategy="epoch",
            logging_steps=10,
            warmup_steps=100,
            optim="paged_adamw_8bit",
            save_total_limit=2,
            report_to="none"
        )
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=512,
            dataset_text_field="text",
            packing=False
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training complete!")

def main():
    training_data_path = os.path.join(settings.TRAINING_DATA_DIR, "training_data.jsonl")
    
    if not os.path.exists(training_data_path):
        logger.error(f"Training data not found at {training_data_path}")
        logger.error("Please run prepare_training_data.py first")
        return
    
    fine_tuner = GemmaFineTuner()
    fine_tuner.load_model_and_tokenizer()
    fine_tuner.prepare_model_for_training()
    fine_tuner.train(training_data_path)

if __name__ == "__main__":
    main()
