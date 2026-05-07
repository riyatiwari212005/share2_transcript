import json
import os
from pathlib import Path
from typing import List, Dict
import logging
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataPreparator:
    def __init__(self, transcript_dir: str = None, output_dir: str = None):
        self.transcript_dir = transcript_dir or settings.TRANSCRIPT_DIR
        self.output_dir = output_dir or settings.TRAINING_DATA_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_transcripts(self) -> List[Dict]:
        transcript_files = list(Path(self.transcript_dir).glob("*.json"))
        transcript_files = [f for f in transcript_files if f.stem != "transcription_summary"]
        
        transcripts = []
        for file_path in transcript_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transcripts.append(data)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(transcripts)} transcripts")
        return transcripts
    
    def create_conversation_pairs(self, transcripts: List[Dict]) -> List[Dict]:
        training_data = []
        
        for transcript in transcripts:
            full_text = transcript["text"]
            segments = transcript["segments"]
            
            for i in range(len(segments) - 1):
                current_segment = segments[i]["text"].strip()
                next_segment = segments[i + 1]["text"].strip()
                
                if len(current_segment) > 10 and len(next_segment) > 10:
                    training_data.append({
                        "instruction": "You are an AI assistant trained on specific video content. Respond in the same tone and style as the training material. Only answer questions related to the topics covered in your training data.",
                        "input": current_segment,
                        "output": next_segment,
                        "source": transcript["video_name"]
                    })
            
            chunk_size = 3
            for i in range(0, len(segments) - chunk_size, chunk_size):
                context = " ".join([s["text"].strip() for s in segments[i:i+chunk_size]])
                question = f"What was discussed about this topic?"
                answer = " ".join([s["text"].strip() for s in segments[i:i+chunk_size+2]])
                
                if len(context) > 20 and len(answer) > 20:
                    training_data.append({
                        "instruction": "You are an AI assistant trained on specific video content. Respond in the same tone and style as the training material. Only answer questions related to the topics covered in your training data.",
                        "input": question + "\n\nContext: " + context,
                        "output": answer,
                        "source": transcript["video_name"]
                    })
        
        logger.info(f"Created {len(training_data)} training examples")
        return training_data
    
    def create_gemma_format(self, training_data: List[Dict]) -> List[Dict]:
        formatted_data = []
        
        for item in training_data:
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": item["input"]
                    },
                    {
                        "role": "assistant",
                        "content": item["output"]
                    }
                ],
                "system": item["instruction"],
                "source": item["source"]
            }
            formatted_data.append(conversation)
        
        return formatted_data
    
    def save_training_data(self, training_data: List[Dict], filename: str = "training_data.jsonl"):
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Training data saved to {output_path}")
        
        stats_path = os.path.join(self.output_dir, "training_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_examples": len(training_data),
                "sources": list(set([item["source"] for item in training_data]))
            }, f, indent=2)
        
        return output_path
    
    def prepare(self):
        transcripts = self.load_transcripts()
        
        if not transcripts:
            logger.warning("No transcripts found. Please run transcribe.py first.")
            return None
        
        training_pairs = self.create_conversation_pairs(transcripts)
        formatted_data = self.create_gemma_format(training_pairs)
        
        output_path = self.save_training_data(formatted_data)
        
        return output_path

def main():
    preparator = TrainingDataPreparator()
    output_path = preparator.prepare()
    
    if output_path:
        logger.info(f"Training data preparation complete: {output_path}")
    else:
        logger.error("Training data preparation failed")

if __name__ == "__main__":
    main()
