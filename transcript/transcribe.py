import whisper
import os
import json
from pathlib import Path
from typing import Dict, List
import logging
from moviepy.editor import VideoFileClip
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoTranscriber:
    def __init__(self, model_size: str = None):
        self.model_size = model_size or settings.WHISPER_MODEL
        logger.info(f"Loading Whisper model: {self.model_size}")
        self.model = whisper.load_model(self.model_size)
        
    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            return True
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {e}")
            return False
    
    def transcribe_video(self, video_path: str, output_dir: str = None) -> Dict:
        output_dir = output_dir or settings.TRANSCRIPT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = Path(video_path).stem
        audio_path = os.path.join(output_dir, f"{video_name}_temp.wav")
        transcript_path = os.path.join(output_dir, f"{video_name}.json")
        text_path = os.path.join(output_dir, f"{video_name}.txt")
        
        logger.info(f"Processing video: {video_path}")
        
        if not self.extract_audio(video_path, audio_path):
            return None
        
        try:
            result = self.model.transcribe(
                audio_path,
                task="transcribe",
                verbose=False,
                word_timestamps=True,
                fp16=False
            )
            
            transcript_data = {
                "video_name": video_name,
                "video_path": video_path,
                "language": result.get("language", "unknown"),
                "text": result["text"],
                "segments": []
            }
            
            for segment in result["segments"]:
                transcript_data["segments"].append({
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": segment.get("words", [])
                })
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            logger.info(f"Transcription saved to {transcript_path}")
            return transcript_data
            
        except Exception as e:
            logger.error(f"Error transcribing {video_path}: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None
    
    def transcribe_all_videos(self, video_dir: str = None, output_dir: str = None) -> List[Dict]:
        video_dir = video_dir or settings.VIDEO_DIR
        output_dir = output_dir or settings.TRANSCRIPT_DIR
        
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(video_dir).glob(f"*{ext}"))
            video_files.extend(Path(video_dir).glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(video_files)} video files")
        
        transcripts = []
        for video_file in video_files:
            transcript = self.transcribe_video(str(video_file), output_dir)
            if transcript:
                transcripts.append(transcript)
        
        return transcripts

def main():
    transcriber = VideoTranscriber()
    
    os.makedirs(settings.VIDEO_DIR, exist_ok=True)
    os.makedirs(settings.TRANSCRIPT_DIR, exist_ok=True)
    
    transcripts = transcriber.transcribe_all_videos()
    
    logger.info(f"Transcription complete. Processed {len(transcripts)} videos.")
    
    summary_path = os.path.join(settings.TRANSCRIPT_DIR, "transcription_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_videos": len(transcripts),
            "transcripts": [t["video_name"] for t in transcripts]
        }, f, indent=2)

if __name__ == "__main__":
    main()
