import os
import argparse
import logging
from huggingface_hub import login, snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
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


def download_gemma_model(model_name: str = None, cache_dir: str = None):
    """Download Gemma model and tokenizer from Hugging Face.
    
    Args:
        model_name: Hugging Face model ID (default from settings)
        cache_dir: Local directory to save the model (default from settings)
    """
    model_name = model_name or settings.MODEL_NAME
    cache_dir = cache_dir or settings.MODEL_PATH
    
    logger.info("=" * 60)
    logger.info("Hugging Face Model Downloader")
    logger.info("=" * 60)
    
    # Step 1: Login
    if not login_to_huggingface():
        logger.error("Authentication required. Cannot proceed with download.")
        logger.error("\nTo get a token:")
        logger.error("1. Go to https://huggingface.co/settings/tokens")
        logger.error("2. Create a new token (read access)")
        logger.error("3. Add it to your .env file as HUGGING_FACE_TOKEN=your_token")
        return False
    
    # Step 2: Accept model license
    logger.info("\nNOTE: You must accept the Gemma model license on Hugging Face:")
    logger.info(f"Visit: https://huggingface.co/{model_name}")
    logger.info("Click 'Access Repository' to accept the terms.\n")
    
    # Step 3: Create output directory
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Save location: {os.path.abspath(cache_dir)}")
    
    try:
        # Method 1: Using snapshot_download (recommended - downloads all files)
        logger.info("\n[1/3] Downloading model files with snapshot_download...")
        snapshot_download(
            repo_id=model_name,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info(f"✓ Model files downloaded to {cache_dir}")
        
        # Method 2: Verify by loading tokenizer
        logger.info("\n[2/3] Verifying tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        logger.info(f"✓ Tokenizer loaded successfully")
        logger.info(f"  Vocab size: {len(tokenizer)}")
        
        # Method 3: Verify model config
        logger.info("\n[3/3] Verifying model configuration...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(cache_dir)
        logger.info(f"✓ Model config loaded")
        logger.info(f"  Model type: {config.model_type}")
        logger.info(f"  Hidden size: {config.hidden_size}")
        logger.info(f"  Num layers: {config.num_hidden_layers}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Download complete!")
        logger.info("=" * 60)
        logger.info(f"\nModel saved to: {os.path.abspath(cache_dir)}")
        logger.info(f"Update .env to use: MODEL_PATH={os.path.abspath(cache_dir)}")
        
        return True
        
    except Exception as e:
        logger.error(f"\nDownload failed: {e}")
        logger.error("\nPossible causes:")
        logger.error("1. Token is invalid or expired")
        logger.error("2. Model license not accepted (visit https://huggingface.co/" + model_name + ")")
        logger.error("3. No internet connection")
        logger.error("4. Insufficient disk space")
        return False


def download_specific_file(repo_id: str, filename: str, cache_dir: str = None):
    """Download a specific file from a Hugging Face repo.
    
    Args:
        repo_id: Hugging Face model ID
        filename: Name of the file to download
        cache_dir: Local directory to save the file
    """
    if not login_to_huggingface():
        return False
    
    try:
        os.makedirs(cache_dir, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info(f"✓ Downloaded {filename} to {downloaded_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False


def list_available_gemma_models():
    """List available Gemma models on Hugging Face."""
    gemma_models = [
        ("google/gemma-2b", "2B parameters, base model"),
        ("google/gemma-2b-it", "2B parameters, instruction tuned"),
        ("google/gemma-4b", "4B parameters, base model"),
        ("google/gemma-4b-it", "4B parameters, instruction tuned"),
        ("google/gemma-7b", "7B parameters, base model"),
        ("google/gemma-7b-it", "7B parameters, instruction tuned"),
        ("google/gemma-9b", "9B parameters, base model"),
        ("google/gemma-9b-it", "9B parameters, instruction tuned"),
        ("google/gemma-27b", "27B parameters, base model"),
        ("google/gemma-27b-it", "27B parameters, instruction tuned"),
    ]
    
    logger.info("\nAvailable Gemma Models:")
    logger.info("=" * 60)
    for model_id, description in gemma_models:
        logger.info(f"  {model_id:<25} - {description}")
    logger.info("=" * 60)
    logger.info("\nRecommended for this project: google/gemma-2b-it (smallest, fastest)")
    logger.info("For better quality: google/gemma-7b-it")
    logger.info("\nNote: All Gemma models require:")
    logger.info("  1. Hugging Face account")
    logger.info("  2. Access token")
    logger.info("  3. Accepting model license on the model page")


def main():
    parser = argparse.ArgumentParser(
        description="Download Gemma models from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_model.py                          # Download default model (google/gemma-2b-it)
  python download_model.py --model google/gemma-7b-it --output ./models/gemma7b
  python download_model.py --list                   # List available models
  python download_model.py --check                  # Check authentication only
        """
    )
    parser.add_argument(
        "--model", "-m",
        default=settings.MODEL_NAME,
        help="Hugging Face model ID (default: google/gemma-2b-it)"
    )
    parser.add_argument(
        "--output", "-o",
        default=settings.MODEL_PATH,
        help="Output directory (default: ./models/fine_tuned_gemma)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available Gemma models"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check Hugging Face authentication"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_gemma_models()
        return
    
    if args.check:
        login_to_huggingface()
        return
    
    # Default: download model
    success = download_gemma_model(args.model, args.output)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
