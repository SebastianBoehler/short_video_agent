#!/usr/bin/env python3
"""
Short Video Agent - CLI entry point.

Usage:
    python run.py --config schemes/my_ad.yaml
    python run.py --config schemes/my_ad.yaml --backend local
    python run.py --config schemes/my_ad.yaml --output outputs/my_video/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.runner import VideoPipeline
from src.generators.ltx import check_ltx_availability


def main():
    parser = argparse.ArgumentParser(
        description="Short Video Agent - Generate TikTok-style videos from YAML configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate video using Replicate API
    python run.py --config schemes/my_ad.yaml
    
    # Generate video using local LTX-2 model
    python run.py --config schemes/my_ad.yaml --backend local
    
    # Check GPU availability for local models
    python run.py --check-gpu
    
    # Specify output directory
    python run.py --config schemes/my_ad.yaml --output outputs/custom/
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML/JSON config file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: outputs/{scheme_name}/)"
    )
    
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["replicate", "local", "hybrid"],
        default="replicate",
        help="Backend for video generation (default: replicate)"
    )
    
    parser.add_argument(
        "--video-model",
        type=str,
        default=None,
        help="Override video model (e.g., wan-2.5-i2v, ltx-2)"
    )
    
    parser.add_argument(
        "--speaker-model",
        type=str,
        default=None,
        help="Override speaker model (e.g., veo-3.1-fast)"
    )
    
    parser.add_argument(
        "--image-model",
        type=str,
        default=None,
        help="Override image model (e.g., seedream-4.5, nano-banana-pro)"
    )
    
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Check GPU availability for local models"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    # Check GPU availability
    if args.check_gpu:
        print("\n🔍 Checking GPU availability for local models...\n")
        result = check_ltx_availability()
        
        print(f"Available: {'✅ Yes' if result['available'] else '❌ No'}")
        if result['device']:
            print(f"Device: {result['device']}")
        if result['vram_gb']:
            print(f"VRAM: {result['vram_gb']:.1f} GB")
        
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  • {rec}")
        
        return 0
    
    # List models
    if args.list_models:
        from src.config.models import ModelRegistry, ModelType
        
        print("\n📋 Available Models:\n")
        
        for model_type in ModelType:
            models = ModelRegistry.list_models(model_type)
            if models:
                print(f"  {model_type.value}:")
                for m in models:
                    default = " (default)" if m.name == ModelRegistry.get_default(model_type) else ""
                    print(f"    • {m.name}{default} - {m.description}")
                print()
        
        return 0
    
    # Require config for generation
    if not args.config:
        parser.print_help()
        print("\n❌ Error: --config is required for video generation")
        return 1
    
    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        return 1
    
    # Run pipeline
    try:
        pipeline = VideoPipeline(
            backend=args.backend,
            video_model=args.video_model,
            speaker_model=args.speaker_model,
            image_model=args.image_model,
        )
        
        final_video = pipeline.run(
            config_path=str(config_path),
            output_dir=args.output,
        )
        
        print(f"\n🎉 Success! Final video: {final_video}")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n💥 Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
