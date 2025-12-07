#!/usr/bin/env python3
"""
Run all 7 video schemes sequentially
"""
import subprocess
import sys
import os

schemes = [
    'schemes/crypto_market_update.yaml',
    'schemes/fortnite_deathrun_challenge.yaml', 
    'schemes/animated_car_poster.yaml',
    'schemes/reddit_beichte_german.yaml',
    'schemes/luxury_watch_affiliate.yaml',
    'schemes/fitness_transformation_story.yaml',
    'schemes/tech_product_showcase.yaml'
]

def run_scheme(scheme):
    """Run a single scheme"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting: {scheme}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, 'pipeline.py', scheme
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {scheme}")
            print(result.stdout)
        else:
            print(f"❌ FAILED: {scheme}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {scheme}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {scheme} - {e}")
        return False
    
    return True

def main():
    """Run all schemes"""
    print("🎬 Running all 7 video schemes...")
    print(f"Total schemes: {len(schemes)}")
    
    success_count = 0
    failed_schemes = []
    
    for i, scheme in enumerate(schemes, 1):
        print(f"\n📊 Progress: {i}/{len(schemes)}")
        
        if run_scheme(scheme):
            success_count += 1
        else:
            failed_schemes.append(scheme)
    
    print(f"\n{'='*60}")
    print(f"🏁 COMPLETE!")
    print(f"✅ Successful: {success_count}/{len(schemes)}")
    if failed_schemes:
        print(f"❌ Failed: {len(failed_schemes)}")
        for scheme in failed_schemes:
            print(f"   - {scheme}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
