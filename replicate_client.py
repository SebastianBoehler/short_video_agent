import os
import replicate
import time
import re
from dotenv import load_dotenv

load_dotenv()
client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

def retry_replicate_call(func, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "status: 429" in str(e) and "rate limit resets in" in str(e):
                reset_match = re.search(r'resets in ~(\d+)s', str(e))
                if reset_match:
                    wait_time = int(reset_match.group(1)) + 2
                    print(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            if attempt == max_retries - 1:
                raise
            time.sleep(delay * (2 ** attempt))

def generate_video(prompt, duration, generate_audio=True):
    def call_replicate():
        input_data = {
            "prompt": prompt,
            "duration": duration,
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "generate_audio": generate_audio
        }
        print(f"Generating video with input: {input_data}")
        return client.run(
            "google/veo-3.1-fast",
            input=input_data
        )
    return retry_replicate_call(call_replicate)

def generate_image(prompt):
    def call_replicate():
        input_data = {
            "prompt": prompt,
            "resolution": "1 MP",
            "aspect_ratio": "9:16",
            "output_format": "webp"
        }
        print(f"Generating image with input: {input_data}")
        return client.run(
            "black-forest-labs/flux-2-pro",
            input=input_data
        )
    return retry_replicate_call(call_replicate)

def remove_background(input_video):
    def call_replicate():
        # Open file as binary for Replicate upload
        with open(input_video, 'rb') as video_file:
            input_data = {
                "input_video": video_file,
                "output_type": "alpha-mask"
            }
            print(f"Removing background with input: {input_data}")
            return client.run(
                "arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
                input=input_data
            )
    return retry_replicate_call(call_replicate)
