output = replicate.run(
"google/veo-3.1-fast",
input={
"image": "https://replicate.delivery/pbxt/NtDCMBJNIQTOU0mZtlnlrqrPLgYvTvpCISbFIiweYPsotGY5/replicate-prediction-gn4tnddn5drme0csx1yt3jvy4c.jpeg",
"prompt": "Overlapping geometric shapes, pulsing to upbeat electronic music with SFX of shifting patterns",
"duration": 8,
"last_frame": "https://replicate.delivery/pbxt/NtDCLnwTQaPfLhgaNDmLevN8QivDFS8V91M8pCwEpDNIN9uA/replicate-prediction-8m82ekaj7hrma0csx1xrkmqjhm.jpeg",
"resolution": "720p",
"aspect_ratio": "9:16",
"generate_audio": True
}
)

# To access the file URL:

print(output.url())
#=> "http://example.com"

# To write the file to disk:

with open("my-image.png", "wb") as file:
file.write(output.read())

output = replicate.run(
"arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
input={
"input_video": "https://replicate.delivery/pbxt/HqiGGuuwynO7sCHpcQdYQsIf04NotwOrDdbhBf4M6Pou6MGg/butter.mp4",
"output_type": "green-screen"
}
)

# To access the file URL:

print(output.url())
#=> "http://example.com"

# To write the file to disk:

with open("my-image.png", "wb") as file:
file.write(output.read())

output = replicate.run(
"black-forest-labs/flux-2-pro",
input={
"prompt": "Glossy candy-colored 3D letters in hot pink, electric orange, and lime green on a sun-drenched poolside patio with bold terrazzo tiles and vintage lounge chairs in turquoise and yellow. Shot on Kodachrome film with a Hasselblad 500C, warm golden afternoon sunlight, dramatic lens flare, punchy oversaturated colors with that distinctive 70s yellow-orange cast, shallow depth of field with the text sharp in the foreground, tropical palms and a sparkling aquamarine pool behind that spells out \"Run FLUX.2 [pro] on Replicate!\"",
"resolution": "1 MP",
"aspect_ratio": "1:1",
"input_images": [],
"output_format": "webp",
"output_quality": 80,
"safety_tolerance": 2
}
)

# To access the file URL:

print(output.url())
#=> "http://example.com"

# To write the file to disk:

with open("my-image.png", "wb") as file:
file.write(output.read())
