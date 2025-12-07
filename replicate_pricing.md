# Replicate API Pricing Analysis

Based on actual generation runs from the video pipeline, this document provides cost breakdowns for different models and operations.

## Model Pricing Data

### Video Generation Models

#### wan-video/wan-2.5-i2v

- **Cost per generation**: $0.50
- **Average duration**: ~1m 45s
- **Use case**: Text-to-video with image input (speaker video generation)
- **Examples from pipeline**:
  - Speaker video generation with female reference images
  - Background video animation from static images
  - 5-second vertical video generation

#### wan-video/wan-2.5-t2v

- **Cost per generation**: $0.50 (estimated based on i2v pricing)
- **Use case**: Pure text-to-video (broll scenes)
- **Note**: Hit billing limits during testing

### Image Generation Models

#### bytedance/seedream-4.5

- **Cost per generation**: $0.04
- **Average duration**: ~22s
- **Use case**: High-quality background image generation
- **Examples from pipeline**:
  - Crypto trading dashboards
  - Fitness progress tracking interfaces
  - Battery life comparison charts
  - Premium product visualizations

### Background Removal Models

#### arielreplicate/robust_video_matting:73d2128

- **Cost per generation**: <$0.01 (practically free)
- **Average duration**: ~1m 20s
- **Use case**: Video background removal for speaker overlay
- **Examples from pipeline**:
  - Green screen removal from speaker videos
  - Alpha mask generation for compositing

## Pipeline Cost Breakdown

### Per Scene Costs (Speaker Scenes)

```
Speaker Video (wan-2.5-i2v):     $0.50
Background Image (seedream-4.5): $0.04
Background Animation (wan-2.5-i2v): $0.50
Background Removal (matting):    <$0.01
---
Total per speaker scene:         ~$1.04
```

### Per Scene Costs (Broll Scenes)

```
Direct Video (wan-2.5-t2v):      $0.50
---
Total per broll scene:           ~$0.50
```

## Complete Scheme Costs

Based on actual pipeline runs:

### Short Schemes (4 scenes, 20s)

- **Estimated cost**: ~$2.50-$4.00
- **Examples**: fortnite_deathrun_challenge, reddit_beichte_german

### Medium Schemes (5 scenes, 25s)

- **Estimated cost**: ~$3.50-$5.00
- **Examples**: crypto_market_update

### Long Schemes (7-8 scenes, 35-40s)

- **Estimated cost**: ~$5.00-$8.00
- **Examples**: tech_product_showcase (7 scenes), fitness_transformation_story (8 scenes)

## Cost Optimization Strategies

### 1. Background Type Selection

- **`image`**: $0.04 per scene (static backgrounds only)
- **`video`**: $0.50 per scene (direct text-to-video)
- **`image_to_video`**: $0.54 per scene (image + animation)

### 2. Scene Type Optimization

- **Broll scenes**: $0.50 vs $1.04 for speaker scenes
- **Mixed approach**: Use broll for product showcases, speaker for narrative

### 3. Duration Management

- **5-second scenes**: Standard pricing ($0.50)
- **10-second scenes**: May cost more (not tested)

## Actual Usage Data

From the provided API logs:

| Model                | Runs   | Total Cost | Avg Duration |
| -------------------- | ------ | ---------- | ------------ |
| wan-2.5-i2v          | 7      | $3.50      | ~1m 45s      |
| seedream-4.5         | 4      | $0.16      | ~22s         |
| robust-video-matting | 4      | <$0.04     | ~1m 20s      |
| **Total**            | **15** | **~$3.70** | -            |

## Billing Considerations

- **Rate limiting**: Encountered 11-second delays on API calls
- **Credit requirements**: Need sufficient Replicate credits for batch processing
- **Concurrent limits**: Sequential processing recommended to avoid rate limits
- **Retry logic**: Built-in retry mechanism handles temporary failures

## Recommendations

1. **Budget Planning**: Allocate ~$1.00 per scene for mixed content
2. **Batch Processing**: Run schemes individually to manage costs
3. **Background Optimization**: Use `image` type for cost-sensitive projects
4. **Monitoring**: Track API usage to avoid billing interruptions
5. **Testing**: Validate single scenes before full scheme generation

## Cost Comparison

| Feature              | Cost          | Efficiency                   |
| -------------------- | ------------- | ---------------------------- |
| Static Background    | $0.04         | Most cost-effective          |
| Broll Video          | $0.50         | Good for product demos       |
| Speaker + Background | $1.04         | Full narrative capability    |
| Audio Mixing         | No extra cost | Adds value to speaker scenes |

This pricing data is based on actual pipeline runs and provides realistic cost estimates for video generation projects.
