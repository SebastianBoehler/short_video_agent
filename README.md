# Short Video Agent

A Python pipeline for generating TikTok-style short-form advertisements using AI models.

## Overview

This project provides a modular system to create vertical video ads from structured scene plans. It integrates with Replicate for text-to-video generation and video background removal.

## Features

- JSON-driven ad scene configuration
- Text-to-video generation via Replicate
- Video background removal and compositing
- Vertical video output (9:16 aspect ratio)
- Modular pipeline architecture

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd short_video_agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`
2. Add your Replicate API token to the `.env` file

## Development Status

This is a work in progress. Core functionality is being implemented and tested.

## Requirements

See `requirements.txt` for dependencies. Requires Python 3.8+ and a Replicate API token.
