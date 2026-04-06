# WhisperDesk

Offline speech-to-text app using local microphone input.

## Features
- Real-time microphone transcription
- Runs locally (no cloud)
- Prints text in terminal

## Tech Stack
- Python
- PyAudio
- faster-whisper

## How it works
Audio is recorded in chunks and transcribed using a local Whisper model.

## Tradeoff
- Smaller chunks → faster, less accurate
- Bigger chunks → slower, more accurate

## My settings
RECORD_SECONDS = 5

Average latency: ~7 seconds
