import os
import sys
import time
import wave
import queue
import tempfile
import subprocess
import threading

import numpy as np
import pyaudio

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
RECORD_SECONDS = 5
SILENCE_THRESHOLD = 450

audio = pyaudio.PyAudio()
stream = None
model = None
BACKEND = None

audio_queue = queue.Queue()
stop_event = threading.Event()
full_transcript = []
chunk_counter = 0


def setup_faster_whisper():
    global model, BACKEND
    if WhisperModel is None:
        return False

    try:
        print("Loading faster-whisper model...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        BACKEND = "faster-whisper"
        print("faster-whisper ready.")
        return True
    except Exception as e:
        print(f"Could not load faster-whisper: {e}")
        return False


def setup_ollama():
    global BACKEND
    try:
        subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        BACKEND = "ollama"
        print("Ollama fallback ready.")
        return True
    except Exception:
        return False


def find_input_device():
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            return i, info.get("name", f"Device {i}")
    return None, None


def open_mic_stream():
    global stream
    device_index, device_name = find_input_device()

    if device_index is None:
        print("No microphone input device found.")
        sys.exit(1)

    print(f"Using mic: {device_name}")

    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
        )
    except Exception as e:
        print(f"Could not open microphone: {e}")
        sys.exit(1)


def is_silent(audio_data):
    if not audio_data:
        return True
    samples = np.frombuffer(audio_data, dtype=np.int16)
    if len(samples) == 0:
        return True
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    return rms < SILENCE_THRESHOLD


def record_chunk():
    frames = []
    total_frames = int(RATE / CHUNK * RECORD_SECONDS)

    for _ in range(total_frames):
        if stop_event.is_set():
            return None
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except Exception:
            return None

    audio_data = b"".join(frames)
    if is_silent(audio_data):
        return None
    return audio_data


def save_audio_to_temp(audio_data):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp.name
    temp.close()

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)

    return temp_path


def transcribe_with_whisper(audio_file_path):
    segments, _ = model.transcribe(
        audio_file_path,
        beam_size=1,
        language="en",
        vad_filter=True,
    )
    text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
    return " ".join(text_parts).strip()


def transcribe_with_ollama(audio_file_path):
    prompt = (
        "You are given a path to a WAV audio file recorded from a microphone. "
        "Attempt to transcribe spoken English from it as accurately as possible. "
        "If you cannot process audio files directly, respond with a short notice."
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:3b", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[ollama error: {e}]"


def transcribe(audio_file_path):
    if BACKEND == "faster-whisper":
        return transcribe_with_whisper(audio_file_path)
    if BACKEND == "ollama":
        return transcribe_with_ollama(audio_file_path)
    return ""


def transcription_worker():
    while not stop_event.is_set() or not audio_queue.empty():
        try:
            item = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        chunk_id, audio_data = item
        temp_path = None

        try:
            temp_path = save_audio_to_temp(audio_data)

            start_time = time.time()
            text = transcribe(temp_path)
            inference_time = time.time() - start_time
            total_latency = RECORD_SECONDS + inference_time

            if text:
                full_transcript.append(text)
                print(
                    f"\n[{chunk_id}] {text}\n"
                    f"    inference: {inference_time:.2f}s | "
                    f"total latency: {total_latency:.2f}s | "
                    f"buffer: {RECORD_SECONDS}s"
                )
            else:
                print(f"\n[{chunk_id}] No speech detected after transcription.")

        except Exception as e:
            print(f"\n[{chunk_id}] Transcription error: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            audio_queue.task_done()


def main():
    global chunk_counter

    if not setup_faster_whisper():
        print("faster-whisper unavailable, trying ollama fallback...")
        if not setup_ollama():
            print("No transcription backend available.")
            print("Install faster-whisper or make sure ollama is installed.")
            sys.exit(1)

    open_mic_stream()

    worker = threading.Thread(target=transcription_worker, daemon=True)
    worker.start()

    print("\nWhisperDesk — Local Speech-to-Text")
    print(f"Backend: {BACKEND}")
    print(f"Chunk size: {RECORD_SECONDS}s")
    print(f"Silence threshold: {SILENCE_THRESHOLD}")
    print("\nSpeak into your microphone.")
    print("Text appears after each chunk.")
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            sys.stdout.write(f"\rListening ({RECORD_SECONDS}s chunk)... ")
            sys.stdout.flush()

            audio_data = record_chunk()

            if audio_data is None:
                sys.stdout.write("\rSilence or stopped.               ")
                sys.stdout.flush()
                continue

            chunk_counter += 1
            audio_queue.put((chunk_counter, audio_data))

            sys.stdout.write("\rQueued for transcription.         ")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    finally:
        audio_queue.join()

        if stream is not None:
            stream.stop_stream()
            stream.close()

        audio.terminate()

        print("\nFull transcript:\n")
        if full_transcript:
            print(" ".join(full_transcript))
        else:
            print("No transcript captured.")


if __name__ == "__main__":
    main()