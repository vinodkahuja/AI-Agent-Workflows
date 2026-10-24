# NOTE: NRP-friendly, fully open-source alternative to 01_embedded_agent_speech.html.
# The original uses OpenAI's Realtime API over WebRTC, a native speech-to-speech
# model with no NRP equivalent (NRP is a plain text chat-completions gateway).
# This version rebuilds the "talk to it, it talks back" experience as a
# cascaded pipeline of three free, local, open-source pieces instead of one
# native audio-to-audio model:
#   1. Speech-to-text : faster-whisper (open-source Whisper reimplementation)
#   2. The agent's reasoning : NRP's gpt-oss, exactly like every other lesson
#   3. Text-to-speech : Piper (open-source, CPU-friendly local TTS)
# Everything runs locally except step 2 (the NRP chat completions call).
# It's turn-based (record -> transcribe -> reply -> speak) rather than the
# Realtime API's continuous, interruptible audio stream -- an honest
# simplification given there's no open, low-latency, full-duplex model that
# fits a laptop-CPU classroom setting the way this pipeline does.
import os
import sys
import wave
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import sounddevice as sd
import soundfile as sf
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    set_default_openai_client,
    set_default_openai_api,
    set_trace_processors,
)
from agents.tracing import TracingProcessor
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from piper import PiperVoice
from piper.download_voices import download_voice

# Load environment variables from .env file
load_dotenv()

# Point the Agents SDK at the NRP (Nautilus) OpenAI-compatible endpoint
client = AsyncOpenAI(
    base_url=os.getenv("NRP_BASE_URL"),
    api_key=os.getenv("NRP_API_KEY"),
)
set_default_openai_client(client, use_for_tracing=False)
set_default_openai_api("chat_completions")


class ConsoleTracingProcessor(TracingProcessor):
    """Prints trace/span utilization (tokens, duration) to the console."""

    def on_trace_start(self, trace):
        print(f"\n[trace] '{trace.name}' started")

    def on_trace_end(self, trace):
        print(f"[trace] '{trace.name}' finished")

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        data = span.span_data.export()
        summary = f"  [span] {data.get('type', 'unknown')}"
        if data.get("name"):
            summary += f" - {data['name']}"
        if data.get("usage"):
            summary += f" | usage={data['usage']}"
        print(summary)

    def shutdown(self):
        pass

    def force_flush(self):
        pass


# Replace the default OpenAI-backend exporter with a local console printer,
# so utilization is visible without needing a real OpenAI API key
set_trace_processors([ConsoleTracingProcessor()])

SAMPLE_RATE = 16000
VOICE_DIR = Path(__file__).with_name("piper_voices")
VOICE_NAME = "en_US-lessac-medium"


def load_whisper() -> WhisperModel:
    print("Loading speech-to-text model (faster-whisper, first run downloads it)...")
    return WhisperModel("tiny.en", device="cpu", compute_type="int8")


def load_piper_voice() -> PiperVoice:
    voice_path = VOICE_DIR / f"{VOICE_NAME}.onnx"
    if not voice_path.exists():
        print("Downloading text-to-speech voice (first run only)...")
        VOICE_DIR.mkdir(exist_ok=True, parents=True)
        download_voice(VOICE_NAME, VOICE_DIR)
    return PiperVoice.load(str(voice_path))


def pick_input_device():
    """Choose the mic: AUDIO_INPUT_DEVICE (index or name) if set, else the system default.

    Some machines (e.g. Jetson) have a raw ALSA default that can't capture, so
    we fall back to the 'pipewire'/'pulse' sound server if the default has no input.
    """
    wanted = os.getenv("AUDIO_INPUT_DEVICE")
    if wanted:
        return int(wanted) if wanted.isdigit() else wanted
    default_in = sd.default.device[0]
    if default_in is not None and default_in >= 0:
        return default_in
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and dev["name"] in ("pipewire", "pulse", "default"):
            return i
    return None


def record_until_enter() -> np.ndarray:
    """Records from the microphone until the user presses Enter.

    Captures at the device's native sample rate (many devices reject 16 kHz)
    and resamples to SAMPLE_RATE for Whisper.
    """
    frames = []
    device = pick_input_device()
    native_rate = int(sd.query_devices(device, "input")["default_samplerate"])

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    print("Recording... press Enter to stop.")
    with sd.InputStream(
        device=device, samplerate=native_rate, channels=1, dtype="int16", callback=callback
    ):
        input()
    if not frames:
        return np.zeros((0,), dtype="int16")
    audio = np.concatenate(frames, axis=0)[:, 0].astype(np.float32)
    if native_rate != SAMPLE_RATE:
        n_out = int(len(audio) * SAMPLE_RATE / native_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio
        )
    return audio.astype("int16")


def transcribe(whisper_model: WhisperModel, audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    segments, _ = whisper_model.transcribe(audio.astype(np.float32) / 32768.0)
    return " ".join(segment.text for segment in segments).strip()


def speak(piper_voice: PiperVoice, text: str) -> None:
    out_path = Path(__file__).with_name("_reply.wav")
    with wave.open(str(out_path), "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)
    data, samplerate = sf.read(out_path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


def main():
    # --text : type instead of speaking (no microphone / Whisper needed)
    # --mute : print the reply only, skip text-to-speech (no speakers needed)
    text_mode = "--text" in sys.argv
    mute = "--mute" in sys.argv

    whisper_model = None if text_mode else load_whisper()
    piper_voice = None if mute else load_piper_voice()

    agent = Agent(
        name="Voice Assistant",
        instructions="""
You are a friendly, concise voice assistant. Keep replies short (1-3 sentences)
since they will be read aloud. Avoid lists, markdown, or special formatting.
""",
        model="gpt-oss",
    )
    session = SQLiteSession("voice_agent_free", db_path=":memory:")

    print("=== Free NRP Voice Agent (Whisper + gpt-oss + Piper) ===")
    if text_mode:
        print("Text mode: type your message. Type 'q' + Enter to quit.\n")
    else:
        print("Press Enter to start talking. Type 'q' + Enter at any prompt to quit.\n")

    while True:
        if text_mode:
            user_text = input("You: ").strip()
            if user_text.lower() == "q":
                break
            if not user_text:
                continue
        else:
            user_ready = input("Press Enter to record (or 'q' to quit): ")
            if user_ready.strip().lower() == "q":
                break

            audio = record_until_enter()
            user_text = transcribe(whisper_model, audio)
            if not user_text:
                print("(Didn't catch that -- no speech detected.)\n")
                continue
            print(f"You said: {user_text}")

        result = Runner.run_sync(agent, user_text, session=session)
        reply_text = result.final_output
        print(f"Assistant: {reply_text}\n")

        if piper_voice:
            try:
                speak(piper_voice, reply_text)
            except Exception as e:
                print(f"(Could not play audio: {e}. Re-run with --mute to skip speech.)\n")


if __name__ == "__main__":
    main()
