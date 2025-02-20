#!/usr/bin/env python3

import os
import time
import tempfile
import subprocess
import json
import glob
from pathlib import Path

# Audio/Video libraries
import ffmpeg
import moviepy.editor as mp
from pydub import AudioSegment, silence

# For transcription
import whisper

# (Optional) For environment variables / config, if needed
from dotenv import load_dotenv
load_dotenv()

# Global config (You can adjust these or move them to .env)
SILENCE_THRESHOLD = -70  # in dBFS, tune for your audio
MIN_SPEECH_DURATION = 500  # in ms
MARGIN_DURATION = 200  # in ms
RAW_FOLDER = "raw"
EDITED_FOLDER = "edited_videos"

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def process_video(video_file: str) -> str:
    """
    Extract the audio track from a video file using ffmpeg.
    Returns the path to the extracted .wav file.
    """
    base_name = os.path.basename(os.path.splitext(video_file)[0])
    audio_output = os.path.join(tempfile.gettempdir(), f"{base_name}_temp_audio.wav")

    # Use ffmpeg to extract audio
    (
        ffmpeg
        .input(video_file)
        .output(audio_output, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run(quiet=True)
    )

    return audio_output


def detect_speech_segments(audio_path: str):
    """
    Detect speech segments by finding silent parts in the audio.
    Returns a list of (start_ms, end_ms) for each speech segment.
    """
    audio = AudioSegment.from_wav(audio_path)
    # Invert logic: detect silent chunks, then invert to get speech segments
    silent_ranges = silence.detect_silence(
        audio,
        min_silence_len=MIN_SPEECH_DURATION,
        silence_thresh=SILENCE_THRESHOLD
    )
    # silent_ranges are tuples of (start_ms, end_ms) for silence
    # We'll invert these to find speech. A simple approach:
    if not silent_ranges:
        # If no silence detected, assume entire file is speech
        return [(0, len(audio))]

    speech_segments = []
    prev_end = 0

    for (sil_start, sil_end) in silent_ranges:
        # The segment before the silence is speech
        if sil_start - prev_end > 0:
            speech_segments.append((prev_end, sil_start))
        prev_end = sil_end

    # last segment after final silent range
    if prev_end < len(audio):
        speech_segments.append((prev_end, len(audio)))

    # Apply margin
    speech_segments_with_margin = []
    for (start, end) in speech_segments:
        start = max(0, start - MARGIN_DURATION)
        end = min(len(audio), end + MARGIN_DURATION)
        speech_segments_with_margin.append((start, end))

    return speech_segments_with_margin


def transcribe_segments(audio_path: str, segments):
    """
    Transcribe each audio segment using Whisper.
    Returns a list of transcribed texts, in the same order as the segments.
    """
    model = whisper.load_model("base")  # or "small", "medium", etc.
    audio = AudioSegment.from_wav(audio_path)

    transcriptions = []
    for (start, end) in segments:
        # Export segment to a temporary file
        segment_audio = audio[start:end]
        segment_path = os.path.join(tempfile.gettempdir(), "temp_segment.wav")
        segment_audio.export(segment_path, format="wav")

        # Transcribe with Whisper
        result = model.transcribe(segment_path, fp16=False)
        text = result.get("text", "").strip()
        transcriptions.append(text)

        # Clean up temporary file
        if os.path.exists(segment_path):
            os.remove(segment_path)

    return transcriptions


def remove_duplicate_phrases(transcriptions):
    """
    Remove repeated phrases by only keeping their last occurrence.
    Return the filtered transcription list.
    """
    filtered = []
    seen_phrases = set()
    # We'll invert the iteration to keep the last occurrence
    for i in range(len(transcriptions) - 1, -1, -1):
        if transcriptions[i] not in seen_phrases:
            seen_phrases.add(transcriptions[i])
            filtered.append(transcriptions[i])
    # Reverse again to restore correct chronological order
    filtered.reverse()
    return filtered


def generate_cleaned_transcript(filtered_transcriptions, segments):
    """
    Merge the filtered transcriptions with timestamps.
    Returns a list of dicts with { 'start': ms, 'end': ms, 'text': str }.
    Also optionally save a JSON log if needed.
    """
    cleaned = []
    # For simplicity, assume the length of filtered_transcriptions matches segments
    # or that you want to pair them in order
    for (segment, text) in zip(segments, filtered_transcriptions):
        start_ms, end_ms = segment
        cleaned.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text
        })

    # Example: save a JSON transcript (optional)
    transcript_path = os.path.join(tempfile.gettempdir(), "cleaned_transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    return cleaned


def compile_final_video(video_file: str, cleaned_transcript):
    """
    Reconstruct the final video by cutting silent/unwanted segments using moviepy.
    Returns the final video clip object (in memory) ready for export.
    """
    # Load the original video
    clip = mp.VideoFileClip(video_file)

    # Build subclips from the cleaned transcript
    subclips = []
    for entry in cleaned_transcript:
        start_s = entry["start_ms"] / 1000.0
        end_s = entry["end_ms"] / 1000.0
        subclip = clip.subclip(start_s, end_s)
        subclips.append(subclip)

    if not subclips:
        return None

    final_clip = mp.concatenate_videoclips(subclips, method="compose")
    return final_clip


def save_output(final_clip, output_folder: str, original_video: str):
    """
    Save the final compiled video to the output folder.
    """
    if not final_clip:
        print(f"No segments found to compile for {original_video}. Skipping save.")
        return

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    base_name = os.path.basename(original_video)
    edited_output_path = os.path.join(output_folder, f"edited_{base_name}")

    # Remove existing edited file if it exists
    if os.path.exists(edited_output_path):
        os.remove(edited_output_path)

    final_clip.write_videofile(edited_output_path, codec="libx264", audio_codec="aac")
    final_output_path = os.path.join(output_folder, f"final_{base_name}")

    # Remove existing final file if it exists
    if os.path.exists(final_output_path):
        os.remove(final_output_path)

    normalize_cmd = f"ffmpeg -y -i \"{edited_output_path}\" -af loudnorm \"{final_output_path}\""
    print(f"Normalizing audio: {normalize_cmd}")
    subprocess.run(normalize_cmd, shell=True, check=True)
    print(f"Saved normalized video to: {final_output_path}")

    # Remove the intermediate edited file after normalization
    if os.path.exists(edited_output_path):
        os.remove(edited_output_path)
        print(f"Removed intermediate file: {edited_output_path}")


def main():
    import shutil
    if not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            os.environ["FFMPEG_BINARY"] = ffmpeg_path
            print("ffmpeg not found in PATH, using imageio-ffmpeg at", ffmpeg_path)
        except ImportError:
            print("ffmpeg not found and imageio-ffmpeg not installed. Please install ffmpeg.")
            return
    # Create output folder if it doesn't exist
    Path(EDITED_FOLDER).mkdir(parents=True, exist_ok=True)

    video_files = glob.glob(os.path.join(RAW_FOLDER, "*.mp4")) + glob.glob(os.path.join(RAW_FOLDER, "*.mkv"))
    if not video_files:
        print(f"No .mp4 or .mkv files found in '{RAW_FOLDER}' folder.")
        return

    for video_file in video_files:
        print(f"Processing video: {video_file}")
        start_time = time.time()
        audio_path = process_video(video_file)
        segments = detect_speech_segments(audio_path)
        transcriptions = transcribe_segments(audio_path, segments)

        filtered_transcriptions = remove_duplicate_phrases(transcriptions)
        cleaned_transcript = generate_cleaned_transcript(filtered_transcriptions, segments)

        final_clip = compile_final_video(video_file, cleaned_transcript)
        save_output(final_clip, EDITED_FOLDER, video_file)
        end_time = time.time()
        processing_time = end_time - start_time
        original_clip = mp.VideoFileClip(video_file)
        original_duration = original_clip.duration
        original_clip.close()
        if final_clip:
            final_duration = final_clip.duration
        else:
            final_duration = 0
        time_cut = original_duration - final_duration
        print(f"Processing time: {format_time(processing_time)}")
        print(f"Original runtime: {format_time(original_duration)}")
        print(f"Final runtime: {format_time(final_duration)}")
        print(f"Total time cut: {format_time(time_cut)}")

        # Clean up temp audio if desired
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Removed temporary audio file: {audio_path}")


if __name__ == "__main__":
    main()