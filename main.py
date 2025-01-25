import torch
from TTS.api import TTS
import json
from pydub import AudioSegment
from tqdm import tqdm
import os
import subprocess
from pathlib import Path
import pickle  # Add this import at the top with other imports
import whisper
import numpy as np
from difflib import SequenceMatcher

# Choose CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def split_into_chunks(text, max_length=200):
    """
    Split text into smaller chunks while preserving sentence structure and respecting token limits.
    Max length reduced to 200 to stay well under the 400 token limit.
    """
    # First split by periods, then clean up
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        # If single sentence is too long, split by commas
        if len(sentence) > max_length:
            comma_parts = [p.strip() + "," for p in sentence.split(",") if p.strip()]
            for part in comma_parts:
                # If part is still too long, split by spaces
                if len(part) > max_length:
                    words = part.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_length:
                            temp_chunk += " " + word if temp_chunk else word
                        else:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                else:
                    chunks.append(part)
        # Try to combine sentences while respecting max_length
        elif len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Final cleanup: ensure no chunk exceeds max_length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # Split oversized chunks
            words = chunk.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_length:
                    temp_chunk += " " + word if temp_chunk else word
                else:
                    final_chunks.append(temp_chunk.strip())
                    temp_chunk = word
            if temp_chunk:
                final_chunks.append(temp_chunk.strip())

    return final_chunks

def create_chapter_file(chapters_info, output_file):
    """Create a chapters metadata file for ffmpeg"""
    with open('chapters.txt', 'w', encoding='utf-8') as f:
        for i, ch in enumerate(chapters_info):
            start_time = int(ch["start_ms"] / 1000)  # Convert to seconds
            end_time = int(ch["end_ms"] / 1000)
            
            # Convert to HH:MM:SS format
            start_str = f"{start_time//3600:02d}:{(start_time%3600)//60:02d}:{start_time%60:02d}"
            end_str = f"{end_time//3600:02d}:{(end_time%3600)//60:02d}:{end_time%60:02d}"
            
            f.write(f"[CHAPTER]\nTIMEBASE=1/1\nSTART={start_time}\nEND={end_time}\ntitle={ch['title']}\n\n")

def text_similarity(text1, text2):
    """
    Calculate similarity ratio between two texts after normalizing them.
    Returns float between 0 and 1, where 1 means identical texts.
    """
    # Normalize texts for comparison
    def normalize(text):
        return ' '.join(text.lower().split())
    
    return SequenceMatcher(None, normalize(text1), normalize(text2)).ratio()

def verify_audio_quality(audio_file: str, original_text: str, whisper_model) -> bool:
    """
    Verify the quality of generated audio by converting it back to text
    and comparing with the original.
    Returns True if quality is acceptable, False otherwise.
    """
    # Transcribe the audio
    result = whisper_model.transcribe(audio_file)
    transcribed_text = result["text"]
    
    # Calculate similarity
    similarity = text_similarity(original_text, transcribed_text)
    
    # You can adjust this threshold based on your needs
    SIMILARITY_THRESHOLD = 0.85
    return similarity >= SIMILARITY_THRESHOLD, similarity

def create_audiobook_from_pickle(pickle_path: str, output_file: str):
    """
    Reads chapters from pickle file, performs TTS on each chapter, 
    and outputs an M4B file with embedded chapter markers.
    """
    # Load pickle array
    with open(pickle_path, "rb") as f:
        chapters = pickle.load(f)

    # Initialize TTS and Whisper models
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    whisper_model = whisper.load_model("base")

    # Prepare empty AudioSegment
    final_audio = AudioSegment.silent(duration=0)

    # Track chapter information
    chapters_info = []
    current_offset = 0  # ms

    # Process chapters
    for chapter in tqdm(chapters, desc="Chapters", unit="chapter"):
        if not chapter["chapter_content"]:
            continue

        chapter_text = " ".join(chapter["chapter_content"]).strip()
        if not chapter_text:
            continue

        chapter_start = current_offset
        text_chunks = split_into_chunks(chapter_text, max_length=250)
        chapter_audio = AudioSegment.silent(duration=0)

        for i, chunk in enumerate(tqdm(text_chunks, desc="Chunks", leave=False)):
            max_attempts = 3
            attempt = 0
            chunk_generated = False

            while attempt < max_attempts and not chunk_generated:
                temp_wav = f"temp_{chapter['chapter_title']}_{i}_attempt_{attempt}.wav"

                try:
                    # Generate audio from text
                    tts.tts_to_file(
                        text=chunk,
                        file_path=temp_wav,
                        speaker="Damien Black",
                        language="en",
                        split_sentences=False
                    )

                    # Verify the audio quality
                    is_good_quality, similarity = verify_audio_quality(temp_wav, chunk, whisper_model)

                    if is_good_quality:
                        chunk_audio = AudioSegment.from_wav(temp_wav)
                        chapter_audio += chunk_audio
                        chunk_generated = True
                        print(f"Chunk {i} generated successfully (similarity: {similarity:.2f})")
                    else:
                        print(f"Chunk {i} attempt {attempt + 1} failed (similarity: {similarity:.2f})")
                        attempt += 1

                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
                    attempt += 1
                finally:
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)

            if not chunk_generated:
                print(f"Warning: Failed to generate good quality audio for chunk after {max_attempts} attempts")
                # Use the last generated audio as fallback
                chunk_audio = AudioSegment.from_wav(temp_wav)
                chapter_audio += chunk_audio

        final_audio += chapter_audio
        chapter_end = current_offset + len(chapter_audio)

        chapters_info.append({
            "start_ms": chapter_start,
            "end_ms": chapter_end,
            "title": chapter["chapter_title"]
        })

        current_offset = chapter_end

    try:
        # Export as temporary audio file
        temp_audio = "temp_audiobook.m4a"
        final_audio.export(temp_audio, format="ipod", codec="aac")

        # Create chapter metadata file
        create_chapter_file(chapters_info, "chapters.txt")

        # Use ffmpeg to create the final m4b with chapters
        cmd = [
            "ffmpeg", "-i", temp_audio,
            "-f", "ffmetadata", "-i", "chapters.txt",
            "-map_chapters", "1",  # Added explicit chapter mapping
            "-map", "0",
            "-codec", "copy",
            "-metadata", "title=Audiobook",
            "-metadata", "artist=TTS",
            "-movflags", "+faststart",  # Added for better player compatibility
            output_file
        ]

        subprocess.run(cmd, check=True)
        print("Successfully created audiobook with chapters")

    except Exception as e:
        print(f"Error creating audiobook: {e}")
        raise
    finally:
        # Clean up temporary files
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        if os.path.exists("chapters.txt"):
            os.remove("chapters.txt")

def main():
    # Define the input directory where pickle files are located
    input_dir = "pickle_files"  # Changed from json_files
    
    # Create output directory if it doesn't exist
    output_dir = "audiobooks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all pickle files in the input directory
    pickle_files = list(Path(input_dir).glob("*.pkl"))
    
    for pickle_file in pickle_files:
        # Create output filename with .m4b extension
        output_filename = pickle_file.stem + ".m4b"
        output_path = os.path.join(output_dir, output_filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_filename} - file already exists")
            continue
        
        print(f"\nProcessing {pickle_file.name}...")
        try:
            create_audiobook_from_pickle(str(pickle_file), output_path)
            print(f"Successfully created {output_filename}")
        except Exception as e:
            print(f"Error processing {pickle_file.name}: {e}")
            continue

if __name__ == "__main__":
    main()