import os
from pydub import AudioSegment
from data_loader import load_rttm_files

def segment_audio(
    rttm_segments, 
    audio_dir='data/audio/voxconverse_test_wav 4', 
    output_dir='data/segments'
    ):
    os.makedirs(output_dir, exist_ok=True)

    for segment in rttm_segments:
        wav_file = segment['file']
        speaker = segment['speaker']
        start_ms = int(segment['start'] * 1000)
        end_ms = int((segment['start'] + segment['duration']) * 1000)

        audio_path = os.path.join(audio_dir, wav_file)
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue

        # Load the audio file
        audio = AudioSegment.from_wav(audio_path)
        segment_audio = audio[start_ms:end_ms]

        # Create subfolder for the audio file inside the output directory
        audio_file_name = os.path.splitext(wav_file)[0]  # Get the base name of the audio file
        audio_output_dir = os.path.join(output_dir, audio_file_name)

        # Ensure the subfolder exists
        os.makedirs(audio_output_dir, exist_ok=True)

        # Generate the segment file name
        segment_file_name = f"{audio_file_name}_{speaker}_{start_ms}_{end_ms}.wav"
        segment_output_path = os.path.join(audio_output_dir, segment_file_name)

        # Export the segment
        segment_audio.export(segment_output_path, format="wav")
        print(f"Exported segment to {segment_output_path}")

# Test
if __name__ == "__main__":
    segments = load_rttm_files(rttm_dir="data/annotations")
    segment_audio(segments, 
                  audio_dir='data/audio/voxconverse_test_wav 4',
                  output_dir='data/segments')