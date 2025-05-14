import os

def load_rttm_files(rttm_dir='data/annotations'):
    """
    Load RTTM files and return a list of dictionaries with segment info.
    Each dictionary contains: file, channel, start, duration, speaker.
    """
    segments = []

    for file_name in os.listdir(rttm_dir):
        if file_name.endswith('.rttm'):
            file_path = os.path.join(rttm_dir, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 9 and parts[0] == 'SPEAKER':
                        segment = {
                            'file': file_name.replace('.rttm', '.wav'),
                            'channel': int(parts[2]),
                            'start': float(parts[3]),
                            'duration': float(parts[4]),
                            'speaker': parts[7]
                        }
                        segments.append(segment)
    return segments

# test
if __name__ == "__main__":
    print(load_rttm_files())