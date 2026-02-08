import librosa
import numpy as np
import os

def analyze_music_for_rowdiness(file_path, target_fps=60):
    """
    Analyzes audio and returns a rowdiness curve scaled by the song's
    Global Energy (BPM + Average Loudness).
    """
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=None)
    
    # 2. Basic Features (Local)
    hop_length = int(sr / target_fps)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # 3. Normalize Local Features (Relative to track peaks)
    # This captures the "shape" of the song (drops, breaks)
    if np.max(rms) > 0:
        rms_norm = rms / np.max(rms)
    else:
        rms_norm = rms
        
    if np.max(onset_env) > 0:
        onset_norm = onset_env / np.max(onset_env)
    else:
        onset_norm = onset_env

    # 4. Create Base Curve (0.0 to 1.0 relative)
    # 60% Loudness, 40% Percussion
    raw_curve = (0.6 * rms_norm) + (0.4 * onset_norm)
    
    # --- 5. CALCULATE GLOBAL ENERGY FACTOR (The Fix) ---
    
    # A. Tempo Factor
    # We assume 'Rowdy' implies higher BPM.
    # Map 60 BPM -> 0.4 multiplier
    # Map 170 BPM -> 1.2 multiplier
    tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo_array, "item"):
        bpm = tempo_array.item()
    else:
        bpm = tempo_array
    
    # Clamp BPM to reasonable range and map to 0.4 - 1.2
    # Logic: (BPM - Min) / (Max - Min)
    bpm_clamped = np.clip(bpm, 60.0, 170.0)
    bpm_factor = 0.4 + ((bpm_clamped - 60.0) / (110.0)) * 0.8
    
    # B. Intensity Factor (Sustained Loudness)
    # Check if the song is consistently loud (RMS mean) vs dynamic
    avg_loudness = np.mean(rms_norm)
    # Map 0.1 (quiet/sparse) -> 0.8, 0.5 (wall of sound) -> 1.2
    loudness_factor = 0.8 + (avg_loudness * 0.8)
    
    # Combine factors
    # Example: Lullaby (87 BPM, Low Density) -> 0.6 * 0.9 = ~0.54 max potential
    # Example: Hype Track (150 BPM, High Density) -> 1.1 * 1.1 = ~1.21 max potential
    global_energy = bpm_factor * loudness_factor
    
    # 6. Apply Global Scaling
    final_curve = raw_curve * global_energy
    
    # 7. Post-Processing
    # Clip to 0.0 - 1.0 (Physics engine expects this range)
    final_curve = np.clip(final_curve, 0.0, 1.0)
    
    return {
        "curve": final_curve,
        "duration": librosa.get_duration(y=y, sr=sr),
        "fps": target_fps,
        "tempo": bpm
    }