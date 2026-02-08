import librosa
import librosa.beat
import librosa.onset
import librosa.feature
import numpy as np

def analyze_music_for_rowdiness(file_path, target_fps=60):
    """
    Analyzes audio and returns a 'Visceral' rowdiness curve.
    Optimized for hard drops, high BPM, and punchy transitions.
    Includes fixes for Trap/Half-time detection and outlier suppression.
    """
    y, sr = librosa.load(file_path, sr=None)
    
    hop_length = int(sr / target_fps)
    
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    def robust_normalize(data):
        if len(data) == 0: return data
        ref_value = np.percentile(data, 96)
        if ref_value <= 0: ref_value = 1.0
        
        norm = data / ref_value
        return np.clip(norm, 0.0, 1.0)

    rms_norm = robust_normalize(rms)
    onset_norm = robust_normalize(onset_env)

    rms_norm = rms_norm ** 2.0
    onset_norm = onset_norm ** 2.0

    raw_curve = (0.40 * rms_norm) + (0.60 * onset_norm)
    
    tempo_result, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo_result)
    
    effective_bpm = bpm
    if effective_bpm < 100.0:
        effective_bpm *= 2.0
    
    bpm_clamped = np.clip(effective_bpm, 60.0, 175.0)
    bpm_factor = 0.5 + ((bpm_clamped - 60.0) / 115.0) * 1.0
    
    avg_loudness = np.mean(rms_norm) 
    loudness_factor = 0.8 + (avg_loudness * 1.2)
    
    global_energy = bpm_factor * loudness_factor
    
    final_curve = raw_curve * global_energy
    
    final_curve = np.clip(final_curve, 0.0, 1.0)
    
    return {
        "curve": final_curve,
        "duration": librosa.get_duration(y=y, sr=sr),
        "fps": target_fps,
        "tempo": bpm
    }
