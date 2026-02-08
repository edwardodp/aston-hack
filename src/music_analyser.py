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
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=None)
    
    # 2. Basic Features (Local)
    hop_length = int(sr / target_fps)
    
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # 3. Robust Normalization (The "FEIN" Fix)
    # Instead of normalizing to the absolute Max (which can be triggered by one click),
    # we normalize to the 96th percentile. This ensures the "drop" hits 1.0/max.
    def robust_normalize(data):
        if len(data) == 0: return data
        ref_value = np.percentile(data, 96) # Top 4% is considered "Max Volume"
        if ref_value <= 0: ref_value = 1.0
        
        # Scale and Clip
        norm = data / ref_value
        return np.clip(norm, 0.0, 1.0)

    rms_norm = robust_normalize(rms)
    onset_norm = robust_normalize(onset_env)

    # Apply Contrast (Squaring)
    # We still want that "snap", but now that we fixed normalization, 
    # the drop will likely be 1.0^2 = 1.0 instead of 0.4^2 = 0.16
    rms_norm = rms_norm ** 2.0
    onset_norm = onset_norm ** 2.0

    # 4. Create Base Curve
    # Re-balanced: 40% Loudness, 60% Impact.
    # Trap music relies on sustained bass (RMS) as much as the kick drum.
    raw_curve = (0.40 * rms_norm) + (0.60 * onset_norm)
    
    # --- 5. CALCULATE GLOBAL ENERGY (Adrenaline Factor) ---
    
    # A. Tempo Factor (Rowdiness Multiplier)
    tempo_result, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo_result)
    
    # FIX: Trap/Dubstep Half-time Correction
    # If BPM is low (e.g. 75), it's likely a 150 BPM Trap track.
    # We double the effective energy BPM for calculation.
    effective_bpm = bpm
    if effective_bpm < 100.0:
        effective_bpm *= 2.0
    
    # Boosted Map: 120 BPM -> 1.0x, 175 BPM -> 1.5x
    bpm_clamped = np.clip(effective_bpm, 60.0, 175.0)
    bpm_factor = 0.5 + ((bpm_clamped - 60.0) / 115.0) * 1.0
    
    # B. Intensity Factor (Wall of Sound)
    avg_loudness = np.mean(rms_norm) 
    # Map 0.1 -> 0.8, 0.5 -> 1.4 (Boosted max intensity)
    loudness_factor = 0.8 + (avg_loudness * 1.2)
    
    global_energy = bpm_factor * loudness_factor
    
    # 6. Apply Global Scaling
    final_curve = raw_curve * global_energy
    
    # 7. Post-Processing
    final_curve = np.clip(final_curve, 0.0, 1.0)
    
    return {
        "curve": final_curve,
        "duration": librosa.get_duration(y=y, sr=sr),
        "fps": target_fps,
        "tempo": bpm # Return original BPM for display, even if we doubled it internally
    }
