#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import fft
from scipy.signal import find_peaks, butter, filtfilt, spectrogram, welch
from scipy.signal.windows import hann

# --- Optional: Für Wavelet-Analyse (falls gewünscht) ---
try:
    import pywt
    HAVE_PYWAVELETS = True
except ImportError:
    HAVE_PYWAVELETS = False

# --- PyCBC-Module für Matched Filtering ---
import pycbc.types
import pycbc.psd
from pycbc.filter import matched_filter
from pycbc.waveform import get_td_waveform

# --------------------------
# KONSTANTEN UND EINSTELLUNGEN
# --------------------------
G = 6.67430e-11             # Gravitationskonstante (m^3 kg^-1 s^-2)
c = 3.0e8                   # Lichtgeschwindigkeit (m/s)
MSUN = 1.989e30             # Sonnenmasse (kg)
SAMPLE_RATE = 16000         # Sampling-Rate (z. B. 16 kHz)
CUTOFF = 50                 # Hochpass-Filter Cutoff (Hz)

# Einstellungen für die Segmentierung (Final Method):
SEGMENT_LENGTH = 4096       # Anzahl Samples pro Segment
OVERLAP = 0.5               # 50% Überlappung der Segmente

# Parameter für das theoretische Modell (segmentierte FFT):
N0 = 4                      # Basissegmentzahl im Vakuum
phi = 1.618                 # Goldener Schnitt (φ)
r0 = 1e-6                   # Minimaler Referenzradius, um Singularitäten zu vermeiden

# Globaler Kalibrierungsfaktor (kann angepasst werden, falls zusätzliche Kalibrierungsdaten vorliegen)
CALIBRATION_FACTOR = 1.0

# Option für Zwischenausgaben und Plotting
PLOT_RESULTS = False

# --------------------------
# UTILITY-FUNKTIONEN (gemeinsam)
# --------------------------
def highpass_filter(data, cutoff=CUTOFF, fs=SAMPLE_RATE, order=5):
    """Wendet einen Butterworth-Hochpassfilter an."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def load_strain(filename):
    """
    Lädt einspaltige Strain-Daten aus einer Textdatei.
    Es wird angenommen, dass die ersten 3 Zeilen Header sind.
    """
    print(f"[Load] Reading file: {filename}")
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_lines = [line.strip() for line in lines[3:] if line.strip()]
    strain_data = np.array([float(x) for x in data_lines])
    print(f"       -> {len(strain_data)} samples read.")
    return strain_data

def preprocess_strain(data):
    """
    Filtert die Daten hochpass, entfernt den DC-Offset und normalisiert 
    sie in den Bereich [-1,1].
    """
    filtered = highpass_filter(data)
    dc_offset = np.mean(filtered)
    filtered -= dc_offset
    max_abs = np.max(np.abs(filtered))
    if max_abs == 0:
        print("[Preprocess] Warning: max abs = 0; skipping normalization.")
        return filtered
    return filtered / max_abs

# --------------------------
# ZUSÄTZLICHE ANALYSEFUNKTIONEN
# --------------------------
def plot_spectrogram(data, fs=SAMPLE_RATE):
    """Erstellt ein Spektrogramm der Daten."""
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=256, noverlap=128)
    plt.figure(figsize=(8,4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Power [dB]')
    plt.show()

def perform_wavelet_analysis(data, fs=SAMPLE_RATE, wavelet='cmor'):
    """Führt eine kontinuierliche Wavelet-Transformation durch und plottet das Skalogramm."""
    if not HAVE_PYWAVELETS:
        print("PyWavelets not installed; skipping wavelet analysis.")
        return None
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=1/fs)
    plt.figure(figsize=(8,4))
    plt.imshow(np.abs(coefficients), extent=[0, len(data)/fs, scales[-1], scales[0]],
               cmap='jet', aspect='auto')
    plt.xlabel('Time [sec]')
    plt.ylabel('Scale')
    plt.title('Wavelet Scalogram')
    plt.colorbar(label='Magnitude')
    plt.show()
    # Rückgabe eines groben Frequenzwertes, hier z.B. der Median der skalierten Frequenzen
    freq_estimate = np.median(pywt.scale2frequency(wavelet, scales) * fs)
    return freq_estimate

def dominant_frequency_welch(data, fs=SAMPLE_RATE):
    """
    Schätzt die dominante Frequenz mittels Welch-Methode.
    """
    f, Pxx = welch(data, fs=fs, window='hann', nperseg=SEGMENT_LENGTH, noverlap=int(SEGMENT_LENGTH*OVERLAP))
    # Begrenze den relevanten Frequenzbereich:
    mask = (f > 30) & (f < 10000)
    if not np.any(mask):
        return None
    f_range = f[mask]
    Pxx_range = Pxx[mask]
    idx = np.argmax(Pxx_range)
    return f_range[idx]

def refine_peak(freq, fft_vals):
    """
    Führt eine parabolische Interpolation um den dominanten Peak der FFT durch,
    um eine sub-bin Auflösung zu erzielen.
    """
    peaks, _ = find_peaks(fft_vals, height=0.1*np.max(fft_vals))
    if len(peaks) == 0:
        return None
    # Index des höchsten Peaks
    peak_idx = peaks[np.argmax(fft_vals[peaks])]
    if peak_idx <= 0 or peak_idx >= len(fft_vals)-1:
        return freq[peak_idx]
    # Parabolische Interpolation: y = ax^2 + bx + c
    y0, y1, y2 = fft_vals[peak_idx-1], fft_vals[peak_idx], fft_vals[peak_idx+1]
    x0 = freq[peak_idx-1]
    x1 = freq[peak_idx]
    x2 = freq[peak_idx+1]
    # Lokale Interpolation: Verschiebung delta = (y0 - y2) / (2*(y0 - 2*y1 + y2))
    delta = (y0 - y2) / (2*(y0 - 2*y1 + y2))
    return x1 + delta * (x2 - x1)

def estimate_naive_snr(data, noise_fraction=0.1):
    """
    Schätzt die SNR naiv, indem der Signalpeak mit der Standardabweichung
    eines Rauschfensters verglichen wird.
    """
    N = len(data)
    noise_window = data[:int(noise_fraction * N)]
    noise_std = np.std(noise_window)
    signal_peak = np.max(np.abs(data))
    if noise_std == 0:
        return np.inf
    return signal_peak / noise_std

def downsample(data, factor):
    """Reduziert die Abtastrate der Daten um den angegebenen Faktor."""
    return data[::factor]

# --------------------------
# ALTE METHODE (Einfache FFT)
# --------------------------
def compute_fft_simple(data, pad_multiplier=1):
    """Berechnet die FFT eines Signals mit Hann-Fenster und optionaler Zero-Padding."""
    N = len(data)
    # Zero-Padding, falls gewünscht
    if pad_multiplier > 1:
        new_N = int(N * pad_multiplier)
        padded = np.zeros(new_N)
        padded[:N] = data
        data = padded
        N = new_N
    window = hann(N)
    fft_vals = np.abs(fft(data * window))
    freq = np.fft.fftfreq(N, d=1.0/SAMPLE_RATE)
    return freq[:N//2], fft_vals[:N//2]

def find_dominant_freq_simple(freq, fft_vals):
    """Findet den dominanten Frequenzpeak im FFT-Spektrum im relevanten Bereich."""
    peaks, _ = find_peaks(fft_vals, height=0.1*np.max(fft_vals))
    valid = [(freq[p], fft_vals[p]) for p in peaks if 30 < freq[p] < 10000]
    if not valid:
        return None
    return max(valid, key=lambda x: x[1])[0]

def compute_effective_freq_old(f_peak):
    """Berechnet eine effektive Frequenz als einfache Korrektur des Peakwerts."""
    if f_peak is None or f_peak <= 0:
        return None
    correction = 1 + np.log10(f_peak)
    return f_peak / correction

def process_file_old(filename):
    """Alte Methode: Berechnet BH-Masse aus einfachen FFT-Daten."""
    data = load_strain(filename)
    norm_data = preprocess_strain(data)
    freq, fft_vals = compute_fft_simple(norm_data, pad_multiplier=1)
    # Nutze Sub-Peak-Interpolation zur Verfeinerung
    f_peak = refine_peak(freq, fft_vals)
    if f_peak is None or f_peak <= 0:
        print("[Old Method] No valid dominant frequency found.")
        return None
    f_eff = compute_effective_freq_old(f_peak)
    print(f"[Old Method] f_peak = {f_peak:.2f} Hz, effective freq = {f_eff:.2f} Hz")
    factor = 0.3737 * c**3 / (2*np.pi*G)
    M_kg = factor / f_eff
    bh_mass = M_kg / MSUN
    print(f"[Old Method] Estimated BH mass = {bh_mass:.2f} Msun")
    return bh_mass

# --------------------------
# FINAL METHODE (Segmentierte FFT mit Dynamischer Kalibrierung)
# --------------------------
def segment_signal(data, segment_length=SEGMENT_LENGTH, overlap=OVERLAP):
    """
    Segmentiert das Signal in überlappende Abschnitte.
    """
    step = int(segment_length * (1 - overlap))
    segments = [data[i:i+segment_length] for i in range(0, len(data)-segment_length+1, step)]
    print(f"[Final Method] Segmented data into {len(segments)} segments.")
    return segments

def compute_fft_segment(segment, pad_multiplier=1):
    """Berechnet die FFT für ein Segment mit Hann-Fenster und optionalem Zero-Padding."""
    N = len(segment)
    if pad_multiplier > 1:
        new_N = int(N * pad_multiplier)
        padded = np.zeros(new_N)
        padded[:N] = segment
        segment = padded
        N = new_N
    window = hann(N)
    fft_vals = np.abs(fft(segment * window))
    freq = np.fft.fftfreq(N, d=1.0/SAMPLE_RATE)
    return freq[:N//2], fft_vals[:N//2]

def find_dominant_frequency(freq, fft_vals):
    """Sucht den dominanten Frequenzpeak in einem Segment."""
    peaks, _ = find_peaks(fft_vals, height=0.1*np.max(fft_vals))
    valid = [(freq[p], fft_vals[p]) for p in peaks if 30 < freq[p] < 10000]
    if not valid:
        return None
    # Sub-Peak-Interpolation
    return refine_peak(freq, fft_vals)

def compute_exact_radius(segment):
    """
    Berechnet einen effektiven Radius (r_eff) aus dem Segment.
    Nutzt den Quotienten A_std/A_mean als dynamischen Kalibrierungsfaktor.
    """
    A_mean = np.mean(np.abs(segment))
    A_std = np.std(segment)
    if A_mean == 0:
        return r0
    lam = A_std / A_mean
    k_eff = (2 * np.log(phi) / np.pi) * lam
    theta_eff = A_std
    return r0 * np.exp(k_eff * theta_eff)

def compute_effective_frequency_final(f_peak, segment):
    """
    Berechnet die effektive Frequenz für ein Segment:
      - Bestimmt r_eff und daraus N_r (dynamisch berechnete Segmentierungszahl)
      - f_eff = f_peak / N_r, kalibriert mit CALIBRATION_FACTOR
    """
    if f_peak is None or f_peak <= 0:
        return None
    r_eff = compute_exact_radius(segment)
    N_r = N0 + (2 * np.log(phi) / np.pi) * np.log(r_eff / r0)
    f_eff = f_peak / N_r
    # Einbeziehen des Kalibrierungsfaktors
    f_eff_calibrated = f_eff * CALIBRATION_FACTOR
    print(f"   >> f_peak = {f_peak:.2f} Hz, r_eff = {r_eff:.2e}, N_r = {N_r:.2f} => f_eff = {f_eff_calibrated:.2f} Hz")
    return f_eff_calibrated

def fft_pipeline_final(filename, pad_multiplier=1):
    """
    Führt die segmentierte FFT-Analyse durch:
      - Lädt und verarbeitet die Strain-Daten
      - Segmentiert das Signal
      - Berechnet für jedes Segment die effektive Frequenz (optionale Zero-Padding)
      - Nutzt zusätzlich die Welch-Methode als alternative Frequenzschätzung
      - Ermittelt einen gewichteten Durchschnitt der effektiven Frequenzen
      - Berechnet daraus eine BH-Masse.
    """
    data = load_strain(filename)
    norm_data = preprocess_strain(data)
    segments = segment_signal(norm_data)
    eff_freqs = []
    weights = []
    for idx, seg in enumerate(segments):
        freq, fft_vals = compute_fft_segment(seg, pad_multiplier=pad_multiplier)
        # Hauptansatz: FFT-Peak mit Sub-Peak-Interpolation
        f_peak = find_dominant_frequency(freq, fft_vals)
        # Alternative: Welch-Methode
        f_welch = dominant_frequency_welch(seg)
        # Alternative: Wavelet-basierte Schätzung (falls verfügbar)
        f_wavelet = perform_wavelet_analysis(seg) if HAVE_PYWAVELETS else None
        # Kombiniere die Ergebnisse (z.B. Mittelwert von verfügbaren Schätzungen)
        estimates = [f for f in [f_peak, f_welch, f_wavelet] if f is not None]
        if not estimates:
            f_eff = 0
        else:
            avg_f_obs = np.mean(estimates)
            f_eff = compute_effective_frequency_final(avg_f_obs, seg)
        weight = np.sum(seg**2)
        weights.append(weight)
        eff_freqs.append(f_eff if f_eff is not None else 0)
        if PLOT_RESULTS:
            plt.figure()
            plt.plot(freq, fft_vals, label=f"Segment {idx+1}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("FFT Amplitude")
            plt.title(f"FFT for Segment {idx+1}")
            plt.legend()
            plt.grid(True)
            plt.show()
    eff_arr = np.array(eff_freqs)
    w_arr = np.array(weights)
    if np.all(eff_arr <= 0):
        print("[Final Method] No valid effective frequencies found.")
        return None
    avg_f_eff = np.sum(eff_arr * w_arr) / np.sum(w_arr)
    print(f"[Final Method] Weighted average effective frequency: {avg_f_eff:.2f} Hz")
    factor = 0.3737 * c**3 / (2*np.pi*G)
    M_kg = factor / avg_f_eff
    bh_mass = M_kg / MSUN
    print(f"[Final Method] Estimated BH mass = {bh_mass:.2f} Msun")
    return bh_mass

# --------------------------
# MATCHED FILTERING PIPELINE (Mit verfeinerter Suche)
# --------------------------
def adjust_template_length(template, target_length):
    """
    Passt die Länge der Vorlage (Template) an die Ziel-Länge an,
    indem sie entweder mit Nullen aufgefüllt oder gekürzt wird.
    """
    current_length = len(template)
    if current_length < target_length:
        pad = target_length - current_length
        padded_array = np.concatenate((template, np.zeros(pad)))
        return pycbc.types.TimeSeries(padded_array, delta_t=template.delta_t)
    elif current_length > target_length:
        return template[:target_length]
    else:
        return template

def matched_filtering_pipeline(filename):
    """
    Nutzt die einspaltigen Strain-Daten (mit generiertem Zeitarray) und führt 
    ein minimales Matched Filtering mit IMRPhenomD-Waveformen durch.
    Zuerst wird ein grobes Massenraster abgearbeitet (5 bis 100 Msun), dann erfolgt eine
    Feinabstimmung im Bereich um den bisher besten Wert.
    """
    try:
        data = load_strain(filename)
    except Exception as e:
        print(f"[Matched Filtering] Error loading file: {e}")
        return None
    N = len(data)
    dt = 1.0 / SAMPLE_RATE
    time = np.arange(N) * dt
    strain_ts = pycbc.types.TimeSeries(data, delta_t=dt)
    psd = pycbc.psd.aLIGOZeroDetHighPower(len(strain_ts)//2+1, strain_ts.delta_f, 20.0)
    best_snr = 0
    best_mass = None
    # Grobes Raster: 5 bis 100 Msun
    masses = np.linspace(5, 100, 10)
    print("[Matched Filtering] Coarse search for best mass between 5 and 100 Msun...")
    for m in masses:
        hp, hc = get_td_waveform(approximant="IMRPhenomD",
                                 mass1=m, mass2=m,
                                 spin1z=0, spin2z=0,
                                 f_lower=20.0,
                                 delta_t=dt)
        template = adjust_template_length(hp, len(strain_ts))
        snr = matched_filter(template, strain_ts, psd=psd, low_frequency_cutoff=20.0)
        peak_snr = abs(snr).numpy().max()
        print(f"Mass = {m:.1f} Msun -> peak SNR = {peak_snr:.2f}")
        if peak_snr > best_snr:
            best_snr = peak_snr
            best_mass = m
    # Feinabstimmung um den besten Massenwert
    if best_mass is not None:
        fine_masses = np.linspace(max(5, best_mass-2), min(100, best_mass+2), 20)
        for m in fine_masses:
            hp, hc = get_td_waveform(approximant="IMRPhenomD",
                                     mass1=m, mass2=m,
                                     spin1z=0, spin2z=0,
                                     f_lower=20.0,
                                     delta_t=dt)
            template = adjust_template_length(hp, len(strain_ts))
            snr = matched_filter(template, strain_ts, psd=psd, low_frequency_cutoff=20.0)
            peak_snr = abs(snr).numpy().max()
            if peak_snr > best_snr:
                best_snr = peak_snr
                best_mass = m
    if best_mass is not None:
        print(f"[Matched Filtering] Best mass = {best_mass:.2f} Msun (SNR = {best_snr:.2f})")
    else:
        print("[Matched Filtering] No valid mass found.")
    return best_mass

# --------------------------
# HAUPTPROGRAMM: Verarbeitung einer Datei (beide Methoden und Kombination)
# --------------------------
def process_ligo_file(filename):
    print(f"\nProcessing file: {filename}")
    
    # Final Method: Segmentierte FFT-Analyse mit optionalem Zero-Padding
    print("\n--- Running Final Method ---")
    m_final = fft_pipeline_final(filename, pad_multiplier=2)  # pad_multiplier=2 für höhere Frequenzauflösung
    
    # Matched Filtering Pipeline mit verfeinerter Suche
    print("\n--- Running Matched Filtering ---")
    m_match = matched_filtering_pipeline(filename)
    
    # Kombination: Berechnung der kombinierten Chirp-Masse
    if m_final is not None and m_match is not None:
        chirp_mass = (m_final * m_match)**(3/5) / (m_final + m_match)**(1/5)
        print(f"\n[Combined Chirp Mass] m_final = {m_final:.2f} Msun, m_match = {m_match:.2f} Msun => Chirp mass = {chirp_mass:.2f} Msun")
    else:
        chirp_mass = None
        print("\n[Combined] Could not compute chirp mass (missing one or both BH masses).")
    
    # Zusätzliche Analyse: Naive SNR-Schätzung
    data = load_strain(filename)
    snr_est = estimate_naive_snr(data)
    print(f"[Extra] Naive SNR estimate: {snr_est:.2f}")
    
    # Speichern der Ergebnisse in einer Datei
    outname = f"result_{filename}.txt"
    with open(outname, 'w') as f:
        f.write(f"File: {filename}\n\n")
        if m_final is not None:
            f.write(f"[Final Method]\nBH mass (m_final) = {m_final:.4f} Msun\n")
        else:
            f.write("[Final Method] No BH mass computed.\n")
        if m_match is not None:
            f.write(f"\n[Matched Filtering]\nBH mass = {m_match:.4f} Msun\n")
        else:
            f.write("\n[Matched Filtering] No BH mass computed.\n")
        if chirp_mass is not None:
            f.write(f"\n[Combined Chirp Mass]\nm_final = {m_final:.4f} Msun, m_match = {m_match:.4f} Msun\nChirp mass = {chirp_mass:.4f} Msun\n")
        else:
            f.write("\nCombined chirp mass could not be computed.\n")
        f.write(f"\n[Extra]\nNaive SNR estimate: {snr_est:.2f}\n")
    print(f"-> Results saved to '{outname}'.\n")

def process_all_txt():
    print("\n[Bulk Processing] Scanning for .txt files in current directory...\n")
    for fname in os.listdir('.'):
        if fname.endswith('.txt') and not fname.startswith(('normalized_', 'result_')):
            process_ligo_file(fname)

if __name__ == "__main__":
    process_all_txt()
