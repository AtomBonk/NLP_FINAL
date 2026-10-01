import json
import os
import sys
import re
import shutil
import warnings
import contextlib
import uuid
import subprocess
import numpy as np
import pretty_midi
import muspy
import music21 as m21
from fractions import Fraction
from difflib import SequenceMatcher
from unittest.mock import patch

# Suppress all noisy warnings
warnings.filterwarnings("ignore")
m21.environment.UserSettings()['warnings'] = 0

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Silences noisy prints from external libraries."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Locate abc2midi robustly
ABC2MIDI_PATH = os.path.expanduser("~/.local/bin/abc2midi")
if not os.path.exists(ABC2MIDI_PATH):
    ABC2MIDI_PATH = shutil.which("abc2midi")
HAS_ABC2MIDI = ABC2MIDI_PATH is not None and os.path.exists(ABC2MIDI_PATH)
HAS_FLUIDSYNTH = shutil.which("fluidsynth") is not None

def clean_v_tags(text):
    """Cleans artifact tags like [V:1 B18] to standard [V:1]"""
    return re.sub(r'\[V:(\d+)\s*B\d+\]', r'[V:\1]', text)

def extract_clean_abc(prompt_text):
    if "### Primary Melody (SMT-ABC):" in prompt_text:
        text = prompt_text.split("### Primary Melody (SMT-ABC):")[-1].strip()
    else:
        text = prompt_text
    return clean_v_tags(text)

def clean_llm_output(text, target_column):
    # Split to get only the accompaniment if needed
    if target_column != "zero_shot_generated":
        split_marker = "### Accompaniment Track (SMT-ABC):"
        if split_marker in text:
            text = text.split(split_marker)[-1]
            
    match = re.search(r'```(?:abc)?(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    
    clean_lines = []
    for line in text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('###'):
            continue
        if re.match(r'^(here is|sure|certainly|this is|below|here\'s|as requested)\b', line_stripped, re.IGNORECASE):
            continue
        clean_lines.append(line)
        
    clean_text = '\n'.join(clean_lines)
    return clean_v_tags(clean_text)

def build_compilable_abc(prompt, generated, target_column):
    clean_prompt = extract_clean_abc(prompt)
    clean_gen = clean_llm_output(generated, target_column)
    combined = f"{clean_prompt.rstrip()}\n{clean_gen.lstrip()}"
    return combined, clean_gen

def check_syntactic_validity_abc2midi(abc_str, pure_gen_text, filepath):
    if not HAS_ABC2MIDI:
        return False, None
        
    if pure_gen_text.count('[') != pure_gen_text.count(']'):
        return False, None
        
    with open(filepath + ".abc", "w", encoding="utf-8") as f:
        f.write(abc_str)
        
    result = subprocess.run([ABC2MIDI_PATH, filepath + ".abc", "-o", filepath + ".mid"], 
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore")
                   
    if "Error in line" in result.stderr or "Fatal error" in result.stderr:
        return False, None
        
    midi_path = filepath + ".mid"
    if os.path.exists(midi_path) and os.path.getsize(midi_path) > 0:
        return True, midi_path
            
    return False, None

def verify_midi_has_multiple_tracks(midi_path):
    """Ensures abc2midi actually compiled V:2 and didn't silently drop it."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        tracks_with_notes = sum(1 for inst in pm.instruments if len(inst.notes) > 0)
        return tracks_with_notes >= 2
    except:
        return False

def calculate_time_signature_compliance(abc_string):
    l_match = re.search(r'^L:\s*(\d+/\d+)', abc_string, re.MULTILINE)
    base_fraction = Fraction(l_match.group(1)) if l_match else Fraction(1, 8)
    
    m_match = re.search(r'^M:\s*(\d+/\d+)', abc_string, re.MULTILINE)
    target_measure = Fraction(m_match.group(1)) if m_match else Fraction(4, 4)
    if re.search(r'^M:\s*C\s*$', abc_string, re.MULTILINE): target_measure = Fraction(4, 4)

    bars = []
    for line in abc_string.split('\n'):
        line = line.strip()
        # Skip header lines but KEEP lines like |: or :|
        if re.match(r'^[a-zA-Z]:', line): 
            continue
            
        line = re.sub(r'\[V:\d+\]', '', line) # Remove voice tags
        line = re.sub(r'".*?"', '', line)     # Remove string annotations
        line = re.sub(r'!.*?!', '', line)     # Remove ABC annotations like !fermata!
        
        # Clean bar lines and split
        clean_line = re.sub(r'\|[:\]]|\[:|::', '|', line)
        bars.extend([b.strip() for b in clean_line.split('|') if b.strip()])

    if not bars: return 0.0
    
    valid_bars = 0
    for bar in bars:
        duration_sum = Fraction(0)
        
        # Simplify chords: Extract only the first note to calculate duration
        def _simplify_chords(m):
            chord_content = m.group(1)
            note_match = re.search(r'[a-gA-GzZ][\',]*(\d*(?:/\d+)?)', chord_content)
            return note_match.group(0) if note_match else ''
            
        bar_no_chords = re.sub(r'\[(.*?)\]', _simplify_chords, bar)
        notes = re.findall(r'[a-gA-GzZ][\',]*(\d*(?:/\d+)?)', bar_no_chords)
        
        for note_len in notes:
            if not note_len: mult = Fraction(1)
            elif note_len == '/': mult = Fraction(1, 2)
            elif '/' in note_len:
                parts = note_len.split('/')
                num = int(parts[0]) if parts[0] else 1
                den = int(parts[1]) if len(parts)>1 and parts[1] else 2
                mult = Fraction(num, den)
            else: mult = Fraction(int(note_len))
            duration_sum += mult * base_fraction
            
        if duration_sum <= target_measure + Fraction(1, 16):
            valid_bars += 1
            
    return valid_bars / len(bars)

def count_bars_robust(abc_text):
    clean_text = re.sub(r'\|[:\]]|\[:|::', '|', abc_text)
    bars = [b for b in clean_text.split('|') if b.strip()]
    return len(bars)

# ==========================================
# MUSIC THEORY METRICS (MUSPY & MUSIC21)
# ==========================================

def get_v2_pretty_midi(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    v2_pm = pretty_midi.PrettyMIDI()
    v2_instruments = pm.instruments[1:] if len(pm.instruments) > 1 else pm.instruments
    v2_pm.instruments.extend(v2_instruments)
    return v2_pm

def calculate_muspy_scale_consistency(gen_midi_path):
    try:
        with suppress_stdout_stderr():
            v2_pm = get_v2_pretty_midi(gen_midi_path)
            music = muspy.from_pretty_midi(v2_pm)
            return float(muspy.scale_consistency(music))
    except:
        return 0.0

def calculate_music21_harmonic_metrics(gen_midi_path, gt_midi_path):
    try:
        with suppress_stdout_stderr():
            gen_stream = m21.converter.parse(gen_midi_path)
            gt_stream = m21.converter.parse(gt_midi_path)
            
            # Ensure both have Accompaniment track to avoid comparing Melody to Accompaniment
            if len(gen_stream.parts) < 2 or len(gt_stream.parts) < 2:
                return 0.0, 0.0
                
            gen_part = gen_stream.parts[1]
            gt_part = gt_stream.parts[1]
            
            # Key Match V2 vs V2
            gen_key = gen_part.analyze('key')
            gt_key = gt_part.analyze('key')
            key_match = 1.0 if (gen_key.tonic.name == gt_key.tonic.name and gen_key.mode == gt_key.mode) else 0.0
            
            # Chord Progression Similarity V2 vs V2
            gen_chords = gen_part.chordify().flatten().getElementsByClass(m21.chord.Chord)
            gt_chords = gt_part.chordify().flatten().getElementsByClass(m21.chord.Chord)
            
            gen_roots = [c.root().name for c in gen_chords]
            gt_roots = [c.root().name for c in gt_chords]
            chord_sim = SequenceMatcher(None, gen_roots, gt_roots).ratio()
            
            return float(key_match), float(chord_sim)
    except:
        return 0.0, 0.0

def calculate_music21_melody_accompaniment_fit(gen_midi_path):
    try:
        with suppress_stdout_stderr():
            stream = m21.converter.parse(gen_midi_path)
            if len(stream.parts) < 2:
                return 0.0, 0.0
                
            v1_part = stream.parts[0]
            v2_part = stream.parts[1]
            
            # 1. Key Compatibility between Melody and Accompaniment
            v1_key = v1_part.analyze('key')
            v2_key = v2_part.analyze('key')
            
            key_compat = 0.0
            if v1_key.tonic.name == v2_key.tonic.name:
                key_compat += 0.7
            if v1_key.mode == v2_key.mode:
                key_compat += 0.3
            else:
                key_compat += 0.1
                
            # 2. Vertical Consonance
            score = m21.stream.Score()
            score.insert(0, v1_part)
            score.insert(0, v2_part)
            chordified = score.chordify()
            
            total_slices = 0
            consonant_slices = 0
            
            for c in chordified.recurse().getElementsByClass(m21.chord.Chord):
                total_slices += 1
                if c.isTriad() or c.isSeventh() or len(c.pitches) <= 2:
                    consonant_slices += 1
                    
            consonance_ratio = (consonant_slices / total_slices) if total_slices > 0 else 0.0
            
            return float(consonance_ratio), float(key_compat)
    except:
        return 0.0, 0.0

# ==========================================

def compute_domain_matched_fad(gen_dir, gt_dir):
    if not HAS_FLUIDSYNTH: return "N/A ('fluidsynth' not installed)"
    gen_wavs = [f for f in os.listdir(gen_dir) if f.endswith('.wav')]
    if len(gen_wavs) < 2: return "N/A (Too few valid audio renderings)"
    
    try:
        with patch('builtins.input', return_value='y'):
            fad = None
            try:
                from frechet_audio_distance import FrechetAudioDistance
                try: fad = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False, verbose=False)
                except TypeError:
                    try: fad = FrechetAudioDistance(model_name="vggish")
                    except TypeError: fad = FrechetAudioDistance()
            except ImportError:
                pass
            
            if fad is None:
                try:
                    from fadtk.fad import FrechetAudioDistance
                    from fadtk.model import VGGish
                    try: fad = FrechetAudioDistance(VGGish(), audio_load_worker=1)
                    except TypeError: fad = FrechetAudioDistance(VGGish())
                except ImportError:
                    pass
                    
            if fad is None: return "N/A (No suitable FAD library found in environment)"
            
            score = fad.score(gt_dir, gen_dir)
            return f"{float(score):.4f}"
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# MAIN EVALUATION PIPELINE
# ==========================================

def evaluate_model_outputs(json_path, target_column, max_samples=1500):
    if not HAS_ABC2MIDI:
        print("CRITICAL ERROR: 'abc2midi' is not installed! Check path or install it.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict): data = [data]
    data = data[:max_samples]
    total_samples = len(data)

    if total_samples == 0: return

    metrics = {
        'syntactically_valid': 0,
        'structural_match': 0,
        'prompt_leakage': 0,
        'missing_accomp': 0,
        'perfect_temporal': 0,
        'total_bar_diff': 0,
        'rhythmic_scores': [],
        'scale_consistency_scores': [],
        'key_match_scores': [],
        'chord_sim_scores': [],
        'v1_v2_consonance_scores': [],
        'v1_v2_key_compat_scores': []
    }

    run_id = uuid.uuid4().hex[:8]
    temp_dir = f"temp_evaluation_workspace_{run_id}"
    gen_wav_dir = os.path.join(temp_dir, "wav_generated")
    gt_wav_dir = "wav_ground_truth_cache" # Shared cache for Ground Truth WAVs to save compute time
    
    os.makedirs(gen_wav_dir, exist_ok=True)
    os.makedirs(gt_wav_dir, exist_ok=True)
    
    soundfont_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "FluidR3_GM.sf2")
    if not os.path.exists(soundfont_path):
        soundfont_path = os.environ.get("SOUNDFONT_PATH", "default_soundfont.sf2")

    print(f"\n[INFO] Starting Evaluation for column: {target_column}...")

    try:
        for idx, row in enumerate(data):
            prompt = row.get("prompt", "")
            gt_comp = row.get("ground_truth_completion", "")
            raw_gen = row.get(target_column, "")

            abc_gen_full, pure_gen_text = build_compilable_abc(prompt, raw_gen, target_column)
            abc_gt_full, pure_gt_text = build_compilable_abc(prompt, gt_comp, "ground_truth_completion")

            # 1. ABC Compilation Success
            is_abc_valid, gen_midi = check_syntactic_validity_abc2midi(abc_gen_full, pure_gen_text, os.path.join(temp_dir, f"gen_{idx}"))
            gt_abc_valid, gt_midi = check_syntactic_validity_abc2midi(abc_gt_full, pure_gt_text, os.path.join(temp_dir, f"gt_{idx}"))

            # 2. Format Compliance
            has_v2_text = "V:2" in pure_gen_text
            has_v1_leak = "V:1" in pure_gen_text
            
            # Verify actual compiled tracks in MIDI to prevent false positives
            has_v2 = has_v2_text and (verify_midi_has_multiple_tracks(gen_midi) if gen_midi else False)
            
            if has_v1_leak: metrics['prompt_leakage'] += 1
            if not has_v2: metrics['missing_accomp'] += 1
            if pure_gen_text.count('[') == pure_gen_text.count(']') and pure_gen_text.count('[') > 0: 
                metrics['structural_match'] += 1

            is_valid = is_abc_valid and has_v2
            gt_is_valid = gt_abc_valid and ("V:2" in pure_gt_text or "V:2" in abc_gt_full)

            if is_valid: metrics['syntactically_valid'] += 1

            # Time (Bars)
            bars_gen = count_bars_robust(pure_gen_text)
            bars_gt = count_bars_robust(pure_gt_text)
            bar_diff = abs(bars_gen - bars_gt)
            metrics['total_bar_diff'] += bar_diff
            if bar_diff == 0 and bars_gt > 0: metrics['perfect_temporal'] += 1

            # MUSIC THEORY, HARMONY & RHYTHM METRICS (Appends 0.0 if failed, eliminating Survivor Bias)
            if is_valid and gt_is_valid and gen_midi and gt_midi:
                scale_cons = calculate_muspy_scale_consistency(gen_midi)
                key_match, chord_sim = calculate_music21_harmonic_metrics(gen_midi, gt_midi)
                v1_v2_cons, v1_v2_key = calculate_music21_melody_accompaniment_fit(gen_midi)
                
                if HAS_FLUIDSYNTH and os.path.exists(soundfont_path):
                    gen_wav_path = os.path.join(gen_wav_dir, f"{idx}.wav")
                    gt_wav_path = os.path.join(gt_wav_dir, f"{idx}.wav")
                    
                    # Generate GT WAV only if not already cached
                    if not os.path.exists(gt_wav_path):
                        subprocess.run(["fluidsynth", "-ni", "-R", "0", "-T", "wav", "-F", gt_wav_path, "-r", "16000", soundfont_path, gt_midi], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Always generate Gen WAV
                    subprocess.run(["fluidsynth", "-ni", "-R", "0", "-T", "wav", "-F", gen_wav_path, "-r", "16000", soundfont_path, gen_midi], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                scale_cons, key_match, chord_sim, v1_v2_cons, v1_v2_key = 0.0, 0.0, 0.0, 0.0, 0.0

            metrics['rhythmic_scores'].append(calculate_time_signature_compliance(abc_gen_full) if is_valid else 0.0)
            metrics['scale_consistency_scores'].append(scale_cons)
            metrics['key_match_scores'].append(key_match)
            metrics['chord_sim_scores'].append(chord_sim)
            metrics['v1_v2_consonance_scores'].append(v1_v2_cons)
            metrics['v1_v2_key_compat_scores'].append(v1_v2_key)

        res = {
            'validity': (metrics['syntactically_valid'] / total_samples) * 100,
            'structure': (metrics['structural_match'] / total_samples) * 100,
            'leakage': (metrics['prompt_leakage'] / total_samples) * 100,
            'missing': (metrics['missing_accomp'] / total_samples) * 100,
            'temporal': (metrics['perfect_temporal'] / total_samples) * 100,
            'bar_diff': metrics['total_bar_diff'] / total_samples,
            'rhythm': np.mean(metrics['rhythmic_scores']) * 100 if metrics['rhythmic_scores'] else 0.0,
            'scale_cons': np.mean(metrics['scale_consistency_scores']) * 100 if metrics['scale_consistency_scores'] else 0.0,
            'key_match': np.mean(metrics['key_match_scores']) * 100 if metrics['key_match_scores'] else 0.0,
            'chord_sim': np.mean(metrics['chord_sim_scores']) * 100 if metrics['chord_sim_scores'] else 0.0,
            'v1_v2_cons': np.mean(metrics['v1_v2_consonance_scores']) * 100 if metrics['v1_v2_consonance_scores'] else 0.0,
            'v1_v2_key': np.mean(metrics['v1_v2_key_compat_scores']) * 100 if metrics['v1_v2_key_compat_scores'] else 0.0,
            'fad': compute_domain_matched_fad(gen_wav_dir, gt_wav_dir)
        }

        print("\n" + "=" * 65)
        print("🎵 COMPLETE ACCOMPANIMENT EVALUATION REPORT 🎵")
        print(f"Target Column Evaluated: {target_column}")
        print("=" * 65)
        print(f"Total Evaluated Samples: {total_samples}")
        print("-" * 65)
        print("1. [SYNTAX] Task Validity (Compiles + Has V:2): {validity:.2f}%".format(**res))
        print("2. [FORMAT] Prompt Leakage (Model wrote V:1): {leakage:.2f}%  <-- (Lower is better)".format(**res))
        print("3. [FORMAT] Missing Accompaniment Track:      {missing:.2f}%  <-- (Lower is better)".format(**res))
        print("4. [TIME] Exact Bar Alignment Rate:           {temporal:.2f}%".format(**res))
        print("5. [TIME] Avg Bar Count Deviation:            {bar_diff:.2f} bars".format(**res))
        print("6. [TIME] Time Signature Compliance:          {rhythm:.2f}%".format(**res))
        print("-" * 65)
        print("7. [HARMONY] Scale Consistency (MusPy):       {scale_cons:.2f}%".format(**res))
        print("8. [HARMONY] Key Match V2 vs V2 (music21):    {key_match:.2f}%".format(**res))
        print("9. [HARMONY] Chord Sim V2 vs V2 (music21):    {chord_sim:.2f}%".format(**res))
        print("10.[HARMONY] V1-V2 Vertical Consonance:       {v1_v2_cons:.2f}%".format(**res))
        print("11.[HARMONY] V1-V2 Key Compatibility:         {v1_v2_key:.2f}%".format(**res))
        print("-" * 65)
        print("12.[AUDIO] Domain-Matched FAD Score:          {fad}".format(**res))
        print("=" * 65)
        
    finally:
        if os.path.exists(temp_dir):
            print(f"[INFO] Cleaning up temporary workspace: {temp_dir} ...")
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("[INFO] Cleanup complete.\n")

if __name__ == "__main__":
    evaluate_model_outputs("merged_results.json", "ground_truth_completion", max_samples=384)
    evaluate_model_outputs("merged_results.json", "zero_shot_generated", max_samples=384)
    evaluate_model_outputs("merged_results.json", "finetuned_generation", max_samples=384)