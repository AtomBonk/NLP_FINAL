import json
import os
import sys
import shutil
import warnings
import contextlib
import uuid
import subprocess
import numpy as np
import pretty_midi
import muspy
import music21 as m21
import difflib
from unittest.mock import patch
import traceback

# Suppress all noisy warnings from libraries
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

# Locate abc2midi & fluidsynth
ABC2MIDI_PATH = os.path.expanduser("~/.local/bin/abc2midi")
if not os.path.exists(ABC2MIDI_PATH):
    ABC2MIDI_PATH = shutil.which("abc2midi")
HAS_ABC2MIDI = ABC2MIDI_PATH is not None and os.path.exists(ABC2MIDI_PATH)
HAS_FLUIDSYNTH = shutil.which("fluidsynth") is not None


def get_soundfont_path():
    """Smartly locates a standard General MIDI SoundFont on the system."""
    paths = [
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "FluidR3_GM.sf2"),
        os.environ.get("SOUNDFONT_PATH", ""),
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",          # Ubuntu/Debian standard
        "/usr/share/sounds/sf2/TimGM6mb.sf2",
        "/usr/share/soundfonts/default.sf2",
        "/opt/homebrew/share/soundfonts/FluidR3_GM.sf2"  # MacOS Homebrew
    ]
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# ==========================================
# STRING PARSING (NO REGEX)
# ==========================================

def clean_abc_text(text):
    """
    Extracts the ABC string and cleans custom dataset tags (e.g., [V:1 B18] -> [V:1])
    using pure string operations, strictly avoiding Regex.
    """
    if "### Primary Melody (SMT-ABC):" in text:
        text = text.split("### Primary Melody (SMT-ABC):")[-1]
    elif "### Accompaniment Track (SMT-ABC):" in text:
        text = text.split("### Accompaniment Track (SMT-ABC):")[-1]
        
    if "```abc" in text:
        text = text.split("```abc")[-1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[-1].split("```")[0]
        
    lines = text.split('\n')
    clean_lines = []
    conversational_starts = ("here is", "sure", "certainly", "this is", "below", "here's", "as requested")
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('###'): 
            continue
        if stripped.lower().startswith(conversational_starts): 
            continue
        clean_lines.append(line)
        
    text = '\n'.join(clean_lines)
    
    # Clean [V:X BYY] tags -> [V:X] by parsing characters
    result_chars = []
    i = 0
    while i < len(text):
        if text[i:i+3] == "[V:":
            end_idx = text.find("]", i)
            if end_idx != -1:
                inner_tag = text[i+1:end_idx] 
                parts = inner_tag.split() 
                if len(parts) > 0 and parts[0].startswith("V:"):
                    result_chars.append(f"[{parts[0]}]")
                else:
                    result_chars.append(f"[{inner_tag}]")
                i = end_idx + 1
                continue
        result_chars.append(text[i])
        i += 1
        
    return "".join(result_chars)

def count_bars_pure(abc_text):
    """Counts bars natively by splitting barline characters."""
    t = abc_text.replace("|]", "|").replace("|:", "|").replace(":|", "|").replace("::", "|")
    bars = [b for b in t.split('|') if b.strip()]
    return len(bars)


# ==========================================
# COMPILATION & MIDI LIBRARIES
# ==========================================

def compile_abc_to_midi(prompt, generated, target_column, filepath):
    clean_prompt = clean_abc_text(prompt)
    clean_gen = clean_abc_text(generated)
    full_abc = f"{clean_prompt.rstrip()}\n{clean_gen.lstrip()}"
    
    if not HAS_ABC2MIDI:
        return False, None, clean_gen
        
    abc_path = filepath + ".abc"
    mid_path = filepath + ".mid"
    
    with open(abc_path, "w", encoding="utf-8") as f:
        f.write(full_abc)
        
    subprocess.run([ABC2MIDI_PATH, abc_path, "-o", mid_path], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                   
    if os.path.exists(mid_path) and os.path.getsize(mid_path) > 0:
        try:
            pretty_midi.PrettyMIDI(mid_path)
            return True, mid_path, clean_gen
        except:
            return False, None, clean_gen
            
    return False, None, clean_gen

def check_format_compliance(clean_gen_text, mid_path):
    has_v1_leak = "[V:1]" in clean_gen_text
    has_v2_text = "[V:2]" in clean_gen_text
    
    actual_has_v2 = False
    if mid_path:
        try:
            pm = pretty_midi.PrettyMIDI(mid_path)
            actual_has_v2 = sum(1 for inst in pm.instruments if len(inst.notes) > 0) >= 2
        except:
            pass
            
    return has_v1_leak, (has_v2_text and actual_has_v2)


# ==========================================
# MUSIC THEORY METRICS (MUSIC21 & MUSPY)
# ==========================================

def calculate_rhythm_via_music21(midi_path):
    try:
        with suppress_stdout_stderr():
            score = m21.converter.parse(midi_path)
            if not score.parts: return 0.0
            
            part = score.parts[1] if len(score.parts) > 1 else score.parts[0]
            
            measures = part.getElementsByClass(m21.stream.Measure)
            if len(measures) == 0:
                part = part.makeMeasures()
                measures = part.getElementsByClass(m21.stream.Measure)
                
            if len(measures) == 0: return 0.0
            
            valid = 0
            for m in measures:
                if m.highestTime <= m.barDuration.quarterLength + 0.125:
                    valid += 1
            return valid / len(measures)
    except:
        return 0.0

def calculate_harmony_metrics(gen_mid, gt_mid):
    scale_cons = key_match = chord_sim = cons_ratio = key_compat = 0.0
    try:
        with suppress_stdout_stderr():
            pm = pretty_midi.PrettyMIDI(gen_mid)
            v2_pm = pretty_midi.PrettyMIDI()
            v2_pm.instruments.extend(pm.instruments[1:] if len(pm.instruments) > 1 else pm.instruments)
            scale_cons = float(muspy.scale_consistency(muspy.from_pretty_midi(v2_pm)))

            gen_score = m21.converter.parse(gen_mid)
            gt_score = m21.converter.parse(gt_mid)

            if len(gen_score.parts) >= 2 and len(gt_score.parts) >= 2:
                gen_v1, gen_v2 = gen_score.parts[0], gen_score.parts[1]
                gt_v2 = gt_score.parts[1]

                g_key = gen_v2.analyze('key')
                t_key = gt_v2.analyze('key')
                if g_key.tonic.name == t_key.tonic.name and g_key.mode == t_key.mode:
                    key_match = 1.0

                g_chords = [c.root().name for c in gen_v2.chordify().recurse().getElementsByClass(m21.chord.Chord)]
                t_chords = [c.root().name for c in gt_v2.chordify().recurse().getElementsByClass(m21.chord.Chord)]
                chord_sim = difflib.SequenceMatcher(None, g_chords, t_chords).ratio()

                k1 = gen_v1.analyze('key')
                if k1.tonic.name == g_key.tonic.name: key_compat += 0.7
                if k1.mode == g_key.mode: key_compat += 0.3
                else: key_compat += 0.1

                score_both = m21.stream.Score()
                score_both.insert(0, gen_v1)
                score_both.insert(0, gen_v2)
                
                chords = score_both.chordify().recurse().getElementsByClass(m21.chord.Chord)
                total_slices, cons_slices = 0, 0
                for c in chords:
                    total_slices += 1
                    if c.isTriad() or c.isSeventh() or len(c.pitches) <= 2:
                        cons_slices += 1
                if total_slices > 0:
                    cons_ratio = cons_slices / total_slices

    except Exception:
        pass
        
    return scale_cons, key_match, chord_sim, cons_ratio, key_compat


# ==========================================
# AUDIO & FAD
# ==========================================

def compute_domain_matched_fad(gen_dir, gt_dir, soundfont_path=None):
    if not HAS_FLUIDSYNTH: return "N/A ('fluidsynth' not installed)"
    gen_wavs = [f for f in os.listdir(gen_dir) if f.endswith('.wav')]
    if len(gen_wavs) < 2: return "N/A (Too few valid audio renderings)"
    
    try:
        with patch('builtins.input', return_value='y'):
            fad = None
            try:
                from frechet_audio_distance import FrechetAudioDistance
                try:
                    fad = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False, verbose=False)
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

def evaluate_model_outputs(json_path, target_column, max_samples=1500, transform_mode=None):
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
        'syntactically_valid': 0, 'structural_match': 0, 'prompt_leakage': 0, 
        'missing_accomp': 0, 'perfect_temporal': 0, 'total_bar_diff': 0,
        'rhythmic_scores': [], 'scale_cons': [], 'key_match': [], 
        'chord_sim': [], 'v1_v2_cons': [], 'v1_v2_key': []
    }

    run_id = uuid.uuid4().hex[:8]
    temp_dir = f"temp_eval_{run_id}"
    gen_wav_dir = os.path.join(temp_dir, "wav_generated")
    gt_wav_dir = "wav_ground_truth_cache" 
    
    os.makedirs(gen_wav_dir, exist_ok=True)
    os.makedirs(gt_wav_dir, exist_ok=True)
    
    soundfont_path = get_soundfont_path()

    print(f"\n[INFO] Starting Evaluation for column: {target_column}...")
    if transform_mode == "v1_to_v2":
        print("[INFO] Transform Mode Active: Converting all V:1 tags to V:2 in generated output.")

    if HAS_FLUIDSYNTH and not soundfont_path:
        print("[WARNING] Fluidsynth is installed but no SoundFont (.sf2) was found.")
        print("          Audio rendering (and FAD metric) will be skipped.")
        print("          -> Fix: Download 'FluidR3_GM.sf2' and place it in this folder.\n")

    try:
        for idx, row in enumerate(data):
            prompt = row.get("prompt", "")
            raw_gen = row.get(target_column, "")
            
            # --- הוספת המרת הקולות לפי המשתנה ---
            if transform_mode == "v1_to_v2":
                # מחליף גם את התגיות הספציפיות כדי שיהפכו לקול השני
                raw_gen = raw_gen.replace("V:1", "V:2")
            # ------------------------------------

            gt_comp = row.get("ground_truth_completion", "")

            is_valid_abc, gen_mid, pure_gen = compile_abc_to_midi(prompt, raw_gen, target_column, os.path.join(temp_dir, f"gen_{idx}"))
            gt_is_valid, gt_mid, pure_gt = compile_abc_to_midi(prompt, gt_comp, "ground_truth_completion", os.path.join(temp_dir, f"gt_{idx}"))

            has_v1_leak, has_v2 = check_format_compliance(pure_gen, gen_mid)
            
            if has_v1_leak: metrics['prompt_leakage'] += 1
            if not has_v2: metrics['missing_accomp'] += 1
            
            if "[" in pure_gen and "]" in pure_gen: metrics['structural_match'] += 1

            is_valid = is_valid_abc and has_v2
            if is_valid: metrics['syntactically_valid'] += 1

            bars_gen = count_bars_pure(pure_gen)
            bars_gt = count_bars_pure(pure_gt)
            bar_diff = abs(bars_gen - bars_gt)
            metrics['total_bar_diff'] += bar_diff
            if bar_diff == 0 and bars_gt > 0: metrics['perfect_temporal'] += 1

            if is_valid and gt_is_valid and gen_mid and gt_mid:
                rhythm = calculate_rhythm_via_music21(gen_mid)
                sc, km, cs, vc, vk = calculate_harmony_metrics(gen_mid, gt_mid)
                
                # Audio Generation
                if HAS_FLUIDSYNTH and soundfont_path:
                    gen_wav = os.path.join(gen_wav_dir, f"{idx}.wav")
                    gt_wav = os.path.join(gt_wav_dir, f"{idx}.wav")
                    
                    subprocess.run(["fluidsynth", "-ni", "-R", "0", "-T", "wav", "-F", gen_wav, "-r", "16000", soundfont_path, gen_mid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if not os.path.exists(gt_wav): 
                        subprocess.run(["fluidsynth", "-ni", "-R", "0", "-T", "wav", "-F", gt_wav, "-r", "16000", soundfont_path, gt_mid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                rhythm = sc = km = cs = vc = vk = 0.0

            metrics['rhythmic_scores'].append(rhythm)
            metrics['scale_cons'].append(sc)
            metrics['key_match'].append(km)
            metrics['chord_sim'].append(cs)
            metrics['v1_v2_cons'].append(vc)
            metrics['v1_v2_key'].append(vk)

        res = {
            'validity': (metrics['syntactically_valid'] / total_samples) * 100,
            'structure': (metrics['structural_match'] / total_samples) * 100,
            'leakage': (metrics['prompt_leakage'] / total_samples) * 100,
            'missing': (metrics['missing_accomp'] / total_samples) * 100,
            'temporal': (metrics['perfect_temporal'] / total_samples) * 100,
            'bar_diff': metrics['total_bar_diff'] / total_samples,
            'rhythm': np.mean(metrics['rhythmic_scores']) * 100 if total_samples else 0,
            'scale_cons': np.mean(metrics['scale_cons']) * 100 if total_samples else 0,
            'key_match': np.mean(metrics['key_match']) * 100 if total_samples else 0,
            'chord_sim': np.mean(metrics['chord_sim']) * 100 if total_samples else 0,
            'v1_v2_cons': np.mean(metrics['v1_v2_cons']) * 100 if total_samples else 0,
            'v1_v2_key': np.mean(metrics['v1_v2_key']) * 100 if total_samples else 0,
            'fad': compute_domain_matched_fad(gen_wav_dir, gt_wav_dir, soundfont_path)
        }

        print("\n" + "=" * 65)
        print("🎵 COMPLETE ACCOMPANIMENT EVALUATION REPORT 🎵")
        print(f"Target Column Evaluated: {target_column}")
        if transform_mode:
            print(f"Applied Transform: {transform_mode}")
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
    evaluate_model_outputs("merged_results.json", "zero_shot_generated", max_samples=384, transform_mode="v1_to_v2")
    evaluate_model_outputs("merged_results.json", "finetuned_generation", max_samples=384)