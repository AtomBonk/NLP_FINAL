# Can a General-Purpose LLM Write the Other Voice?

**Instruction-tuned accompaniment generation in bar-indexed ABC notation**

NLP final project, Tel Aviv University, 2026.
Amit Cohen, Shai Rotman, Yamit Kfir, Hila Etziony.

Given a complete melody written as text, can a general-purpose language model
learn to write a matching accompaniment that follows the melody bar by bar? We
fine-tune `meta-llama/Meta-Llama-3-8B` with QLoRA on four-part chorales: the
soprano (melody) is the prompt and the alto (accompaniment) is the target. Both
are written in ABC notation, with every bar tagged by voice and bar number so
that matching bars can be lined up:

```
Prompt  [V:1 B1] z3 E | [V:1 B2] A B c d | [V:1 B3] B3/2 A/ !fermata!G G | ...
Target  [V:2 B1] z3 C/D/ | [V:2 B2] E E E D | [V:2 B3] D/E/ ^F !fermata!E E | ...
```

## Results

384 test examples (32 held-out pieces, each in 12 keys). Full numbers in
`logsS/eval_947864.out`.

| | Untuned model | Untuned, output relabelled as accompaniment | Fine-tuned |
|---|---|---|---|
| Valid two-voice score | 0% | 63.3% | **100%** |
| Repeats the melody voice | 97.1% | 0% | **0%** |
| One accompaniment bar per melody bar, in order | 0% | 0% | **99.7%** |
| Fréchet Audio Distance to human reference (lower is better) | n/a | 0.49 | **0.011** |
| Note-sequence similarity to human accompaniment | n/a | 12.1% | **34.8%** |

Without fine-tuning, the model simply continues the melody. After fine-tuning
on 3,012 examples (251 pieces in 12 keys), it writes the accompaniment in valid
notation and follows the melody's bar structure.

## Repository contents

| Path | Contents |
|---|---|
| `music_abc_splits/` | Final train / validation / test splits: 251 / 31 / 32 pieces (Hugging Face `datasets` format) |
| `music_abc_splits_augmented/` | The same splits transposed to 12 keys: 3,012 / 372 / 384 examples. Used for training and evaluation |
| `{train,validation,test}_split.json` | JSON copies of `music_abc_splits/` |
| `music_finetuned_model/final/` | The trained LoRA adapter (84 MB) and tokenizer |
| `merged_results.json` | Every test prompt with its human reference, untuned output and fine-tuned output |
| `*.out`, `*.err`, `logsS/` | Logs of the runs reported in the project |

## Pipeline

Run in this order. GPU steps were run as Slurm jobs (`sbatch`) on the Tel Aviv
University cluster.

| # | Step | Command | Output |
|---|---|---|---|
| 1 | Build the dataset and splits | `python prepare_dataset.py` | `music_abc_splits/` |
| 2 | Transpose every piece to 12 keys | `python pitch_shift_training_test.py` | `music_abc_splits_augmented/` |
| 3 | Check sequence lengths (optional) | `python check_length.py` | printed statistics |
| 4 | Sanity check: overfit 8 examples | `sbatch run_sanity_current.sh` | `sanity_current_results.json` |
| 5 | Hyperparameter search | `sbatch run_sanity.sh` (runs `train_sanity_check.py`) | `grid_search_results.json` |
| 6 | Untuned (zero-shot) baseline | `sbatch run_baseline.sh` | `zero_shot_baseline_results_NEW.json` |
| 7 | Fine-tune | `sbatch run_full_training.sh` | `music_finetuned_model/` |
| 8 | Generate with the fine-tuned model | `sbatch run_generate_f.sh` | `finetuned_model_train_results.json` |
| 9 | Merge both sets of outputs | `python merge_j.py` | `merged_results.json` |
| 10 | Evaluate | `sbatch run_evaluation.sh` | `logsS/eval_<job>.out` |

Logs of the reported runs: `finetune_917578.*` (training),
`gen_922486.*` (generation), `zero_shot_922975.*` (baseline),
`logsS/eval_947864.*` (evaluation), `sanity_current_956324.*` (sanity check),
`sanity_804442.*` (hyperparameter search, run on an earlier version of the
dataset).

Helper and debugging scripts: `inspect_dataset.py`, `arrow_to_json.py`,
`verify_splits.py`, `pitch_analysis.py`, `pitch_shift_fast*.py`,
`syntax_debug*.py`.

## Using the trained model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto")
model = PeftModel.from_pretrained(base, "music_finetuned_model/final")
tokenizer = AutoTokenizer.from_pretrained("music_finetuned_model/final")
```

Prompts must follow the training format exactly (see any `prompt` field in
`merged_results.json`), followed by a newline. Decoding in the project was
greedy (`do_sample=False`).

## Requirements

Not included in this repository:

- **Base model:** `meta-llama/Meta-Llama-3-8B` from Hugging Face. Access is
  gated: request it on the model page and log in with `huggingface-cli login`.
- **Python packages:** `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`,
  `datasets`, `music21`, `muspy`, `pretty_midi`, `frechet_audio_distance`.
- **abcMIDI** (`abc2midi`, `abc2abc`) on `PATH`, used for transposition and
  for compiling outputs to MIDI.
- **FluidSynth** and the **FluidR3_GM** General MIDI soundfont, used for the
  audio metric. `evaluate.py` looks for `FluidR3_GM.sf2` next to it or at
  `$SOUNDFONT_PATH`.

## Notes

- The scripts use absolute paths from the university cluster
  (`/home/morg/NLP_2526b/shairotman/NLP_final`). Change `PERSISTENT_BASE` and
  similar constants to run them elsewhere. `verify_splits.py` and
  `run_sanity.sh` point to a team member's directory.
- The scripts log in to Hugging Face with the token in the `HF_TOKEN`
  environment variable. Set it before running them
  (`export HF_TOKEN=hf_...`).
- The prompt and target headings in the data read "SMT-ABC", the
  synchronised multi-track format (MuPT) that our representation is adapted
  from. Our format keeps the voices in sequence and links bars by number
  instead of interleaving them.
