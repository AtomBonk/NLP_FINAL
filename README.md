# ````markdown

# \# Can a General-Purpose LLM Write the Other Voice?

# 

# \*\*Instruction-tuned accompaniment generation in bar-indexed ABC notation\*\*

# 

# NLP final project, Tel Aviv University, 2026.

# Amit Cohen, Shai Rotman, Yamit Kfir, Hila Etziony.

# 

# Given a complete melody written as text, can a general-purpose language model

# learn to write a matching accompaniment that follows the melody bar by bar? We

# fine-tune `meta-llama/Meta-Llama-3-8B` with QLoRA on four-part chorales: the

# soprano (melody) is the prompt and the alto (accompaniment) is the target. Both

# are written in ABC notation, with every bar tagged by voice and bar number so

# that matching bars can be lined up:

# 

# ```

# Prompt  \[V:1 B1] z3 E | \[V:1 B2] A B c d | \[V:1 B3] B3/2 A/ !fermata!G G | ...

# Target  \[V:2 B1] z3 C/D/ | \[V:2 B2] E E E D | \[V:2 B3] D/E/ ^F !fermata!E E | ...

# ```

# 

# \## Results

# 

# 384 test examples (32 held-out pieces, each in 12 keys). Full numbers in

# `logsS/eval\_947864.out`.

# 

# | | Untuned model | Untuned, output relabelled as accompaniment | Fine-tuned |

# |---|---|---|---|

# | Valid two-voice score | 0% | 63.3% | \*\*100%\*\* |

# | Repeats the melody voice | 97.1% | 0% | \*\*0%\*\* |

# | One accompaniment bar per melody bar, in order | 0% | 0% | \*\*99.7%\*\* |

# | Fréchet Audio Distance to human reference (lower is better) | n/a | 0.49 | \*\*0.011\*\* |

# | Note-sequence similarity to human accompaniment | n/a | 12.1% | \*\*34.8%\*\* |

# 

# Without fine-tuning, the model simply continues the melody. After fine-tuning

# on 3,012 examples (251 pieces in 12 keys), it writes the accompaniment in valid

# notation and follows the melody's bar structure.

# 

# \## Repository contents

# 

# | Path | Contents |

# |---|---|

# | `music\_abc\_splits/` | Final train / validation / test splits: 251 / 31 / 32 pieces (Hugging Face `datasets` format) |

# | `music\_abc\_splits\_augmented/` | The same splits transposed to 12 keys: 3,012 / 372 / 384 examples. Used for training and evaluation |

# | `{train,validation,test}\_split.json` | JSON copies of `music\_abc\_splits/` |

# | `music\_finetuned\_model/final/` | The trained LoRA adapter (84 MB) and tokenizer |

# | `merged\_results.json` | Every test prompt with its human reference, untuned output and fine-tuned output |

# | `\*.out`, `\*.err`, `logsS/` | Logs of the runs reported in the project |

# 

# \## Pipeline

# 

# Run in this order. GPU steps were run as Slurm jobs (`sbatch`) on the Tel Aviv

# University cluster.

# 

# | # | Step | Command | Output |

# |---|---|---|---|

# | 1 | Build the dataset and splits | `python prepare\_dataset.py` | `music\_abc\_splits/` |

# | 2 | Transpose every piece to 12 keys | `python pitch\_shift\_training\_test.py` | `music\_abc\_splits\_augmented/` |

# | 3 | Check sequence lengths (optional) | `python check\_length.py` | printed statistics |

# | 4 | Sanity check: overfit 8 examples | `sbatch run\_sanity\_current.sh` | `sanity\_current\_results.json` |

# | 5 | Hyperparameter search | `sbatch run\_sanity.sh` (runs `train\_sanity\_check.py`) | `grid\_search\_results.json` |

# | 6 | Untuned (zero-shot) baseline | `sbatch run\_baseline.sh` | `zero\_shot\_baseline\_results\_NEW.json` |

# | 7 | Fine-tune | `sbatch run\_full\_training.sh` | `music\_finetuned\_model/` |

# | 8 | Generate with the fine-tuned model | `sbatch run\_generate\_f.sh` | `finetuned\_model\_train\_results.json` |

# | 9 | Merge both sets of outputs | `python merge\_j.py` | `merged\_results.json` |

# | 10 | Evaluate | `sbatch run\_evaluation.sh` | `logsS/eval\_<job>.out` |

# 

# Logs of the reported runs: `finetune\_917578.\*` (training),

# `gen\_922486.\*` (generation), `zero\_shot\_922975.\*` (baseline),

# `logsS/eval\_947864.\*` (evaluation), `sanity\_current\_956324.\*` (sanity check),

# `sanity\_804442.\*` (hyperparameter search, run on an earlier version of the

# dataset).

# 

# Helper and debugging scripts: `inspect\_dataset.py`, `arrow\_to\_json.py`,

# `verify\_splits.py`, `pitch\_analysis.py`, `pitch\_shift\_fast\*.py`,

# `syntax\_debug\*.py`.

# 

# \## Using the trained model

# 

# ```python

# import torch

# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# from peft import PeftModel

# 

# base = AutoModelForCausalLM.from\_pretrained(

# &#x20;   "meta-llama/Meta-Llama-3-8B",

# &#x20;   quantization\_config=BitsAndBytesConfig(

# &#x20;       load\_in\_4bit=True, bnb\_4bit\_quant\_type="nf4",

# &#x20;       bnb\_4bit\_compute\_dtype=torch.bfloat16),

# &#x20;   device\_map="auto")

# model = PeftModel.from\_pretrained(base, "music\_finetuned\_model/final")

# tokenizer = AutoTokenizer.from\_pretrained("music\_finetuned\_model/final")

# ```

# 

# Prompts must follow the training format exactly (see any `prompt` field in

# `merged\_results.json`), followed by a newline. Decoding in the project was

# greedy (`do\_sample=False`).

# 

# \## Requirements

# 

# Not included in this repository:

# 

# \- \*\*Base model:\*\* `meta-llama/Meta-Llama-3-8B` from Hugging Face. Access is

# &#x20; gated: request it on the model page and log in with `huggingface-cli login`.

# \- \*\*Python packages:\*\* `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`,

# &#x20; `datasets`, `music21`, `muspy`, `pretty\_midi`, `frechet\_audio\_distance`.

# \- \*\*abcMIDI\*\* (`abc2midi`, `abc2abc`) on `PATH`, used for transposition and

# &#x20; for compiling outputs to MIDI.

# \- \*\*FluidSynth\*\* and the \*\*FluidR3\_GM\*\* General MIDI soundfont, used for the

# &#x20; audio metric. `evaluate.py` looks for `FluidR3\_GM.sf2` next to it or at

# &#x20; `$SOUNDFONT\_PATH`.

# 

# \## Notes

# 

# \- The scripts use absolute paths from the university cluster

# &#x20; (`/home/morg/NLP\_2526b/shairotman/NLP\_final`). Change `PERSISTENT\_BASE` and

# &#x20; similar constants to run them elsewhere. `verify\_splits.py` and

# &#x20; `run\_sanity.sh` point to a team member's directory.

# \- The scripts log in to Hugging Face with a token written in the code. Replace

# &#x20; it with your own, or remove the `login(...)` call and use

# &#x20; `huggingface-cli login`.

# \- The prompt and target headings in the data read "SMT-ABC", the

# &#x20; synchronised multi-track format (MuPT) that our representation is adapted

# &#x20; from. Our format keeps the voices in sequence and links bars by number

# &#x20; instead of interleaving them.

# ````

