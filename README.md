CivBot - Mistral 7B fine-tune for Civ VI leader dialogue
==========================================================

Fine-tunes mistralai/Mistral-7B-Instruct-v0.3 on synthetic and authentic
Civ VI leader dialogue using QLoRA. Train on a free Colab T4, Kaggle 2x T4, run locally on 
CPU via Ollama or llama.cpp.

Repo Links
-----------
https://github.com/gavbraz/GenAI-Civ_Bot
https://huggingface.co/gavbraz/Civbot/upload/main     (trained model download)


Files
-----
  civ6_combined_training.jsonl  3,846 Alpaca-style training examples
                                (2,618 unique + 1,228 canonical duplicated 2x
                                to weight Firaxis over synthetic data)
  civbot_colab.ipynb            Run on Colab to train, and then export (modifications were made to run on Kaggle)
  train.py                      Local training if you have a better pc than me (needs >=8 GB VRAM)
  inference.py                  Main Thang. Python REPL / one-shot inference
  merge_adapter.py              Bake the LoRA adapter into the base model
                                (required step before GGUF/Ollama export)
  scrape_canonical_quotes.py    Pulls canonical Firaxis quotes from the
                                Fandom wiki via MediaWiki API
  eval.py                       Score the model on a holdout set
  requirements.txt              Python dependencies


Quick start - download the trained model (no training needed)
-------------------------------------------------------------
The fine-tuned model is published on Hugging Face Hub as a 4-bit GGUF.
If you only want to *use* it (not retrain), skip every "train" step
below and pull it directly into Ollama:

  ollama pull hf.co/<gavbraz>/civbot-q4_k_m:Q4_K_M
  ollama run hf.co/<gavbraz>/civbot-q4_k_m:Q4_K_M

Or grab the raw GGUF file with the HF CLI:

  pip install huggingface_hub
  huggingface-cli download <gavbraz>/civbot-q4_k_m \
      civbot-q4_k_m.gguf --local-dir .

  # then create a Modelfile next to it (see Step 2 below) and:
  ollama create civbot -f Modelfile
  ollama run civbot

The model is ~4 GB. First-time download takes a few minutes; after that
inference is local and offline.

Step 1 - Train on Colab
-----------------------
  1. Upload civbot_colab.ipynb to https://colab.research.google.com.
  2. Runtime -> Change runtime type -> T4 GPU.
  3. Accept the Mistral license at
     https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and
     create a read token at https://huggingface.co/settings/tokens.
  4. Runtime -> Run all. Paste the token when prompted, upload the
     dataset when prompted.
  5. Wait
  6. Download civbot-q4_k_m.gguf (~4 GB). Use the Drive cell if the
     browser download stalls.


Merge step (what merge_adapter.py does)
------------------------------------------------
Training produces a small LoRA adapter (~40 MB) that sits *on top of*
the base Mistral model. Two consumers, two ways to load:

  * Python + transformers + PEFT  ->  uses base + adapter together
                                      (this is what inference.py does)
  * Ollama / llama.cpp / GGUF     ->  needs the adapter PERMANENTLY
                                      baked into the base weights

The colab/kaggle notebook handles this automatically (cell 7 calls
merge_and_unload before the GGUF conversion). If you're training
elsewhere and want to do it manually:

  python merge_adapter.py --adapter civbot-lora/final --output civbot-merged

This takes the 14 GB Mistral base in fp16 + your 40 MB adapter and
writes a single 14 GB merged HF model directory, which is what the
GGUF converter then consumes. You can skip merging only if you plan to
serve the model through Python (and accept that 4 GB VRAM won't hold
Mistral-7B, so you'll fall back to slow CPU inference through
transformers


Step 2 - Run locally
------------------------------------------
  1. Install Ollama from https://ollama.com/download.
  2. Put civbot-q4_k_m.gguf in e.g. C:\models\.
  3. In the same folder create a file named "Modelfile" containing:

       FROM ./civbot-q4_k_m.gguf
       TEMPLATE """<s>[INST] {{ .Prompt }} [/INST]"""
       PARAMETER temperature 0.8
       PARAMETER top_p 0.9
       PARAMETER stop "</s>"

  4. From PowerShell in that folder:

       ollama create civbot -f Modelfile
       ollama run civbot

  5. Paste a prompt like:

       Generate dynamic dialogue for the following Civilization VI
       leader based on the game state.

       Leader: Cleopatra
       State: Greeting, Hostile

  HTTP API for app integration: POST to
  http://localhost:11434/api/generate.


Valid query state labels
------------------------
The model was trained on specific state labels. Type them near identically for best
results - novel states still produce output but less idiosyncratic.

  Canonical (Firaxis-voiced, 100+ examples each):
    Greeting
    Defeated
    Declares War
    Attacked
    Denounces Player
    Denounced by Player
    Agenda-based Approval
    Agenda-based Disapproval
    Delegation
    Invitation to Capital
    Invitation to City
    Quote from Civilopedia
    Requests Declaration of Friendship
    Player Accepts Declaration of Friendship
    Player Rejects Declaration of Friendship
    Accepts Delegation from Player
    Rejects Delegation from Player
    Too Many Troops Near His Border
    Too Many Troops Near Her Border

  Relation-tagged greeting/trade variants (synthetic, 60-70 each):
    Greeting, Neutral / Friendly / Hostile
    Trade Request, Neutral / Unfair to Player
    Trade Acceptance, Friendly
    Trade Rejection, Hostile
    War Declaration, Hostile

  Player-action contextual states (synthetic, 25-70 each):
    Player attacking ally
    Player declared surprise war
    Player eliminated another civ
    Player nearing victory
    Player spreading religion
    Player shares continent, Friendly
    Player high culture
    Player high gold, strong alliance
    Player weak military, high gold
    Player strong military, low gold
    Player strong military, warmonger
    Player weak military, high faith
    Player caught spying
    Player dropping atom bomb

  See civ6_combined_training.jsonl for the full set (356 distinct labels).


Hyperparameters (current defaults)
-----------------------------------
Inference (Modelfile + inference.py + Ollama API):
    temperature        0.8     creativity vs. determinism (0.6 safer, 1.0 wilder)
    top_p              0.9     nucleus sampling cutoff
    stop               </s>    Mistral end-of-turn token
    num_predict / max_new_tokens  120 (lines are usually <30 tokens)

Training (train.py / civbot_colab.ipynb):
    base model         mistralai/Mistral-7B-Instruct-v0.3
    quantization       4-bit NF4, double-quant, bf16 compute
    LoRA rank          16
    LoRA alpha         32
    LoRA dropout       0.05
    target modules     q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    epochs             3
    per-device batch   2
    grad accumulation  8        (effective batch 16)
    learning rate      2e-4 (cosine, 3% warmup)
    optimizer          paged AdamW 8-bit
    max sequence len   512
    eval split         5%

Quantization for export:
    GGUF format, Q4_K_M (~4 GB final size, near-fp16 quality)


-- OR --


Step 2 - Run locally (llama.cpp alternative)
--------------------------------------------
  1. Download a prebuilt Windows release from
     https://github.com/ggerganov/llama.cpp/releases (pick *-bin-win-*,
     AVX2 is the safe default).
  2. Put civbot-q4_k_m.gguf in the same folder.
  3. CLI:
       .\llama-cli.exe -m civbot-q4_k_m.gguf ^
         -p "[INST] Leader: Gandhi`nState: War Declaration, Hostile [/INST]" ^
         -n 120 --temp 0.8 --top-p 0.9
  4. Server: .\llama-server.exe -m civbot-q4_k_m.gguf -c 2048


Evaluation
----------
After Ollama is running with the civbot model, score it on a holdout:

  pip install -r requirements.txt
  python eval.py --num_samples 50

This holds out 50 random examples, generates predictions, and prints
BLEU-4 + character similarity, plus a few side-by-side examples.
(eval.py needs the requests + sacrebleu libraries, both pinned in
requirements.txt.)

Summary
---------------
  Train QLoRA, 3 epochs       Colab/Kaggle T4 free   
  Merge + GGUF + quantize     Colab/Kaggle T4 free     
  Inference per leader line   CPU via Ollama  free   
