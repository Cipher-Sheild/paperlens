# Deploying PaperLens (free, no install for your users)

## Option A - Hugging Face Spaces (recommended)
1. Create an account at huggingface.co -> **New Space** -> SDK: **Gradio** -> Hardware: **CPU basic (free)**.
2. Upload the project files (everything except `tests/`, `docs/`, `notebooks/` is enough: `app.py`, `paperlens/`, `requirements.txt`, `sample_data/`).
3. Put this block at the very top of the Space's `README.md`:
   ```yaml
   ---
   title: PaperLens
   emoji: 🔍
   colorFrom: indigo
   colorTo: pink
   sdk: gradio
   sdk_version: 6.29.0        # use the Gradio version you tested with
   app_file: app.py
   pinned: false
   license: mit
   short_description: BERT-powered research paper summarizer
   ---
   ```
4. In the Space's `requirements.txt` **delete the `gradio` line** (Spaces installs it from `sdk_version`).
5. Wait for the build (3-6 minutes). Your public link: `https://huggingface.co/spaces/<you>/<space-name>`.

Notes: the free CPU tier handles one paper in roughly 10-30 s after the first model download; the Space sleeps when idle and wakes on the next visit.
For public hosting the link downloader already blocks private/internal addresses; keep the 30 MB limit.

## Option B - Google Colab (for demos)
Upload the zip, run `!unzip -q paperlens.zip`, `%cd paperlens`, `!pip -q install -r requirements.txt`, `!python app.py --share`.
The `gradio.live` link lasts while the notebook runs.

## Option C - your own machine / server
```bash
pip install -r requirements.txt
python app.py --port 7860        # add --share for a temporary public link
```
