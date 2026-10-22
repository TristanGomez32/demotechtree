# Azure VLM-as-judge for `demotechtree`

This add-on evaluates the generated clips in `data/clips/` with a vision-capable model deployed in Azure OpenAI / Microsoft Foundry. It samples **four frames per clip**, sends them directly as base64 images (no Blob Storage upload required), and stores every response locally.

## What it does

- Reads `data/clips/*.mp4` created by `make_defect_clips.py`.
- Extracts four evenly distributed frames per clip, resizing each to at most 768 px on its longest side and JPEG-encoding it to reduce image-token cost.
- Sends the four frames in one request to one Azure deployment.
- Saves one JSON file per clip/deployment, including the full prompt, prompt hash, clip hash, frame indices, raw model output, response ID, and token usage when Azure returns it.
- Skips successful cached outputs when the clip content, prompt, deployment and relevant settings are unchanged.
- Uses `--limit 3` for a small pilot and `--dry-run` to inspect planned work without calling Azure.
- Saves an append-only `summary.jsonl` run log.

Errors are recorded too, but are not treated as successful cache entries; a later run retries them. Use `--force` to deliberately re-run completed inference.

## 1. Install

From the root of your cloned `demotechtree` repository, copy `azure_vlm_judge.py`, `prompts/quality_judge.txt`, `requirements-azure.txt`, and `.env.example` into the repo root / matching folders.

```bash
python -m pip install -r requirements-azure.txt
```

## 2. Azure portal / Microsoft Foundry setup

1. Open [Microsoft Foundry](https://ai.azure.com/) and select the correct directory and subscription.
2. Create or select a Foundry project and an Azure OpenAI resource/connection.
3. In **Models + endpoints** (or the model catalog), deploy an image-capable model. Start with **GPT-4.1 nano** if it is available for your subscription/region; otherwise use **GPT-4o mini**. Both accept images; confirm the current model catalog, pricing and availability before deployment. A deployment name is the custom name you assign, and can differ from the underlying model ID.
4. From the resource's **Keys and Endpoint** page, copy the endpoint and key. Keep the key private; never commit `.env`.
5. Put the endpoint, key and deployment name in `.env` (example below).

The official Microsoft vision guide documents deploying an image-capable model, obtaining the endpoint/key, and sending image inputs: https://learn.microsoft.com/azure/ai-services/openai/how-to/gpt-with-vision

No Azure Blob Storage account is required by this script: image bytes are embedded in the API request. You do not need to upload entire video files to Azure Storage.

## 3. Configure `.env`

Copy `.env.example` to `.env` and fill in your values:

```dotenv
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_API_KEY=your-secret-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21
```

Do not put quotes around the values unless your shell setup requires them. Do not commit `.env`.

## 4. Start with a three-clip pilot

Assuming you already ran `make_defect_clips.py` and it created `data/clips/`:

```bash
python azure_vlm_judge.py --limit 3 --dry-run
python azure_vlm_judge.py --limit 3
```

The dry run lists which clips would be evaluated and makes **no API calls**. The actual run makes at most three inference calls for one deployment, unless some requests fail and are retried by the SDK.

Review the output JSON files in `results/vlm_judge/<deployment>/`. Each JSON includes token usage if supplied by the API. Azure billing can lag, so compare the test with Cost Management after usage has appeared.

## 5. Run the complete set

```bash
python azure_vlm_judge.py
```

Only clips without a matching successful cache result are inferred. To evaluate a second deployment, change `AZURE_OPENAI_DEPLOYMENT` or pass `--deployment ANOTHER_DEPLOYMENT`; results are kept in separate model folders.

## 6. Useful options

```bash
# Three clips, no network/API requests
python azure_vlm_judge.py --limit 3 --dry-run

# Test with a particular deployment name
python azure_vlm_judge.py --limit 3 --deployment gpt-4.1-nano

# Lower image resolution to reduce image-token cost (may hurt quality assessment)
python azure_vlm_judge.py --limit 3 --max-side 512

# Re-run even if cached
python azure_vlm_judge.py --limit 3 --force
```

`--deployment` must match the **deployment name** shown in Azure, not necessarily the catalog's model ID.

## Cost notes

- The script makes one request per clip per deployment, with four images in that request.
- The exact cost depends on model/deployment, image-token accounting, resolution/detail settings, output tokens, and your Azure offer/region. Do not rely on a hard-coded estimate; check current Azure pricing and the Azure Cost Management view.
- The first pilot's JSON logs token usage returned by the API. Image tokens may be included in the prompt-token total; Azure may expose usage details differently by model/version.
- For the cheapest pilot, use the least expensive image-capable deployment available in your Azure account (start by checking GPT-4.1 nano; fall back to GPT-4o mini if needed). Availability and prices can change.
- Keep `--limit 3` for the first real run. Four frames are sent at once to avoid four separate calls and to let the model judge consistency across the clip.

## Output structure

```text
results/vlm_judge/
  summary.jsonl
  <deployment>/
    <clip_id>.json
```

Each successful result contains the raw answer in `output_text`, the exact prompt text and hash, the clip SHA-256, selected frame indices/hashes/dimensions, settings, response ID, and token usage. The model's JSON is stored as raw text rather than silently transformed, so malformed JSON can still be inspected.

## Important experimental note

The prompt asks the model for subjective ordinal scores from 0–4. Compare them to `clips_manifest.csv` (`defect_code`, `severity_level`, `param_value`) and the objective measures (`measured_sharpness_lapvar`, `measured_mean_luma`). Treat the VLM as a judge to validate experimentally, not as ground truth: semantic content and scene lighting can bias its scores.
