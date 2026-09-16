#!/usr/bin/env python3
"""
Pipeline: intent image -> Gemini prompt generation -> OmniFlash video editing -> download.

For each test video in the input directory:
1. Read the intent image (and optional reference image) alongside the video.
2. Call Gemini 3.8 Flash to generate an OmniFlash editing prompt from the intent.
3. Call OmniFlash (gemini-omni-1.1-flash-preview) with the video, prompt, and
   optional reference image.
4. Save the output video locally with a timestamp suffix.
"""

import argparse
import base64
import glob
import logging
import os
import re
import sys
import time
from datetime import datetime

from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT = "cloud-llm-preview1"
LOCATION = "global"
GEMINI_FLASH_MODEL = "gemini-3.8-flash"
OMNIFLASH_MODEL = "gemini-omni-1.1-flash-preview"
GCS_OUTPUT_PATH = "gs://agolis-allen-first-ai/omniflash_edit/omniflash_edit_output"

PROMPT_GENERATION_SYSTEM = """\
You are an expert prompt engineer for a video editing AI model called OmniFlash.
Given an input video and an intent image that describes the desired edits, you must
produce a single, clear, concise editing prompt that OmniFlash can follow.

Rules:
- Include ONLY the edits explicitly described in the intent image. Do NOT add,
  invent, or suggest any changes beyond what the intent specifies. If the intent
  says to change text and material, change only those — nothing else.
- Be specific about what to change: text replacements, material changes, object
  removal, color changes, lighting adjustments, etc.
- CRITICAL: The intent image uses red boxes to highlight target objects, but these
  red boxes do NOT exist in the actual video. You MUST NOT reference "red box" in
  your prompt. Instead, identify the target objects by their distinguishing visual
  features — shape, color, text, position, size, or type (e.g. "the large curved
  billboard on the building facade", "the Starbucks logo and signage", "the row of
  3D metallic letters reading FTC 丰泰城 on the stone ledge"). Use descriptions
  that would uniquely identify the object to someone watching the video.
- If the intent mentions a reference image, instruct OmniFlash to use the provided
  reference image for the replacement appearance.
- Keep everything else in the video unchanged — preserve the original scene,
  background, lighting, shadows, perspective, scale, and motion continuity so the
  result looks natural.
- Keep the prompt under 200 words.
- Output ONLY the prompt text, nothing else — no preamble, no explanation.
"""


def build_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        http_options=types.HttpOptions(headers={"Api-Revision": "2026-05-20"}),
    )


def read_file_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def discover_test_cases(input_dir: str) -> list[dict]:
    """Find all (video, intent, optional ref_image) groups by name prefix."""
    videos = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    cases = []
    for video_path in videos:
        basename = os.path.basename(video_path)
        # Extract prefix: everything before _<duration>.mp4
        # e.g. "Test Video 1_15s.mp4" -> prefix "Test Video 1"
        match = re.match(r"^(.+?)_\d+s\.mp4$", basename)
        if not match:
            log.warning("Skipping video with unexpected name format: %s", basename)
            continue
        prefix = match.group(1)

        intent_path = os.path.join(input_dir, f"{prefix}_intend.jpg")
        if not os.path.exists(intent_path):
            log.warning("No intent image found for %s, skipping", basename)
            continue

        ref_image_path = os.path.join(input_dir, f"{prefix}_ref_image.jpg")
        has_ref = os.path.exists(ref_image_path)

        cases.append({
            "prefix": prefix,
            "video_path": video_path,
            "intent_path": intent_path,
            "ref_image_path": ref_image_path if has_ref else None,
        })

    return cases


def generate_prompt(client: genai.Client, case: dict) -> str:
    """Call Gemini 3.8 Flash to read the intent image and produce an editing prompt."""
    log.info("[%s] Generating OmniFlash prompt via Gemini Flash...", case["prefix"])

    intent_b64 = read_file_as_base64(case["intent_path"])
    video_b64 = read_file_as_base64(case["video_path"])

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=base64.b64decode(video_b64), mime_type="video/mp4"),
                types.Part.from_bytes(data=base64.b64decode(intent_b64), mime_type="image/jpeg"),
            ],
        )
    ]

    user_text = (
        "Look at the input video and the intent image. The intent image shows "
        "a screenshot of the video with annotations (red boxes, numbered instructions) "
        "describing what edits should be made. IMPORTANT: The red boxes are only "
        "annotations in this intent image — they do NOT appear in the actual video. "
        "You must identify the target objects by their visual features (shape, color, "
        "text, position, type) so that OmniFlash can locate them in the video without "
        "any reference to red boxes."
    )

    if case["ref_image_path"]:
        ref_b64 = read_file_as_base64(case["ref_image_path"])
        contents[0].parts.append(
            types.Part.from_bytes(data=base64.b64decode(ref_b64), mime_type="image/jpeg")
        )
        user_text += (
            " A reference image is also provided — the edit should incorporate "
            "the visual content from this reference image as described in the intent."
        )

    user_text += (
        "\n\nGenerate a concise, precise prompt for OmniFlash to edit this video. "
        "Include ONLY the changes described in the intent — nothing more."
    )
    contents[0].parts.append(types.Part.from_text(text=user_text))

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=PROMPT_GENERATION_SYSTEM,
                    temperature=1,
                ),
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                log.warning("[%s] Rate limited, retrying in %ds...", case["prefix"], wait)
                time.sleep(wait)
            else:
                raise

    prompt = response.text.strip()
    log.info("[%s] Generated prompt: %s", case["prefix"], prompt)
    return prompt


def call_omniflash(client: genai.Client, case: dict, prompt: str) -> bytes | None:
    """Call OmniFlash to edit the video and return raw video bytes."""
    log.info("[%s] Calling OmniFlash for video editing...", case["prefix"])

    video_b64 = read_file_as_base64(case["video_path"])

    content_parts = [
        {"type": "video", "data": video_b64, "mime_type": "video/mp4"},
    ]

    if case["ref_image_path"]:
        ref_b64 = read_file_as_base64(case["ref_image_path"])
        content_parts.append(
            {"type": "image", "data": ref_b64, "mime_type": "image/jpeg"}
        )

    content_parts.append({"type": "text", "text": prompt})

    interaction = client.interactions.create(
        model=OMNIFLASH_MODEL,
        input=[
            {
                "type": "user_input",
                "content": content_parts,
            },
        ],
    )

    for step in interaction.steps:
        if step.type == "model_output" and step.content:
            for part in step.content:
                if part.type == "text":
                    log.info("[%s] OmniFlash text: %s", case["prefix"], part.text)
                elif part.type == "video":
                    if part.data:
                        return base64.b64decode(part.data)
                    if part.uri:
                        from google.cloud import storage as gcs
                        bucket_name, blob_name = part.uri[len("gs://"):].split("/", 1)
                        return (
                            gcs.Client()
                            .bucket(bucket_name)
                            .blob(blob_name)
                            .download_as_bytes()
                        )

    log.warning("[%s] No video output received from OmniFlash", case["prefix"])
    return None


def upload_to_gcs(local_path: str, gcs_base_path: str) -> str:
    """Upload a local file to GCS. Returns the gs:// URI."""
    from google.cloud import storage as gcs

    bucket_name, prefix = gcs_base_path[len("gs://"):].split("/", 1)
    blob_name = f"{prefix}/{os.path.basename(local_path)}"
    bucket = gcs.Client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket_name}/{blob_name}"
    log.info("Uploaded to GCS: %s", uri)
    return uri


def save_video(video_bytes: bytes, prefix: str, timestamp: str, output_dir: str) -> str:
    """Save video bytes to output_dir with a timestamp suffix."""
    os.makedirs(output_dir, exist_ok=True)
    safe_prefix = prefix.replace(" ", "_")
    filename = f"{safe_prefix}_{timestamp}.mp4"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        f.write(video_bytes)
    log.info("Saved output video: %s", output_path)
    return output_path


def save_prompt(prompt: str, prefix: str, timestamp: str, output_dir: str) -> str:
    """Save the generated prompt to a text file alongside the video."""
    os.makedirs(output_dir, exist_ok=True)
    safe_prefix = prefix.replace(" ", "_")
    filename = f"{safe_prefix}_{timestamp}_prompt.txt"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    log.info("Saved prompt: %s", output_path)
    return output_path


def process_case(
    client: genai.Client, case: dict, output_dir: str, upload_gcs: bool
) -> bool:
    """Run the full pipeline for a single test case. Returns True on success."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt = generate_prompt(client, case)
        prompt_path = save_prompt(prompt, case["prefix"], timestamp, output_dir)
        if upload_gcs:
            upload_to_gcs(prompt_path, GCS_OUTPUT_PATH)
        video_bytes = call_omniflash(client, case, prompt)
        if video_bytes:
            local_path = save_video(video_bytes, case["prefix"], timestamp, output_dir)
            if upload_gcs:
                upload_to_gcs(local_path, GCS_OUTPUT_PATH)
            return True
        return False
    except Exception:
        log.exception("[%s] Failed to process", case["prefix"])
        return False


def main():
    parser = argparse.ArgumentParser(description="OmniFlash video editing pipeline")
    parser.add_argument(
        "--input-dir",
        default="omniflash_edit_cut",
        help="Directory containing input videos, intent images, and reference images",
    )
    parser.add_argument(
        "--output-dir",
        default="omniflash_edit_output",
        help="Directory to save edited videos",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Process only cases whose prefix contains this substring (e.g. 'Video 1')",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Disable uploading output videos to GCS (upload is ON by default)",
    )
    args = parser.parse_args()

    cases = discover_test_cases(args.input_dir)
    if not cases:
        log.error("No test cases found in %s", args.input_dir)
        sys.exit(1)

    if args.filter:
        cases = [c for c in cases if args.filter in c["prefix"]]
        if not cases:
            log.error("No test cases match filter '%s'", args.filter)
            sys.exit(1)

    upload_gcs = not args.no_upload
    log.info("Found %d test case(s): %s", len(cases), [c["prefix"] for c in cases])
    log.info("GCS upload: %s", "ON" if upload_gcs else "OFF")

    client = build_client()

    succeeded, failed = 0, 0
    for case in cases:
        log.info("=" * 60)
        log.info("Processing: %s", case["prefix"])
        if case["ref_image_path"]:
            log.info("  Reference image: %s", case["ref_image_path"])
        log.info("=" * 60)

        if process_case(client, case, args.output_dir, upload_gcs):
            succeeded += 1
        else:
            failed += 1

    log.info("Done. %d succeeded, %d failed out of %d total.", succeeded, failed, len(cases))


if __name__ == "__main__":
    main()
