import json
import os
import time
from pathlib import Path

from multimodal.context_builder import ContextBuilder
from multimodal.gemini_processor import (
    GeminiDailyQuotaExceeded,
    GeminiVLMProcessor,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Phase 3.2 must consume the Phase 3.1 multimodal document.
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "multimodal_document.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "multimodal_enriched_document.json"
)


VISUAL_TYPES = {
    "image",
    "chart",
}


VALID_MODALITIES = {
    "text",
    "layout",
    "table",
    "visual",
    "unknown",
}


# ==============================================================
# LOAD DOCUMENT
# ==============================================================

def load_document():
    """
    Load the existing enrichment checkpoint if available.

    Otherwise start from the Phase 3.1 multimodal document.
    """

    if OUTPUT_FILE.exists():

        print(
            "Existing enrichment checkpoint found."
        )

        print(
            f"Loading: {OUTPUT_FILE}"
        )

        with OUTPUT_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Phase 3.1 multimodal document was not found.\n"
            f"Expected: {INPUT_FILE}\n\n"
            "Run Phase 3.1 first:"
            "\npython src/run_multimodal_processing.py"
        )

    print(
        "No enrichment checkpoint found."
    )

    print(
        f"Loading Phase 3.1 multimodal document: "
        f"{INPUT_FILE}"
    )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ==============================================================
# VALIDATE MULTIMODAL DOCUMENT
# ==============================================================

def validate_multimodal_document(document):
    """
    Validate that the Phase 3.1 multimodal representation
    contains explicit modality information.

    This prevents Phase 3.2 from silently producing an enriched
    document that later phases cannot interpret.
    """

    pages = document.get("pages")

    if not isinstance(pages, list):

        raise ValueError(
            "Invalid multimodal document: "
            "'pages' must be a list."
        )

    total_elements = 0
    missing_modality = []
    invalid_modality = []
    modality_counts = {}

    for page in pages:

        elements = page.get("elements", [])

        for element in elements:

            total_elements += 1

            element_id = element.get(
                "element_id",
                f"element_{total_elements}",
            )

            modality = element.get(
                "modality"
            )

            if modality is None:

                missing_modality.append(
                    element_id
                )

                continue

            if modality not in VALID_MODALITIES:

                invalid_modality.append(
                    {
                        "element_id": element_id,
                        "modality": modality,
                    }
                )

                continue

            modality_counts[modality] = (
                modality_counts.get(
                    modality,
                    0,
                )
                + 1
            )

    print()
    print("=" * 80)
    print("MULTIMODAL DOCUMENT VALIDATION")
    print("=" * 80)

    print(
        f"Pages             : {len(pages)}"
    )

    print(
        f"Total elements    : {total_elements}"
    )

    print()
    print("Modality counts:")

    for modality, count in sorted(
        modality_counts.items()
    ):

        print(
            f"  {modality:<12} {count}"
        )

    # ----------------------------------------------------------
    # Missing modality
    # ----------------------------------------------------------

    if missing_modality:

        preview = ", ".join(
            missing_modality[:10]
        )

        raise ValueError(
            "Phase 3.1 validation failed: "
            f"{len(missing_modality)} elements are missing "
            f"'modality'.\n"
            f"Examples: {preview}\n\n"
            "Run Phase 3.1 again before running Gemini enrichment."
        )

    # ----------------------------------------------------------
    # Invalid modality
    # ----------------------------------------------------------

    if invalid_modality:

        preview = ", ".join(
            f"{item['element_id']}="
            f"{item['modality']}"
            for item in invalid_modality[:10]
        )

        raise ValueError(
            "Phase 3.1 validation failed: "
            "invalid modality values were found.\n"
            f"Examples: {preview}"
        )

    print()
    print(
        "Validation: PASSED"
    )

    print("=" * 80)


# ==============================================================
# SAVE CHECKPOINT
# ==============================================================

def save_checkpoint(document):
    """
    Save the document checkpoint safely.

    A temporary file is written first so that the existing
    checkpoint is not directly overwritten during serialization.

    Windows + OneDrive can temporarily lock the destination file,
    so os.replace() is retried several times before failing.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = OUTPUT_FILE.with_suffix(
        ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            document,
            f,
            indent=2,
            ensure_ascii=False,
        )

    last_error = None

    for attempt in range(5):

        try:

            os.replace(
                temporary_file,
                OUTPUT_FILE,
            )

            print(
                f"Checkpoint saved -> {OUTPUT_FILE}"
            )

            return

        except PermissionError as exc:

            last_error = exc

            if attempt < 4:

                wait_time = 1 + attempt

                print(
                    f"Checkpoint file temporarily locked. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )

            else:

                raise last_error


# ==============================================================
# IMAGE PATH RESOLUTION
# ==============================================================

def resolve_image_path(image_path):
    """
    Resolve MinerU's relative image paths against the project
    output directory.

    Returns:
        Path | None
    """

    if not image_path:
        return None

    path = Path(image_path)

    # Absolute path.
    if path.is_absolute():

        if path.exists():
            return path

        return None

    # ----------------------------------------------------------
    # Candidate 1:
    # MinerU output directory.
    # ----------------------------------------------------------

    candidate = (
        PROJECT_ROOT
        / "data"
        / "output"
        / path
    )

    if candidate.exists():
        return candidate

    # ----------------------------------------------------------
    # Candidate 2:
    # Project root.
    # ----------------------------------------------------------

    candidate = (
        PROJECT_ROOT
        / path
    )

    if candidate.exists():
        return candidate

    return None


# ==============================================================
# VISUAL DISCOVERY
# ==============================================================

def get_visual_elements(document):
    """
    Return all image/chart elements while preserving their
    page and element indexes.
    """

    visuals = []

    for page_index, page in enumerate(
        document["pages"]
    ):

        for element_index, element in enumerate(
            page["elements"]
        ):

            if element.get("type") in VISUAL_TYPES:

                visuals.append(
                    {
                        "page_index": page_index,
                        "element_index": element_index,
                        "page_number": page["page_number"],
                        "element": element,
                    }
                )

    return visuals


# ==============================================================
# ENRICHMENT STATE
# ==============================================================

def is_processed(element):
    """
    Determine whether a visual already contains a valid
    Gemini analysis.

    A failed visual is NOT considered processed and can therefore
    be retried on a future run.
    """

    analysis = element.get(
        "visual_analysis"
    )

    return (
        isinstance(
            analysis,
            dict,
        )
        and isinstance(
            analysis.get("analysis"),
            dict,
        )
        and bool(
            analysis["analysis"]
        )
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print()
    print("=" * 80)
    print("MULTIMODAL ENRICHMENT")
    print("=" * 80)

    # ----------------------------------------------------------
    # Load document/checkpoint
    # ----------------------------------------------------------

    document = load_document()

    # ----------------------------------------------------------
    # Validate modality BEFORE Gemini calls
    # ----------------------------------------------------------

    validate_multimodal_document(
        document
    )

    pages = document["pages"]

    # ----------------------------------------------------------
    # Discover visuals
    # ----------------------------------------------------------

    visuals = get_visual_elements(
        document
    )

    total_visuals = len(visuals)

    # ----------------------------------------------------------
    # Calculate existing state BEFORE this run
    # ----------------------------------------------------------

    already_processed = sum(
        1
        for item in visuals
        if is_processed(
            item["element"]
        )
    )

    pending_visuals = (
        total_visuals
        - already_processed
    )

    print()
    print(
        f"Total visual elements : {total_visuals}"
    )

    print(
        f"Already processed     : "
        f"{already_processed}"
    )

    print(
        f"Pending               : "
        f"{pending_visuals}"
    )

    print(
        f"Output checkpoint     : "
        f"{OUTPUT_FILE}"
    )

    print()

    # ----------------------------------------------------------
    # Nothing left to process
    # ----------------------------------------------------------

    if pending_visuals == 0:

        print(
            "All visual elements are already enriched."
        )

        document[
            "multimodal_processing"
        ] = {
            "processor": "GeminiVLMProcessor",
            "model": os.getenv(
                "GEMINI_VLM_MODEL",
                "gemini-3.6-flash",
            ),
            "context_builder": (
                "RAG-Anything-style page context"
            ),
            "context_window": 1,
            "total_visuals": total_visuals,
            "already_processed": already_processed,
            "pending": 0,
            "status": "complete",
        }

        save_checkpoint(
            document
        )

        return

    # ----------------------------------------------------------
    # Context builder
    # ----------------------------------------------------------

    context_builder = ContextBuilder(
        context_window=1,
        context_mode="page",
        max_context_tokens=2000,
        include_headers=False,
        include_captions=True,
        context_filter_content_types=[
            "text"
        ],
    )

    # ----------------------------------------------------------
    # Gemini processor
    # ----------------------------------------------------------

    gemini = GeminiVLMProcessor(
        max_retries=3,
        retry_delay=5.0,
    )

    # ----------------------------------------------------------
    # Current-run statistics
    # ----------------------------------------------------------

    run_successful = 0
    run_skipped = 0
    run_failed = 0

    quota_stopped = False

    # ----------------------------------------------------------
    # Process visuals
    # ----------------------------------------------------------

    for visual_number, visual_info in enumerate(
        visuals,
        start=1,
    ):

        page_index = visual_info[
            "page_index"
        ]

        element_index = visual_info[
            "element_index"
        ]

        page_number = visual_info[
            "page_number"
        ]

        element = visual_info[
            "element"
        ]

        element_id = element[
            "element_id"
        ]

        print()
        print("=" * 80)

        print(
            f"[{visual_number}/{total_visuals}] "
            f"{element_id} "
            f"(page {page_number})"
        )

        print("=" * 80)

        # ------------------------------------------------------
        # Resume logic
        # ------------------------------------------------------

        if is_processed(
            element
        ):

            print(
                f"SKIP: {element_id} "
                f"already processed."
            )

            run_skipped += 1

            continue

        # ------------------------------------------------------
        # Resolve visual
        # ------------------------------------------------------

        image_path = resolve_image_path(
            element.get(
                "image_path"
            )
        )

        if image_path is None:

            error_message = (
                f"Could not resolve image path "
                f"for {element_id}: "
                f"{element.get('image_path')}"
            )

            print(
                f"FAILED: {element_id}"
            )

            print(
                f"Error: {error_message}"
            )

            element[
                "visual_processing_error"
            ] = {
                "element_id": element_id,
                "page_number": page_number,
                "error": error_message,
            }

            run_failed += 1

            save_checkpoint(
                document
            )

            continue

        # ------------------------------------------------------
        # Build RAG-Anything-style context
        # ------------------------------------------------------

        context_result = (
            context_builder.build_context(
                pages=pages,
                target_page_index=page_index,
                target_element_index=element_index,
            )
        )

        # ------------------------------------------------------
        # Gemini VLM
        # ------------------------------------------------------

        try:

            result = gemini.analyze_visual(
                image_path=str(
                    image_path
                ),
                caption=element.get(
                    "caption"
                ),
                footnote=element.get(
                    "footnote"
                ),
                context=context_result[
                    "context"
                ],
            )

            # --------------------------------------------------
            # Store visual analysis
            # --------------------------------------------------

            element[
                "visual_analysis"
            ] = {
                "model": result[
                    "model"
                ],
                "analysis": result[
                    "analysis"
                ],
            }

            # --------------------------------------------------
            # Store provenance/context
            # --------------------------------------------------

            element[
                "visual_context"
            ] = {
                "context_mode": (
                    context_result[
                        "context_mode"
                    ]
                ),
                "context_window": (
                    context_result[
                        "context_window"
                    ]
                ),
                "pages_included": (
                    context_result[
                        "pages_included"
                    ]
                ),
                "context": (
                    context_result[
                        "context"
                    ]
                ),
            }

            # --------------------------------------------------
            # Remove previous error
            # --------------------------------------------------

            element.pop(
                "visual_processing_error",
                None,
            )

            run_successful += 1

            print(
                f"SUCCESS: {element_id}"
            )

            # --------------------------------------------------
            # Save immediately after every success.
            # --------------------------------------------------

            save_checkpoint(
                document
            )

        except GeminiDailyQuotaExceeded:

            quota_stopped = True

            print()
            print(
                "Daily Gemini quota exhausted."
            )

            print(
                "Saving checkpoint before stopping..."
            )

            save_checkpoint(
                document
            )

            break

        except Exception as exc:

            run_failed += 1

            print(
                f"FAILED: {element_id}"
            )

            print(
                f"Error: {exc}"
            )

            element[
                "visual_processing_error"
            ] = {
                "element_id": element_id,
                "page_number": page_number,
                "error": str(exc),
            }

            # --------------------------------------------------
            # Save failure immediately.
            # --------------------------------------------------

            save_checkpoint(
                document
            )

    # ----------------------------------------------------------
    # Recalculate actual document state
    # ----------------------------------------------------------

    visuals_after_run = get_visual_elements(
        document
    )

    processed_after_run = sum(
        1
        for item in visuals_after_run
        if is_processed(
            item["element"]
        )
    )

    failed_after_run = sum(
        1
        for item in visuals_after_run
        if item["element"].get(
            "visual_processing_error"
        )
        and not is_processed(
            item["element"]
        )
    )

    remaining_after_run = (
        total_visuals
        - processed_after_run
    )

    # ----------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------

    document[
        "multimodal_processing"
    ] = {
        "processor": "GeminiVLMProcessor",
        "model": gemini.model,
        "context_builder": (
            "RAG-Anything-style page context"
        ),
        "context_window": 1,

        "total_visuals": total_visuals,
        "processed": processed_after_run,
        "remaining": remaining_after_run,
        "failed": failed_after_run,

        "run_successful": run_successful,
        "run_skipped": run_skipped,
        "run_failed": run_failed,

        "quota_stopped": quota_stopped,

        "status": (
            "quota_stopped"
            if quota_stopped
            else (
                "complete"
                if remaining_after_run == 0
                else "partial"
            )
        ),
    }

    # ----------------------------------------------------------
    # Final checkpoint
    # ----------------------------------------------------------

    save_checkpoint(
        document
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print()
    print("=" * 80)
    print("MULTIMODAL ENRICHMENT SUMMARY")
    print("=" * 80)

    print(
        f"Total visuals       : "
        f"{total_visuals}"
    )

    print(
        f"Processed           : "
        f"{processed_after_run}"
    )

    print(
        f"Remaining           : "
        f"{remaining_after_run}"
    )

    print(
        f"Failed              : "
        f"{failed_after_run}"
    )

    print()
    print(
        f"This run successful : "
        f"{run_successful}"
    )

    print(
        f"This run skipped    : "
        f"{run_skipped}"
    )

    print(
        f"This run failed     : "
        f"{run_failed}"
    )

    print(
        f"Quota stopped       : "
        f"{quota_stopped}"
    )

    print()
    print(
        f"Status              : "
        f"{document['multimodal_processing']['status']}"
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()