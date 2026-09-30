from typing import Any, Dict


EXCLUDED_MODALITIES = {
    "layout",
}


def is_semantically_chunkable(
    element: Dict[str, Any]
) -> bool:
    """
    Determine whether an element should participate
    in semantic chunking.

    The canonical multimodal document remains unchanged.
    This function only controls which elements are passed
    to the semantic chunker.
    """

    modality = element.get("modality")

    # ------------------------------------------------------
    # Layout elements
    # ------------------------------------------------------

    if modality in EXCLUDED_MODALITIES:
        return False

    # ------------------------------------------------------
    # Text
    # ------------------------------------------------------

    if modality == "text":

        text = element.get("text")

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            return False

    # ------------------------------------------------------
    # Tables
    # ------------------------------------------------------

    # Keep all table elements, even if MinerU did not
    # extract their contents into the normalized document.
    #
    # This preserves the table as a multimodal document
    # element for later table/visual enrichment.
    if modality == "table":
        return True

    # ------------------------------------------------------
    # Visual elements
    # ------------------------------------------------------

    if modality == "visual":

        has_image = bool(
            element.get("image_path")
        )

        has_analysis = bool(
            element.get("visual_analysis")
        )

        text = element.get("text")

        has_text = (
            isinstance(text, str)
            and bool(text.strip())
        )

        if not (
            has_image
            or has_analysis
            or has_text
        ):
            return False

    # ------------------------------------------------------
    # Unknown elements
    # ------------------------------------------------------

    if modality == "unknown":

        text = element.get("text")

        has_text = (
            isinstance(text, str)
            and bool(text.strip())
        )

        has_content = bool(
            element.get("content")
        )

        if not (
            has_text
            or has_content
        ):
            return False

    return True