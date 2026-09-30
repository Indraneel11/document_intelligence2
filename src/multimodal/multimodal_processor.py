from typing import Any, Dict


TEXT_TYPES = {
    "paragraph",
    "title",
    "list",
    "page_footnote",
}

LAYOUT_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
}

TABLE_TYPES = {
    "table",
}

VISUAL_TYPES = {
    "image",
    "chart",
}


class MultimodalProcessor:
    """
    Converts normalized MinerU elements into modality-aware
    representations.

    This stage does NOT call a VLM.
    """

    def process_element(
        self,
        element: Dict[str, Any]
    ) -> Dict[str, Any]:

        element_type = element.get("type")

        if element_type in TEXT_TYPES:
            return self._process_text(element)

        if element_type in LAYOUT_TYPES:
            return self._process_layout(element)

        if element_type in TABLE_TYPES:
            return self._process_table(element)

        if element_type in VISUAL_TYPES:
            return self._process_visual(element)

        return self._process_unknown(element)

    def _base_metadata(
        self,
        element: Dict[str, Any]
    ) -> Dict[str, Any]:

        return {
            "element_id": element.get("element_id"),
            "page_number": element.get("page_number"),
            "bbox": element.get("bbox"),
            "type": element.get("type"),
        }

    def _process_text(
        self,
        element: Dict[str, Any]
    ) -> Dict[str, Any]:

        result = self._base_metadata(element)

        result.update({
            "modality": "text",
            "text": element.get("text", ""),
            "metadata": element.get("metadata", {})
        })

        return result

    def _process_layout(
        self,
        element: Dict[str, Any]
    ) -> Dict[str, Any]:

        result = self._base_metadata(element)

        result.update({
            "modality": "layout",
            "text": element.get("text", ""),
            "metadata": element.get("metadata", {})
        })

        return result

    def _process_table(
        self,
        element: Dict[str, Any]
    ) -> Dict[str, Any]:

        result = self._base_metadata(element)

        result.update({
            "modality": "table",
            "text": element.get("text", ""),
            "caption": element.get("caption"),
            "footnote": element.get("footnote"),
            "metadata": element.get("metadata", {})
        })

        return result

    def _process_visual(
        self,
        element: Dict[str, Any]
    ) -> Dict[str, Any]:

        result = self._base_metadata(element)

        result.update({
            "modality": "visual",
            "image_path": element.get("image_path"),
            "caption": element.get("caption"),
            "footnote": element.get("footnote"),

            # Gemini analysis will be added later.
            "visual_analysis": None,

            "metadata": element.get("metadata", {})
        })

        return result

    def _process_unknown(
        self,
        element: Dict[str, Any]
    ) -> Dict[str, Any]:

        result = self._base_metadata(element)

        result.update({
            "modality": "unknown",
            "text": element.get("text", ""),
            "metadata": element.get("metadata", {})
        })

        return result