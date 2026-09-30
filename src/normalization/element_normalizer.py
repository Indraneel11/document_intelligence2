import json
from pathlib import Path


class MinerUNormalizer:
    """
    Converts MinerU content_list_v2.json into the
    internal document representation used by our RAG pipeline.
    """

    def __init__(self, content_list_path: Path):
        self.content_list_path = content_list_path

    def load_content(self):
        """Load MinerU content_list_v2.json."""

        with open(
            self.content_list_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    def normalize(self):
        """
        Convert MinerU's page-oriented content list into
        our normalized document representation.
        """

        raw_pages = self.load_content()

        if not isinstance(raw_pages, list):
            raise ValueError(
                "Expected content_list_v2.json to contain a list."
            )

        normalized_pages = []

        for page_index, page_elements in enumerate(raw_pages):

            page_number = page_index + 1

            if not isinstance(page_elements, list):
                continue

            normalized_elements = []

            for element_index, element in enumerate(
                page_elements
            ):

                if not isinstance(element, dict):
                    continue

                normalized_element = self._normalize_element(
                    element=element,
                    page_number=page_number,
                    element_index=element_index,
                )

                normalized_elements.append(
                    normalized_element
                )

            normalized_pages.append(
                {
                    "page_number": page_number,
                    "elements": normalized_elements,
                }
            )

        return {
            "document": {
                "source": self.content_list_path.name,
                "total_pages": len(normalized_pages),
            },
            "pages": normalized_pages,
        }

    # ========================================================
    # Element normalization
    # ========================================================

    def _normalize_element(
        self,
        element: dict,
        page_number: int,
        element_index: int,
    ):
        """
        Normalize a single MinerU element.
        """

        element_type = element.get(
            "type",
            "unknown"
        )

        content = element.get(
            "content",
            {}
        )

        bbox = element.get(
            "bbox"
        )

        normalized = {
            "element_id": (
                f"p{page_number:03d}"
                f"_e{element_index + 1:03d}"
            ),
            "page_number": page_number,
            "type": element_type,
            "bbox": bbox,
            "text": None,
            "image_path": None,
            "caption": [],
            "footnote": [],
            "metadata": {},
        }

        # ----------------------------------------------------
        # Paragraph
        # ----------------------------------------------------

        if element_type == "paragraph":

            normalized["text"] = (
                self._extract_text_items(
                    content.get(
                        "paragraph_content",
                        []
                    )
                )
            )

        # ----------------------------------------------------
        # List
        # ----------------------------------------------------

        elif element_type == "list":

            list_items = content.get(
                "list_items",
                []
            )

            texts = []

            if isinstance(list_items, list):

                for item in list_items:

                    if not isinstance(item, dict):
                        continue

                    item_content = item.get(
                        "item_content",
                        []
                    )

                    text = self._extract_text_items(
                        item_content
                    )

                    if text:
                        texts.append(text)

            normalized["text"] = (
                "\n".join(texts)
                if texts
                else None
            )

            normalized["metadata"]["list_type"] = (
                content.get("list_type")
            )

            normalized["metadata"]["item_count"] = (
                len(texts)
            )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        elif element_type == "title":

            normalized["text"] = (
                self._extract_text_items(
                    content.get(
                        "title_content",
                        []
                    )
                )
            )

            if "level" in content:
                normalized["metadata"]["level"] = (
                    content["level"]
                )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        elif element_type == "image":

            self._normalize_visual(
                normalized,
                content
            )

        # ----------------------------------------------------
        # Chart
        # ----------------------------------------------------

        elif element_type == "chart":

            self._normalize_visual(
                normalized,
                content
            )

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        elif element_type == "table":

            normalized["text"] = (
                self._extract_table_content(
                    content
                )
            )

            normalized["caption"] = (
                self._extract_text_items(
                    content.get(
                        "table_caption",
                        []
                    )
                )
            )

            normalized["footnote"] = (
                self._extract_text_items(
                    content.get(
                        "table_footnote",
                        []
                    )
                )
            )

        # ----------------------------------------------------
        # Page header
        # ----------------------------------------------------

        elif element_type == "page_header":

            normalized["text"] = (
                self._extract_text_items(
                    content.get(
                        "page_header_content",
                        []
                    )
                )
            )

        # ----------------------------------------------------
        # Page footer
        # ----------------------------------------------------

        elif element_type == "page_footer":

            normalized["text"] = (
                self._extract_text_items(
                    content.get(
                        "page_footer_content",
                        []
                    )
                )
            )

        # ----------------------------------------------------
        # Page number
        # ----------------------------------------------------

        elif element_type == "page_number":

            normalized["text"] = (
                self._extract_text_items(
                    content.get(
                        "page_number_content",
                        []
                    )
                )
            )

        # ----------------------------------------------------
        # Page footnote
        # ----------------------------------------------------

        elif element_type == "page_footnote":

            normalized["text"] = (
                self._extract_text_items(
                    content.get(
                        "page_footnote_content",
                        []
                    )
                )
            )

        # ----------------------------------------------------
        # Unknown / future MinerU types
        # ----------------------------------------------------

        else:

            normalized["metadata"][
                "original_content"
            ] = content

        return normalized

    # ========================================================
    # Visual normalization
    # ========================================================

    def _normalize_visual(
        self,
        normalized: dict,
        content: dict,
    ):
        """
        Normalize image/chart content.
        """

        image_source = content.get(
            "image_source",
            {}
        )

        if isinstance(image_source, dict):

            normalized["image_path"] = (
                image_source.get("path")
            )

        normalized["caption"] = (
            self._extract_text_items(
                content.get(
                    "image_caption",
                    content.get(
                        "chart_caption",
                        []
                    )
                )
            )
        )

        normalized["footnote"] = (
            self._extract_text_items(
                content.get(
                    "image_footnote",
                    content.get(
                        "chart_footnote",
                        []
                    )
                )
            )
        )

    # ========================================================
    # Text extraction
    # ========================================================

    @staticmethod
    def _extract_text_items(items):
        """
        Extract text from MinerU's structured text items.
        """

        if not isinstance(items, list):
            return None

        texts = []

        for item in items:

            if not isinstance(item, dict):
                continue

            text = item.get(
                "content"
            )

            if isinstance(text, str) and text.strip():
                texts.append(
                    text.strip()
                )

        if not texts:
            return None

        return "\n".join(texts)

    # ========================================================
    # Table extraction
    # ========================================================

    @staticmethod
    def _extract_table_content(content):
        """
        Preserve the table representation produced by MinerU.
        """

        if not isinstance(content, dict):
            return None

        for key in (
            "html",
            "table_body",
            "content",
        ):

            value = content.get(key)

            if isinstance(value, str) and value.strip():
                return value

        return None

    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
        output_path: Path
    ):
        """
        Normalize and save the document.
        """

        normalized_document = self.normalize()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                normalized_document,
                file,
                indent=2,
                ensure_ascii=False
            )

        return normalized_document