import json
from pathlib import Path

from multimodal.multimodal_processor import MultimodalProcessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "normalized_document.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "multimodal_document.json"
)


def main():

    print("Loading normalized document...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Normalized document not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        document = json.load(file)

    print("Normalized document loaded.")

    processor = MultimodalProcessor()

    output_pages = []

    total_elements = 0

    for page in document.get("pages", []):

        processed_elements = []

        for element in page.get("elements", []):

            processed_element = processor.process_element(
                element
            )

            processed_elements.append(
                processed_element
            )

            total_elements += 1

        output_pages.append({
            "page_number": page.get("page_number"),
            "elements": processed_elements
        })

    output = {
        "document": document.get("document", {}),
        "pages": output_pages
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("===================================")
    print("Phase 3.1 completed successfully.")
    print("===================================")
    print(f"Pages:    {len(output_pages)}")
    print(f"Elements: {total_elements}")
    print(f"Output:   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()