from pathlib import Path

from normalization.element_normalizer import MinerUNormalizer


def main():

    project_root = Path(__file__).resolve().parent.parent

    output_dir = project_root / "data" / "output"

    # Find the latest MinerU content_list_v2.json
    content_list_files = sorted(
        output_dir.glob("*_content_list_v2.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not content_list_files:
        raise FileNotFoundError(
            f"No MinerU content list found in: {output_dir}"
        )

    content_list_path = content_list_files[0]

    print(
        f"Using MinerU content list: {content_list_path}"
    )

    # Output normalized document
    output_path = (
        output_dir /
        "normalized_document.json"
    )

    normalizer = MinerUNormalizer(
        content_list_path=content_list_path
    )

    normalized_document = normalizer.save(
        output_path
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    pages = normalized_document.get(
        "pages",
        []
    )

    total_elements = sum(
        len(page.get("elements", []))
        for page in pages
    )

    print("\nNormalization complete.")
    print(
        f"Pages: {len(pages)}"
    )
    print(
        f"Elements: {total_elements}"
    )
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()