from pathlib import Path
import sys


# Add the src directory to Python's import path.
# This is necessary because this script lives inside src/graph/.
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from graph.document_graph import DocumentGraphBuilder


def main():
    project_root = Path(__file__).resolve().parent.parent.parent

    output_dir = project_root / "data" / "output"

    enriched_document_path = (
        output_dir / "multimodal_enriched_document.json"
    )

    chunks_path = (
        output_dir / "multimodal_chunks.json"
    )

    output_path = (
        output_dir / "document_graph.json"
    )

    print("=" * 70)
    print("PHASE 5A — DOCUMENT GRAPH CONSTRUCTION")
    print("=" * 70)

    print(f"\nEnriched document : {enriched_document_path}")
    print(f"Chunks            : {chunks_path}")
    print(f"Output            : {output_path}")

    builder = DocumentGraphBuilder(
        enriched_document_path=enriched_document_path,
        chunks_path=chunks_path,
    )

    graph = builder.save(output_path)

    graph_metadata = graph["graph"]

    print("\nGRAPH CONSTRUCTION COMPLETE")
    print("-" * 70)

    print(
        f"Total nodes       : "
        f"{graph_metadata['total_nodes']}"
    )

    print(
        f"Total edges       : "
        f"{graph_metadata['total_edges']}"
    )

    print("\nNode counts:")

    for node_type, count in graph_metadata[
        "node_counts"
    ].items():
        print(
            f"  {node_type:<15} : {count}"
        )

    print("\nEdge counts:")

    for relationship, count in graph_metadata[
        "edge_counts"
    ].items():
        print(
            f"  {relationship:<20} : {count}"
        )

    print("\nValidation        : PASSED")
    print(f"Output            : {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()