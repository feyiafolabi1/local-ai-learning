from pathlib import Path

from pypdf import PdfReader


DOCUMENTS_DIR = (
    Path.home()
    / "local-ai-learning"
    / "05-vector-database"
    / "documents"
)


SEARCH_TERMS = [
    "Kimi Delta Attention",
    "Helios",
    "scale-up",
]


def main():
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)

        print("\n" + "=" * 80)
        print(pdf_path.name)
        print("=" * 80)

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text() or ""
            lower_text = text.lower()

            for term in SEARCH_TERMS:
                if term.lower() in lower_text:
                    print(
                        f"\nFound '{term}' "
                        f"on PDF page {page_number}"
                    )

                    position = lower_text.find(
                        term.lower()
                    )

                    start = max(0, position - 250)
                    end = min(
                        len(text),
                        position + 500,
                    )

                    snippet = text[start:end]

                    print(snippet)
                    print("-" * 80)


if __name__ == "__main__":
    main()