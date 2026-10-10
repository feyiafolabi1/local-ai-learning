from pathlib import Path
from dataclasses import dataclass
from typing import List
import ast
import re


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

REPO_ROOT = Path.home() / "local-ai-learning"

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "18-final-ai-app",
}

SUPPORTED_SUFFIXES = {
    ".md",
    ".py",
}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# --------------------------------------------------
# DATA STRUCTURE
# --------------------------------------------------

@dataclass
class DocumentChunk:
    text: str
    source_path: str
    chapter: str
    file_name: str
    file_type: str
    chunk_index: int
    section_name: str


# --------------------------------------------------
# FILE DISCOVERY
# --------------------------------------------------

def should_skip(path: Path) -> bool:
    return any(
        part in EXCLUDED_DIRS
        for part in path.parts
    )


def discover_files(repo_root: Path) -> List[Path]:
    files = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue

        if should_skip(path):
            continue

        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        files.append(path)

    return sorted(files)


# --------------------------------------------------
# METADATA
# --------------------------------------------------

def get_chapter_name(
    path: Path,
    repo_root: Path,
) -> str:

    relative_path = path.relative_to(repo_root)

    if len(relative_path.parts) > 1:
        return relative_path.parts[0]

    return "repo-root"


def get_file_type(path: Path) -> str:
    if path.suffix.lower() == ".md":
        return "markdown"

    if path.suffix.lower() == ".py":
        return "python"

    return "unknown"


# --------------------------------------------------
# FILE READING
# --------------------------------------------------

def read_file(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


# --------------------------------------------------
# LINE-AWARE FALLBACK CHUNKING
# --------------------------------------------------

def chunk_by_lines(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split oversized text into chunks without cutting
    through the middle of individual lines.

    This is used only when a logical Markdown or Python
    block is too large to keep as one chunk.
    """

    if not text.strip():
        return []

    lines = text.splitlines()

    chunks = []
    current_lines = []
    current_size = 0

    for line in lines:

        line_size = len(line) + 1

        if (
            current_lines
            and current_size + line_size > chunk_size
        ):
            chunk = "\n".join(
                current_lines
            ).strip()

            if chunk:
                chunks.append(chunk)

            # Preserve approximately `overlap`
            # characters worth of COMPLETE lines.
            overlap_lines = []
            overlap_size = 0

            for previous_line in reversed(current_lines):

                previous_size = len(previous_line) + 1

                if (
                    overlap_lines
                    and overlap_size + previous_size > overlap
                ):
                    break

                overlap_lines.insert(
                    0,
                    previous_line,
                )

                overlap_size += previous_size

            current_lines = overlap_lines
            current_size = sum(
                len(existing_line) + 1
                for existing_line in current_lines
            )

        current_lines.append(line)
        current_size += line_size

    if current_lines:

        final_chunk = "\n".join(
            current_lines
        ).strip()

        if final_chunk:
            chunks.append(final_chunk)

    return chunks


# --------------------------------------------------
# MARKDOWN CHUNKING
# --------------------------------------------------

def split_markdown_sections(
    text: str,
) -> List[tuple[str, str]]:
    """
    Split Markdown according to headings.

    Example:

    ## PagedAttention
    explanation...

    becomes one logical section.
    """

    lines = text.splitlines()

    sections = []

    current_heading = "markdown-preamble"
    current_lines = []

    heading_pattern = re.compile(
        r"^#{1,6}\s+"
    )

    for line in lines:

        if heading_pattern.match(line):

            if current_lines:

                section_text = "\n".join(
                    current_lines
                ).strip()

                if section_text:
                    sections.append(
                        (
                            current_heading,
                            section_text,
                        )
                    )

            current_heading = line.strip()
            current_lines = [line]

        else:
            current_lines.append(line)

    if current_lines:

        section_text = "\n".join(
            current_lines
        ).strip()

        if section_text:
            sections.append(
                (
                    current_heading,
                    section_text,
                )
            )

    return sections


def chunk_markdown(
    text: str,
) -> List[tuple[str, str]]:
    """
    Keep Markdown heading sections intact whenever
    possible.

    Large sections are split on complete lines.
    """

    output = []

    sections = split_markdown_sections(text)

    for heading, section_text in sections:

        if len(section_text) <= CHUNK_SIZE:

            output.append(
                (
                    heading,
                    section_text,
                )
            )

        else:

            subchunks = chunk_by_lines(
                section_text
            )

            for i, subchunk in enumerate(subchunks):

                output.append(
                    (
                        f"{heading} [part {i + 1}]",
                        subchunk,
                    )
                )

    return output


# --------------------------------------------------
# PYTHON HELPERS
# --------------------------------------------------

def get_node_source(
    node: ast.AST,
    lines: List[str],
) -> str:
    """
    Extract the original source code represented by
    an AST node.
    """

    start_line = getattr(
        node,
        "lineno",
        None,
    )

    end_line = getattr(
        node,
        "end_lineno",
        None,
    )

    if start_line is None:
        return ""

    if end_line is None:
        end_line = start_line

    return "\n".join(
        lines[
            start_line - 1:
            end_line
        ]
    ).strip()


def get_assignment_name(
    node: ast.AST,
) -> str:
    """
    Return a readable assignment target name.

    Examples:

    MODELS = [...]
    → MODELS

    x: int = 5
    → x
    """

    if isinstance(node, ast.Assign):

        names = []

        for target in node.targets:

            if isinstance(target, ast.Name):
                names.append(target.id)

            else:
                names.append(
                    ast.unparse(target)
                )

        return ", ".join(names)

    if isinstance(node, ast.AnnAssign):

        if isinstance(node.target, ast.Name):
            return node.target.id

        return ast.unparse(node.target)

    return "assignment"


def get_assignment_value(
    node: ast.AST,
):

    if isinstance(node, ast.Assign):
        return node.value

    if isinstance(node, ast.AnnAssign):
        return node.value

    return None


# --------------------------------------------------
# LARGE PYTHON COLLECTIONS
# --------------------------------------------------

def split_large_sequence_assignment(
    assignment_name: str,
    value_node: ast.AST,
    lines: List[str],
) -> List[tuple[str, str]]:
    """
    Split large list/tuple/set assignments according
    to their individual elements.

    Example:

    TASKS = [
        {...},
        {...},
        {...},
    ]

    becomes:

    assignment TASKS item 1
    assignment TASKS item 2
    assignment TASKS item 3

    This is especially useful for benchmark prompt
    collections.
    """

    elements = getattr(
        value_node,
        "elts",
        [],
    )

    output = []

    for index, element in enumerate(elements):

        element_text = get_node_source(
            element,
            lines,
        )

        if not element_text:
            continue

        section_name = (
            f"assignment {assignment_name} "
            f"item {index + 1}"
        )

        if len(element_text) <= CHUNK_SIZE:

            output.append(
                (
                    section_name,
                    element_text,
                )
            )

        else:

            subchunks = chunk_by_lines(
                element_text
            )

            for part_index, subchunk in enumerate(subchunks):

                output.append(
                    (
                        (
                            f"{section_name} "
                            f"[part {part_index + 1}]"
                        ),
                        subchunk,
                    )
                )

    return output


def split_large_dict_assignment(
    assignment_name: str,
    value_node: ast.Dict,
    lines: List[str],
) -> List[tuple[str, str]]:
    """
    Split a large dictionary assignment into its
    individual key/value pairs.
    """

    output = []

    for index, (
        key_node,
        value_item,
    ) in enumerate(
        zip(
            value_node.keys,
            value_node.values,
        )
    ):

        key_text = (
            get_node_source(
                key_node,
                lines,
            )
            if key_node is not None
            else "<unpacked>"
        )

        value_text = get_node_source(
            value_item,
            lines,
        )

        pair_text = (
            f"{key_text}: {value_text}"
        )

        section_name = (
            f"assignment {assignment_name} "
            f"entry {index + 1}"
        )

        if len(pair_text) <= CHUNK_SIZE:

            output.append(
                (
                    section_name,
                    pair_text,
                )
            )

        else:

            subchunks = chunk_by_lines(
                pair_text
            )

            for part_index, subchunk in enumerate(subchunks):

                output.append(
                    (
                        (
                            f"{section_name} "
                            f"[part {part_index + 1}]"
                        ),
                        subchunk,
                    )
                )

    return output


# --------------------------------------------------
# PYTHON NODE PROCESSING
# --------------------------------------------------

def chunk_python_assignment(
    node: ast.AST,
    lines: List[str],
) -> List[tuple[str, str]]:
    """
    Preserve normal assignments as single logical
    chunks.

    Large collection assignments are split according
    to collection structure instead of arbitrary
    character boundaries.
    """

    assignment_name = get_assignment_name(
        node
    )

    block_text = get_node_source(
        node,
        lines,
    )

    value_node = get_assignment_value(
        node
    )

    if len(block_text) <= CHUNK_SIZE:

        return [
            (
                f"assignment {assignment_name}",
                block_text,
            )
        ]

    if isinstance(
        value_node,
        (
            ast.List,
            ast.Tuple,
            ast.Set,
        ),
    ):

        split_items = split_large_sequence_assignment(
            assignment_name,
            value_node,
            lines,
        )

        if split_items:
            return split_items

    if isinstance(
        value_node,
        ast.Dict,
    ):

        split_entries = split_large_dict_assignment(
            assignment_name,
            value_node,
            lines,
        )

        if split_entries:
            return split_entries

    # Large string/constants or other assignment types.
    # Keep complete lines rather than slicing arbitrary
    # characters.
    return [
        (
            (
                f"assignment {assignment_name} "
                f"[part {i + 1}]"
            ),
            subchunk,
        )
        for i, subchunk in enumerate(
            chunk_by_lines(block_text)
        )
    ]


def chunk_python_function_or_class(
    node: ast.AST,
    lines: List[str],
) -> List[tuple[str, str]]:

    block_text = get_node_source(
        node,
        lines,
    )

    if isinstance(
        node,
        ast.AsyncFunctionDef,
    ):
        section_name = (
            f"async function {node.name}"
        )

    elif isinstance(
        node,
        ast.FunctionDef,
    ):
        section_name = (
            f"function {node.name}"
        )

    elif isinstance(
        node,
        ast.ClassDef,
    ):
        section_name = (
            f"class {node.name}"
        )

    else:
        section_name = "python-block"

    if len(block_text) <= CHUNK_SIZE:

        return [
            (
                section_name,
                block_text,
            )
        ]

    return [
        (
            (
                f"{section_name} "
                f"[part {i + 1}]"
            ),
            subchunk,
        )
        for i, subchunk in enumerate(
            chunk_by_lines(block_text)
        )
    ]


# --------------------------------------------------
# PYTHON CHUNKING
# --------------------------------------------------

def chunk_python(
    text: str,
) -> List[tuple[str, str]]:
    """
    Parse the Python file as ordered top-level AST
    blocks.

    Structure preserved:

    imports
    assignments
    functions
    classes
    other module-level statements
    """

    try:
        tree = ast.parse(text)

    except SyntaxError:

        return [
            (
                f"python-file [part {i + 1}]",
                chunk,
            )
            for i, chunk in enumerate(
                chunk_by_lines(text)
            )
        ]

    lines = text.splitlines()

    output = []

    pending_imports = []


    def flush_imports():
        nonlocal pending_imports

        if not pending_imports:
            return

        import_text = "\n".join(
            get_node_source(
                import_node,
                lines,
            )
            for import_node in pending_imports
        ).strip()

        if import_text:

            if len(import_text) <= CHUNK_SIZE:

                output.append(
                    (
                        "imports",
                        import_text,
                    )
                )

            else:

                for i, subchunk in enumerate(
                    chunk_by_lines(import_text)
                ):
                    output.append(
                        (
                            f"imports [part {i + 1}]",
                            subchunk,
                        )
                    )

        pending_imports = []


    for node in tree.body:

        # ------------------------------------------
        # IMPORTS
        # ------------------------------------------

        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
            ),
        ):

            pending_imports.append(node)
            continue

        flush_imports()

        # ------------------------------------------
        # ASSIGNMENTS
        # ------------------------------------------

        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):

            output.extend(
                chunk_python_assignment(
                    node,
                    lines,
                )
            )

            continue

        # ------------------------------------------
        # FUNCTIONS / CLASSES
        # ------------------------------------------

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):

            output.extend(
                chunk_python_function_or_class(
                    node,
                    lines,
                )
            )

            continue

        # ------------------------------------------
        # OTHER TOP-LEVEL NODES
        # ------------------------------------------

        block_text = get_node_source(
            node,
            lines,
        )

        if not block_text:
            continue

        node_type = type(node).__name__

        section_name = (
            f"module-level {node_type}"
        )

        if len(block_text) <= CHUNK_SIZE:

            output.append(
                (
                    section_name,
                    block_text,
                )
            )

        else:

            for i, subchunk in enumerate(
                chunk_by_lines(block_text)
            ):

                output.append(
                    (
                        (
                            f"{section_name} "
                            f"[part {i + 1}]"
                        ),
                        subchunk,
                    )
                )

    flush_imports()

    return output


# --------------------------------------------------
# FILE-TYPE-AWARE CHUNKING
# --------------------------------------------------

def chunk_file(
    text: str,
    file_type: str,
) -> List[tuple[str, str]]:

    if file_type == "markdown":
        return chunk_markdown(text)

    if file_type == "python":
        return chunk_python(text)

    return [
        (
            f"fallback [part {i + 1}]",
            chunk,
        )
        for i, chunk in enumerate(
            chunk_by_lines(text)
        )
    ]


# --------------------------------------------------
# INGESTION
# --------------------------------------------------

def ingest_repo(
    repo_root: Path,
) -> List[DocumentChunk]:

    files = discover_files(
        repo_root
    )

    all_chunks = []

    for path in files:

        text = read_file(path)

        file_type = get_file_type(
            path
        )

        chunks = chunk_file(
            text,
            file_type,
        )

        chapter = get_chapter_name(
            path,
            repo_root,
        )

        relative_path = str(
            path.relative_to(
                repo_root
            )
        )

        for chunk_index, (
            section_name,
            chunk_text,
        ) in enumerate(chunks):

            all_chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source_path=relative_path,
                    chapter=chapter,
                    file_name=path.name,
                    file_type=file_type,
                    chunk_index=chunk_index,
                    section_name=section_name,
                )
            )

    return all_chunks


# --------------------------------------------------
# REPORTING
# --------------------------------------------------

def print_summary(
    chunks: List[DocumentChunk],
):

    if not chunks:
        print("No chunks were created.")
        return

    unique_files = {
        chunk.source_path
        for chunk in chunks
    }

    chapters = sorted({
        chunk.chapter
        for chunk in chunks
    })

    markdown_chunks = sum(
        1
        for chunk in chunks
        if chunk.file_type == "markdown"
    )

    python_chunks = sum(
        1
        for chunk in chunks
        if chunk.file_type == "python"
    )

    total_characters = sum(
        len(chunk.text)
        for chunk in chunks
    )

    average_chunk_size = (
        total_characters
        / len(chunks)
    )

    print()
    print("=" * 70)
    print("REPO INGESTION SUMMARY")
    print("=" * 70)

    print(
        f"Repo root:            "
        f"{REPO_ROOT}"
    )

    print(
        f"Files discovered:     "
        f"{len(unique_files)}"
    )

    print(
        f"Chapters discovered:  "
        f"{len(chapters)}"
    )

    print(
        f"Total chunks:         "
        f"{len(chunks)}"
    )

    print(
        f"Markdown chunks:      "
        f"{markdown_chunks}"
    )

    print(
        f"Python chunks:        "
        f"{python_chunks}"
    )

    print(
        f"Average chunk size:   "
        f"{average_chunk_size:.1f} characters"
    )

    print()
    print("=" * 70)
    print("DISCOVERED CHAPTERS / TOP-LEVEL FOLDERS")
    print("=" * 70)

    for chapter in chapters:
        print(chapter)


def print_sample_chunks(
    chunks: List[DocumentChunk],
    count: int = 15,
):

    print()
    print("=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    for chunk in chunks[:count]:

        print()
        print("-" * 70)

        print(
            f"Chapter:      {chunk.chapter}"
        )

        print(
            f"File:         {chunk.source_path}"
        )

        print(
            f"Type:         {chunk.file_type}"
        )

        print(
            f"Section:      {chunk.section_name}"
        )

        print(
            f"Chunk index:  {chunk.chunk_index}"
        )

        print()

        preview = chunk.text[:700]

        print(preview)

        if len(chunk.text) > 700:
            print("...")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print()
    print(
        f"Scanning repository:\n"
        f"{REPO_ROOT}"
    )

    chunks = ingest_repo(
        REPO_ROOT
    )

    print_summary(
        chunks
    )

    print_sample_chunks(
        chunks,
        count=15,
    )


if __name__ == "__main__":
    main()