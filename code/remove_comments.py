import ast
import io
import tokenize
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent


def _detect_comment_ranges(tokens):
    ranges = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            sl, sc = tok.start
            el, ec = tok.end
            ranges.append((sl, sc, el, ec))
    return ranges


def _is_str_expr(node):
    if isinstance(node, ast.Expr):
        val = node.value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return True
        if isinstance(val, ast.Str):
            return True
    return False


def _detect_docstring_ranges(source):
    ranges = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ranges

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for child in node.body:
                if _is_str_expr(child):
                    pos = _node_range(child)
                    if pos:
                        ranges.append(pos)

    return ranges


def _node_range(node):
    if hasattr(node, "end_lineno") and hasattr(node, "end_col_offset"):
        return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
    return None


def remove_comments_from_source(source: str) -> str:
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))

    comment_ranges = _detect_comment_ranges(tokens)
    docstring_ranges = _detect_docstring_ranges(source)

    all_ranges = comment_ranges + docstring_ranges
    all_ranges.sort(key=lambda x: (x[0], x[1]), reverse=True)

    lines = source.split("\n")

    for sl, sc, el, ec in all_ranges:
        li0 = sl - 1
        li1 = el - 1

        if li0 == li1:
            line = lines[li0]
            while sc > 0 and line[sc - 1] in (" ", "\t"):
                sc -= 1
            lines[li0] = line[:sc] + line[ec:]
        else:
            lines[li0] = lines[li0][:sc]
            for mid in range(li0 + 1, li1):
                lines[mid] = ""
            lines[li1] = lines[li1][ec:]

    result = "\n".join(lines)
    return result


def main():
    py_files = list(CODE_DIR.rglob("*.py"))
    py_files = [f for f in py_files if f.name != "remove_comments.py"]

    print(f"Found {len(py_files)} .py files")

    for fp in py_files:
        try:
            original = fp.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  Skipping {fp}: {e}")
            continue

        try:
            cleaned = remove_comments_from_source(original)
        except Exception as e:
            print(f"  Failed for {fp}: {e}")
            continue

        if cleaned != original:
            fp.write_text(cleaned, encoding="utf-8")
            print(f"  Cleaned: {fp.relative_to(CODE_DIR)}")
        else:
            print(f"  No change: {fp.relative_to(CODE_DIR)}")


if __name__ == "__main__":
    main()