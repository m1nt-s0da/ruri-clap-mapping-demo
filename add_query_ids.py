import argparse
import json
from pathlib import Path
from uuid import uuid7

DEFAULT_INPUT = Path(".var/generated-prompt.json")


def add_query_ids(input_path: Path) -> tuple[int, int]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    added_count = 0
    total_count = 0

    for record in data:
        for query in record.get("queries", []):
            total_count += 1
            if "id" not in query:
                query["id"] = str(uuid7())
                added_count += 1

    input_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return added_count, total_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add missing UUIDv7 values to query objects in a JSON dataset."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    added_count, total_count = add_query_ids(args.input)
    print(f"Added {added_count} UUIDv7 IDs to {total_count} queries.")


if __name__ == "__main__":
    main()
