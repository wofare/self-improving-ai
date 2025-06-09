#!/usr/bin/env python3
import argparse
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Search Hugging Face datasets")
    parser.add_argument("--query", default="", help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Number of results")
    parser.add_argument("--show_license", action="store_true", help="Include license info")
    args = parser.parse_args()

    api = HfApi()
    datasets = list(api.list_datasets(search=args.query, limit=args.limit))
    for ds in datasets:
        line = ds.id
        if args.show_license:
            try:
                info = api.dataset_info(ds.id)
                if info.cardData and "license" in info.cardData:
                    line += f" ({info.cardData['license']})"
            except Exception:
                pass
        print(line)


if __name__ == "__main__":
    main()
