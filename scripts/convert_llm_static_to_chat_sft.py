import json
from pathlib import Path

import click

from src.llm.sft_format import chat_example_to_json, static_example_to_chat
from src.llm.static_dataset import StaticDatasetExample


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/llm_static_dataset.jsonl"),
    show_default=True,
    help="静的データセット(JSONL)のパス",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/llm_sft_chat_dataset.jsonl"),
    show_default=True,
    help="SFT 用チャット形式データセット(JSONL)の出力パス",
)
def main(input_path: Path, output_path: Path) -> None:
    """
    静的データセットを LLM-jp 向けチャット形式データセットに変換する。

    使い方:
        uv run python scripts/convert_llm_static_to_chat_sft.py \\
            --input data/llm_static_dataset.jsonl \\
            --output data/llm_sft_chat_dataset.jsonl
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if not line.strip():
                continue
            raw = json.loads(line)
            ex = StaticDatasetExample(**raw)
            chat_ex = static_example_to_chat(ex)
            fout.write(chat_example_to_json(chat_ex) + "\n")
            count += 1

    click.echo(f"wrote {count} chat examples to {output_path}")


if __name__ == "__main__":
    main()
