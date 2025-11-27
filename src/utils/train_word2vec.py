import csv
import glob
import logging
import os

from gensim.models import Word2Vec

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def get_pokemon_name_with_form(name, form, item):
    """
    Returns the Pokémon name with its form.

    special case
    - ザマゼンタ ... ザマゼンタ(れきせん) か ザマゼンタ(たてのおう)
    - ザシアン ... ザシアン(れきせん) か ザシアン(けんのおう)
    - テラパゴス ... テラパゴス(テラスタル)
    - ウーラオス ... ウーラオス(いちげき) か ウーラオス(れんげき)
    - ネクロズマ ... ネクロズマ(たそがれ) か ネクロズマ(あかつき)
    - バドレックス ... バドレックス(こくば) か バドレックス(はくば)
    - ランドロス ... ランドロス(けしん) か ランドロス(れいじゅう)
    - オーガポン ... オーガポン(かまど) か オーガポン(いしずえ) か オーガポン(いど)
    - ガチグマ ... ガチグマ(アカツキ) か ガチグマ
    - ギラティナ ... ギラティナ(オリジン) か ギラティナ(アナザー)
    - ガラル、ヒスイ、アローラ は prefix につける

    Args:
        name (str): The base name of the Pokémon.
        form (str): The form of the Pokémon.

    Returns:
        str: The full name of the Pokémon with its form.
    """

    if "テラパゴス" in name:
        return "テラパゴス(テラスタル)"

    if "ザマゼンタ" in name:
        if item == "くちたたて":
            return "ザマゼンタ(たてのおう)"
        else:
            return "ザマゼンタ(れきせん)"
    if "ザシアン" in name:
        if item == "くちたけん":
            return "ザシアン(けんのおう)"
        else:
            return "ザシアン(れきせん)"

    if "ウーラオス" in name:
        if form == "いちげきのかた":
            return "ウーラオス(いちげき)"
        elif form == "れんげきのかた":
            return "ウーラオス(れんげき)"

    if "ネクロズマ" in name:
        if form == "たそがれのたてがみ":
            return "ネクロズマ(たそがれ)"
        elif form == "あかつきのつばさ":
            return "ネクロズマ(あかつき)"

    if "バドレックス" in name:
        if form == "こくばじょうのすがた":
            return "バドレックス(こくば)"
        elif form == "はくばじょうのすがた":
            return "バドレックス(はくば)"

    if "ランドロス" in name:
        if form == "けしんフォルム":
            return "ランドロス(けしん)"
        elif form == "れいじゅうフォルム":
            return "ランドロス(れいじゅう)"

    if "オーガポン" in name:
        if form == "かまどのめん":
            return "オーガポン(かまど)"
        elif form == "いしずえのめん":
            return "オーガポン(いしずえ)"
        elif form == "いどのめん":
            return "オーガポン(いど)"
        else:
            return "オーガポン"

    if "ガチグマ" in name:
        if form == "アカツキのすがた":
            return "ガチグマ(アカツキ)"
        else:
            return "ガチグマ"

    if "ギラティナ" in name:
        if form == "オリジンフォルム":
            return "ギラティナ(オリジン)"
        elif form == "アナザーフォルム":
            return "ギラティナ(アナザー)"
        else:
            return "ギラティナ"

    if "ロトム" in name:
        return form

    if "ルガルガン" in name:
        if form == "まひるのすがた":
            return "ルガルガン(まひる)"
        elif form == "まよなかのすがた":
            return "ルガルガン(まよなか)"
        else:
            return "ルガルガン(たそがれ)"

    if "ポリゴン２" in name:
        return "ポリゴン2"

    if "モルペコ" in name:
        return "モルペコ(まんぷく)"

    # Handle forms for other Pokémon
    # Check if the form is a valid prefix
    if form in ["アローラのすがた", "ヒスイのすがた", "ガラルのすがた"]:
        # のすがた、を除外して prefixをつける
        prefix = form.replace("のすがた", "")
        # Return the Pokémon name with the prefix
        return f"{prefix}{name}"

    # For other Pokémon, just return the name
    return name


def extract_teams(input_csv_dir):
    """
    Extracts 6 Pokémon teams from the CSV file and saves them as sentences for Word2Vec training.

    Args:
        input_csv_dir (str): Path to the input CSV file dir.
    """

    # Get all CSV files in the directory
    input_csv_files = glob.glob(input_csv_dir)

    team_sentences = []
    for input_csv_path in input_csv_files:
        # Read each CSV file
        with open(input_csv_path, mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Extract Pokémon names for the 6 slots
                team = [
                    get_pokemon_name_with_form(
                        row["ポケモン_1"], row["フォルム_1"], row["もちもの_1"]
                    ),
                    get_pokemon_name_with_form(
                        row["ポケモン_2"], row["フォルム_2"], row["もちもの_2"]
                    ),
                    get_pokemon_name_with_form(
                        row["ポケモン_3"], row["フォルム_3"], row["もちもの_3"]
                    ),
                    get_pokemon_name_with_form(
                        row["ポケモン_4"], row["フォルム_4"], row["もちもの_4"]
                    ),
                    get_pokemon_name_with_form(
                        row["ポケモン_5"], row["フォルム_5"], row["もちもの_5"]
                    ),
                    get_pokemon_name_with_form(
                        row["ポケモン_6"], row["フォルム_6"], row["もちもの_6"]
                    ),
                ]
                team_sentences.append(team)

    return team_sentences


def train_word2vec_model(team_sentences, model_output_path):
    """
    Main function to train the Word2Vec model.
    """
    set_team_sentences = [
        v.split() for v in set([" ".join(sorted(v)) for v in team_sentences])
    ]

    # Output Word2Vec model path
    model_output_path = "models/word2vec_model.bin"

    # Train the Word2Vec model
    model = Word2Vec(
        set_team_sentences,
        vector_size=300,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=10,
    )
    # Save the model
    model.save(model_output_path)


def main():
    # Input CSV file path
    input_csv_dir = "data/battle_database/*"
    model_output_path = "models/word2vec_model.bin"

    # Extract teams and save as Word2Vec training data
    team_sentences = extract_teams(input_csv_dir)
    print(f"Word2Vec training data extracted with {len(team_sentences)} teams.")

    train_word2vec_model(team_sentences, model_output_path)
    print(f"Word2Vec model trained and saved to {model_output_path}")


if __name__ == "__main__":
    main()
