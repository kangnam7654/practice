from pathlib import Path
import json
from tqdm import tqdm

def main():
    DATA_DIR = "/home/kangnam/datasets/raw/ko_en_translation"
    data_path = Path(DATA_DIR)

    files = list(data_path.rglob("*.json"))
    
    # 파일을 루프 밖에서 한 번만 열기
    with open("./words.txt", "w", encoding="utf-8") as f:
        for file in tqdm(files):
            with open(file, "r", encoding="utf-8") as data_file:
                data = json.load(data_file)["data"]

            for datum in data:
                kor = datum["ko"]
                eng = datum["en"]
                f.write(kor + "\n")
                f.write(eng + "\n")

if __name__ == "__main__":
    main()
