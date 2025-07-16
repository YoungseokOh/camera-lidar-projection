from pathlib import Path

def save_file_list_to_parent(folder_path, output_txt="pcd_file_list.txt"):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 폴더가 존재하지 않습니다: {folder_path}")
        return False

    files = [f.name for f in folder.iterdir() if f.is_file()]
    parent_folder = folder.parent
    output_path = parent_folder / output_txt

    with open(output_path, "w", encoding="utf-8") as f:
        for filename in files:
            f.write(filename + "\n")

    print(f"✅ {len(files)}개 파일 이름을 {output_path}에 저장했습니다.")

    # txt 파일 라인 수 체크
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) == len(files):
        print("✅ txt 파일의 행 수와 폴더 내 파일 수가 일치합니다.")
    else:
        print(f"⚠️ txt 파일 행 수({len(lines)})와 실제 파일 수({len(files)})가 다릅니다.")

if __name__ == "__main__":
    folder_path = input("폴더 경로를 입력하세요: ").strip().strip('"')
    save_file_list_to_parent(folder_path)