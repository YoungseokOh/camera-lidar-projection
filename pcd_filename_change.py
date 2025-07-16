from pathlib import Path
import shutil
import re

def extract_index(filename):
    """파일명에서 마지막 언더바 뒤 숫자 추출 (예: 2602_465659778_29.pcd → 29)"""
    match = re.search(r'_(\d+)\.pcd$', filename)
    if match:
        return int(match.group(1))
    return None

def rename_pcd_files(folder_path):
    folder = Path(folder_path)
    pcd_folder = folder / "pcd"
    file_list_path = folder / "pcd_file_list.txt"
    output_folder = folder / "pcd_renamed"
    output_folder.mkdir(exist_ok=True)

    if not pcd_folder.exists():
        print(f"❌ pcd 폴더가 존재하지 않습니다: {pcd_folder}")
        return
    if not file_list_path.exists():
        print(f"❌ pcd_file_list.txt 파일이 없습니다: {file_list_path}")
        return

    # 파일 리스트 읽기
    with open(file_list_path, "r", encoding="utf-8") as f:
        file_names = [line.strip() for line in f if line.strip()]

    # pcd 폴더 내 파일 목록
    pcd_files = {f.name: f for f in pcd_folder.iterdir() if f.is_file() and f.suffix.lower() == ".pcd"}

    success_count = 0
    for idx, name in enumerate(file_names):
        # 확장자 제거
        base_name = name.split('.')[0]
        # pcd 폴더에서 일치하는 파일 찾기
        matched_file = None
        for fname in pcd_files:
            if fname.startswith(base_name):
                matched_file = pcd_files[fname]
                break
        if not matched_file:
            print(f"⚠️ {name}에 해당하는 .pcd 파일을 찾을 수 없습니다.")
            continue

        # 인덱스 추출
        file_index = extract_index(matched_file.name)
        if file_index is None:
            print(f"⚠️ {matched_file.name}에서 인덱스를 추출할 수 없습니다.")
            continue

        # 새 파일명 생성 (10자리수)
        new_name = f"{file_index:010d}.pcd"
        new_path = output_folder / new_name

        # 파일 복사
        shutil.copy2(matched_file, new_path)
        success_count += 1
        print(f"✅ {matched_file.name} → {new_name}")

    print(f"\n총 {success_count}개 파일을 {output_folder}에 저장 완료.")

if __name__ == "__main__":
    folder_path = input("폴더 경로를 입력하세요: ").strip().strip('"')
    rename_pcd_files(folder_path)