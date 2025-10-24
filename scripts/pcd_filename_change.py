from pathlib import Path
import shutil
import re
import os

def extract_frame_number_from_img(filename):
    """'frame_000001.png'와 같은 이미지 파일 이름에서 프레임 번호를 추출합니다."""
    match = re.search(r'frame_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return None

def extract_number_from_pcd(filename):
    """PCD 파일 이름 ('..._716.pcd' 또는 '716.pcd')에서 마지막 숫자 부분을 추출합니다."""
    stem = Path(filename).stem
    numbers = re.findall(r'\d+', stem)
    if numbers:
        return int(numbers[-1])
    return None

def rename_pcd_files_based_on_images(folder_path, img_folder_name="image_a5"):
    """
    PCD 파일을 기준으로, 각 파일의 번호와 일치하는 이미지 프레임 번호를 찾아 이름을 변경합니다.
    각 PCD 파일은 단 한 번만 처리되며, 출력 파일 수는 입력 PCD 파일 수를 초과하지 않습니다.
    """
    folder = Path(folder_path)
    img_folder = folder / img_folder_name
    pcd_folder = folder / "pcd"
    output_folder = folder / "pcd_renamed"
    output_folder.mkdir(exist_ok=True)

    if not img_folder.exists():
        print(f"❌ 이미지 폴더가 존재하지 않습니다: {img_folder}")
        return
    if not pcd_folder.exists():
        print(f"❌ PCD 폴더가 존재하지 않습니다: {pcd_folder}")
        return

    # 1. 이미지 파일 목록을 {프레임 번호: 경로} 맵으로 미리 만들어둡니다.
    img_files_by_frame = {}
    for f in img_folder.iterdir():
        if f.is_file() and f.suffix.lower() == ".png":
            frame_num = extract_frame_number_from_img(f.name)
            if frame_num is not None:
                img_files_by_frame[frame_num] = f
    
    if not img_files_by_frame:
        print(f"❌ {img_folder_name} 폴더에 'frame_XXXXXX.png' 형식의 이미지 파일이 없습니다.")
        return
    print(f"ℹ️ 이미지 파일 {len(img_files_by_frame)}개 확인 완료.")

    # 2. PCD 폴더의 모든 파일을 순회하며 작업을 수행합니다.
    pcd_files = list(pcd_folder.glob('*.pcd'))
    print(f"ℹ️ PCD 파일 {len(pcd_files)}개 처리 시작...")
    
    success_count = 0
    unmatched_count = 0

    for pcd_path in pcd_files:
        pcd_num = extract_number_from_pcd(pcd_path.name)
        if pcd_num is None:
            print(f"⚠️ {pcd_path.name}에서 숫자를 추출할 수 없어 건너뜁니다.")
            unmatched_count += 1
            continue

        # 3. PCD 번호와 일치하는 이미지가 있는지 확인합니다.
        if pcd_num in img_files_by_frame:
            img_path = img_files_by_frame[pcd_num]
            pcd_num -= 1  # PCD 번호는 1부터 시작하므로, 이미지 프레임 번호와 맞추기 위해 1을 빼줍니다.
            # 새 파일명은 PCD 번호를 기준으로 생성합니다.
            new_name = f"{pcd_num:010d}.pcd"
            new_path = output_folder / new_name

            shutil.copy2(pcd_path, new_path)
            success_count += 1
            # print(f"✅ {pcd_path.name} → {new_name} (Matched with {img_path.name})")
        else:
            # print(f"❌ {pcd_path.name} (번호: {pcd_num}): 매칭되는 이미지 파일을 찾을 수 없습니다.")
            unmatched_count += 1

    print("\n--- 작업 요약 ---")
    print(f"총 {len(pcd_files)}개의 PCD 파일 중,")
    print(f"✅ {success_count}개의 파일 이름 변경 성공.")
    print(f"❌ {unmatched_count}개의 파일은 매칭되는 이미지를 찾지 못함.")
    print(f"결과물은 '{output_folder}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    folder_path = input("PCD 파일과 이미지 폴더가 있는 상위 폴더 경로를 입력하세요: ").strip().strip('"')
    rename_pcd_files_based_on_images(folder_path, img_folder_name="img")