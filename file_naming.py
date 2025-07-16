import os
import re
import shutil
from pathlib import Path

def extract_frame_number(filename):
    """파일명에서 프레임 번호를 추출합니다."""
    # frame_000001.jpg 형태에서 숫자 부분 추출
    match1 = re.search(r'frame_(\d+)', filename)
    if match1:
        return int(match1.group(1))
    
    # frame_count_1.jpg 형태에서 숫자 부분 추출
    match2 = re.search(r'frame_count_(\d+)', filename)
    if match2:
        return int(match2.group(1))
    
    # 기타 숫자만 있는 파일명 (예: 0000000001.jpg)
    match3 = re.search(r'^(\d+)\.[a-zA-Z]+$', filename)
    if match3:
        return int(match3.group(1))
    
    return None

def verify_image_sequence(folder_path):
    """이미지 시퀀스의 연속성과 순서를 검증합니다."""
    print("🔍 이미지 시퀀스 검증 중...")
    
    # 이미지 파일들 찾기
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file_path in Path(folder_path).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            frame_num = extract_frame_number(file_path.name)
            if frame_num is not None:
                image_files.append((frame_num, file_path))
    
    if not image_files:
        print("❌ 프레임 형태의 이미지 파일을 찾을 수 없습니다.")
        return None
    
    # 프레임 번호로 정렬
    image_files.sort(key=lambda x: x[0])
    
    print(f"📊 총 {len(image_files)}개의 이미지 파일 발견")
    print(f"🔢 프레임 범위: {image_files[0][0]} ~ {image_files[-1][0]}")
    
    # 연속성 검증
    frame_numbers = [frame_num for frame_num, _ in image_files]
    missing_frames = []
    
    for i in range(image_files[0][0], image_files[-1][0] + 1):
        if i not in frame_numbers:
            missing_frames.append(i)
    
    if missing_frames:
        print(f"⚠️ 누락된 프레임들: {missing_frames[:10]}" + 
              (f" (외 {len(missing_frames)-10}개)" if len(missing_frames) > 10 else ""))
        response = input("누락된 프레임이 있습니다. 계속 진행하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            return None
    else:
        print("✅ 모든 프레임이 연속적으로 존재합니다.")
    
    # 중복 검증
    duplicates = []
    seen_frames = set()
    for frame_num, file_path in image_files:
        if frame_num in seen_frames:
            duplicates.append(frame_num)
        seen_frames.add(frame_num)
    
    if duplicates:
        print(f"❌ 중복된 프레임 번호들: {duplicates}")
        return None
    
    print("✅ 중복된 프레임이 없습니다.")
    
    # 처음 몇 개와 마지막 몇 개 파일 표시
    print("\n📝 파일 순서 미리보기:")
    preview_count = min(5, len(image_files))
    
    print("처음 파일들:")
    for i in range(preview_count):
        frame_num, file_path = image_files[i]
        print(f"  {i:2d}: frame_{frame_num:06d} -> {file_path.name}")
    
    if len(image_files) > preview_count * 2:
        print("  ...")
    
    print("마지막 파일들:")
    for i in range(max(preview_count, len(image_files) - preview_count), len(image_files)):
        frame_num, file_path = image_files[i]
        new_index = i
        print(f"  {new_index:2d}: frame_{frame_num:06d} -> {file_path.name}")
    
    return image_files

def rename_images_to_zero_based(folder_path, dry_run=True):
    """이미지들을 0부터 시작하는 10자리 숫자로 이름을 변경합니다."""
    
    # 시퀀스 검증
    image_files = verify_image_sequence(folder_path)
    if image_files is None:
        return False
    
    print(f"\n{'🧪 테스트 모드' if dry_run else '🔄 실제 변경 모드'}")
    print("=" * 50)
    
    # 백업 폴더 생성 (실제 실행 시)
    backup_folder = None
    if not dry_run:
        backup_folder = Path(folder_path) / "backup_original_names"
        backup_folder.mkdir(exist_ok=True)
        print(f"📁 백업 폴더 생성: {backup_folder}")
    
    # 변경 계획 생성
    rename_plan = []
    for new_index, (original_frame_num, file_path) in enumerate(image_files):
        new_name = f"{new_index:010d}{file_path.suffix}"
        new_path = file_path.parent / new_name
        
        rename_plan.append({
            'original_path': file_path,
            'new_path': new_path,
            'original_frame': original_frame_num,
            'new_index': new_index
        })
    
    # 변경 계획 미리보기
    print("\n📋 변경 계획:")
    preview_count = min(10, len(rename_plan))
    
    for i in range(preview_count):
        plan = rename_plan[i]
        print(f"  {plan['original_path'].name} -> {plan['new_path'].name}")
    
    if len(rename_plan) > preview_count:
        print(f"  ... (외 {len(rename_plan) - preview_count}개 파일)")
    
    if dry_run:
        print("\n⚠️ 테스트 모드입니다. 실제 파일은 변경되지 않습니다.")
        response = input("실제로 파일 이름을 변경하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            return rename_images_to_zero_based(folder_path, dry_run=False)
        return True
    
    # 실제 파일 이름 변경
    print(f"\n🔄 {len(rename_plan)}개 파일 이름 변경 시작...")
    
    success_count = 0
    error_count = 0
    
    try:
        # 임시 이름으로 먼저 변경 (충돌 방지)
        temp_plans = []
        for i, plan in enumerate(rename_plan):
            temp_name = f"temp_{i:010d}{plan['original_path'].suffix}"
            temp_path = plan['original_path'].parent / temp_name
            
            try:
                # 원본을 백업에 복사
                backup_path = backup_folder / plan['original_path'].name
                shutil.copy2(plan['original_path'], backup_path)
                
                # 임시 이름으로 변경
                plan['original_path'].rename(temp_path)
                temp_plans.append((temp_path, plan['new_path']))
                
            except Exception as e:
                print(f"❌ 오류 발생: {plan['original_path'].name} -> {e}")
                error_count += 1
        
        # 임시 이름에서 최종 이름으로 변경
        for temp_path, final_path in temp_plans:
            try:
                temp_path.rename(final_path)
                success_count += 1
                if success_count % 100 == 0:
                    print(f"  진행률: {success_count}/{len(rename_plan)}")
                    
            except Exception as e:
                print(f"❌ 최종 변경 오류: {temp_path.name} -> {e}")
                error_count += 1
        
        print(f"\n✅ 변경 완료: 성공 {success_count}개, 실패 {error_count}개")
        
        if error_count == 0:
            print("🎉 모든 파일이 성공적으로 변경되었습니다!")
        else:
            print("⚠️ 일부 파일 변경에 실패했습니다. 백업을 확인해주세요.")
            
    except Exception as e:
        print(f"❌ 심각한 오류 발생: {e}")
        return False
    
    return error_count == 0

def main():
    print("🖼️ 이미지 파일 이름 변경 도구")
    print("=" * 50)
    
    # 폴더 경로 설정
    folder_path = input("이미지 폴더 경로를 입력하세요: ").strip().strip('"')
    
    if not folder_path:
        # 기본 경로 사용 (필요시 수정)
        folder_path = r"Y:\adasip\Temp\20250711_LC_test\20250711_A6_A5_LC_test\ncdb_a6_dataset"
        print(f"기본 경로 사용: {folder_path}")
    
    if not Path(folder_path).exists():
        print(f"❌ 폴더가 존재하지 않습니다: {folder_path}")
        return
    
    print(f"📁 대상 폴더: {folder_path}")
    
    # 테스트 모드로 먼저 실행
    success = rename_images_to_zero_based(folder_path, dry_run=True)
    
    if success:
        print("\n✅ 파일 이름 변경이 완료되었습니다!")
    else:
        print("\n❌ 파일 이름 변경에 실패했습니다.")

if __name__ == "__main__":
    main()