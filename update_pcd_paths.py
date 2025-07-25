import os
import json
import shutil
import re
from pathlib import Path
from typing import Optional

def _update_path_number(path_str: Optional[str]) -> Optional[str]:
    """
    경로 문자열을 받아 파일명의 숫자에서 1을 뺀 후,
    결과를 10자리(zero-padded)로 포맷팅하여 새 경로 문자열을 반환합니다.
    """
    if not path_str:
        return None

    p = Path(path_str)
    stem = p.stem

    # 파일 이름에서 마지막 숫자 그룹을 찾습니다.
    match = re.search(r'(\d+)(?!.*\d)', stem)
    if not match:
        print(f"⚠️ 경고: '{stem}'에서 숫자를 찾을 수 없어 건너뜁니다.")
        return path_str

    old_num_str = match.group(1)
    old_num = int(old_num_str)
    new_num = old_num - 1

    if new_num < 0:
        print(f"⚠️ 경고: '{stem}'의 숫자를 1 빼면 음수가 되어 건너뜁니다.")
        return path_str

    # 새 숫자를 10자리 문자열로 포맷팅합니다.
    new_num_str = f"{new_num:010d}"
    
    # 새 파일 이름 생성 (기존 숫자 부분을 새 숫자로 교체)
    parts = stem.rsplit(old_num_str, 1)
    new_stem = f"{parts[0]}{new_num_str}{parts[1]}"
    
    # 새 경로 조합
    new_path = p.parent / (new_stem + p.suffix)
    
    return str(new_path).replace('\\', '/')

def update_all_paths_in_mapping():
    """
    'mapping_data.json' 파일에서 'pcd_original_path'와 'a5_original_path'의
    숫자에서 1을 빼고 10자리로 포맷팅하여 경로를 업데이트합니다.
    """
    mapping_file = Path('mapping_data.json')
    backup_file = Path('mapping_data.json.bak')

    if not mapping_file.exists():
        print(f"❌ 오류: '{mapping_file}' 파일이 현재 폴더에 없습니다.")
        return

    try:
        shutil.copy2(mapping_file, backup_file)
        print(f"✅ 1. 원본 파일을 '{backup_file}'(으)로 안전하게 백업했습니다.")
    except Exception as e:
        print(f"❌ 오류: 파일 백업에 실패했습니다. {e}")
        return

    try:
        with open(backup_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        modified_entries = 0
        for entry in data:
            # 두 경로 모두 업데이트
            new_pcd_path = _update_path_number(entry.get('pcd_original_path'))
            new_a5_path = _update_path_number(entry.get('a5_original_path'))
            
            # 하나라도 변경되었으면 카운트
            if (new_pcd_path != entry.get('pcd_original_path') or 
                new_a5_path != entry.get('a5_original_path')):
                modified_entries += 1

            entry['pcd_original_path'] = new_pcd_path
            entry['a5_original_path'] = new_a5_path

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 2. 총 {modified_entries}개 항목의 경로를 수정하여 '{mapping_file}'에 저장했습니다.")

    except Exception as e:
        print(f"❌ 오류: JSON 파일 수정 중 문제가 발생했습니다. {e}")
        return

    # --- 3. 결과 검증 ---
    try:
        print("\n--- 결과 검증 시작 ---")
        with open(backup_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            modified_data = json.load(f)

        errors = []
        for i, (orig_entry, mod_entry) in enumerate(zip(original_data, modified_data)):
            for key in ['pcd_original_path', 'a5_original_path']:
                orig_path_str = orig_entry.get(key)
                mod_path_str = mod_entry.get(key)

                if not orig_path_str or not mod_path_str:
                    continue

                orig_match = re.search(r'(\d+)(?!.*\d)', Path(orig_path_str).stem)
                mod_match = re.search(r'(\d+)(?!.*\d)', Path(mod_path_str).stem)

                if not orig_match or not mod_match:
                    continue

                orig_num = int(orig_match.group(1))
                mod_num = int(mod_match.group(1))

                if mod_num != orig_num - 1:
                    errors.append(f"  - 항목 {i}, 키 '{key}': 원본({orig_num}) -> 수정({mod_num}). -1 관계가 아닙니다.")
        
        if not errors:
            print(f"✅ 3. 검증 성공! 모든 경로가 정확히 -1 처리되었습니다.")
        else:
            print(f"❌ 3. 검증 실패! 아래 항목들을 확인하세요:")
            for error in errors:
                print(error)

    except Exception as e:
        print(f"❌ 오류: 결과 검증 중 문제가 발생했습니다. {e}")

if __name__ == "__main__":
    update_all_paths_in_mapping()
