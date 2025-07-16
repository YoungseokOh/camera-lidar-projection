import os
from pathlib import Path

def analyze_folder_structure(root_path, max_depth=None, show_files=True, show_hidden=False):
    """
    폴더 구조를 분석하고 트리 형태로 출력합니다.
    
    Args:
        root_path (str): 분석할 루트 폴더 경로
        max_depth (int): 최대 탐색 깊이 (None이면 무제한)
        show_files (bool): 파일도 표시할지 여부
        show_hidden (bool): 숨김 파일/폴더도 표시할지 여부
    """
    
    def print_tree(path, prefix="", depth=0):
        if max_depth is not None and depth > max_depth:
            return
            
        try:
            items = list(Path(path).iterdir())
            if not show_hidden:
                items = [item for item in items if not item.name.startswith('.')]
            
            # 폴더와 파일을 분리하고 정렬
            folders = sorted([item for item in items if item.is_dir()])
            files = sorted([item for item in items if item.is_file()]) if show_files else []
            
            all_items = folders + files
            
            for i, item in enumerate(all_items):
                is_last = i == len(all_items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = prefix + ("    " if is_last else "│   ")
                
                # 파일 크기 정보 추가
                size_info = ""
                if item.is_file():
                    try:
                        size = item.stat().st_size
                        if size < 1024:
                            size_info = f" ({size} bytes)"
                        elif size < 1024 * 1024:
                            size_info = f" ({size/1024:.1f} KB)"
                        elif size < 1024 * 1024 * 1024:
                            size_info = f" ({size/(1024*1024):.1f} MB)"
                        else:
                            size_info = f" ({size/(1024*1024*1024):.1f} GB)"
                    except:
                        size_info = " (크기 정보 없음)"
                
                print(f"{prefix}{current_prefix}{item.name}{size_info}")
                
                # 폴더인 경우 재귀적으로 탐색
                if item.is_dir():
                    print_tree(item, next_prefix, depth + 1)
                    
        except PermissionError:
            print(f"{prefix}├── [접근 권한 없음]")
        except Exception as e:
            print(f"{prefix}├── [오류: {str(e)}]")

    # 루트 경로 정보 출력
    root = Path(root_path)
    if not root.exists():
        print(f"오류: 경로 '{root_path}'가 존재하지 않습니다.")
        return
    
    print(f"📁 폴더 구조 분석: {root_path}")
    print("=" * 50)
    print(f"{root.name}/")
    
    print_tree(root_path)
    
    # 통계 정보 출력
    print("\n" + "=" * 50)
    print("📊 통계 정보:")
    
    total_folders = 0
    total_files = 0
    total_size = 0
    
    try:
        for item in root.rglob('*'):
            if item.is_dir():
                total_folders += 1
            elif item.is_file():
                total_files += 1
                try:
                    total_size += item.stat().st_size
                except:
                    pass
    except:
        pass
    
    print(f"총 폴더 수: {total_folders}")
    print(f"총 파일 수: {total_files}")
    if total_size < 1024 * 1024 * 1024:
        print(f"총 크기: {total_size/(1024*1024):.1f} MB")
    else:
        print(f"총 크기: {total_size/(1024*1024*1024):.1f} GB")

def main():
    # 분석할 폴더 경로
    target_path = r"Y:\adasip\Temp\20250711_LC_test\20250711_A6_A5_LC_test\ncdb_a6_dataset"
    
    print("🔍 폴더 구조 분석 도구")
    print("=" * 50)
    
    # 기본 분석 (깊이 3까지, 파일 포함)
    print("\n1️⃣ 기본 구조 분석 (깊이 3까지):")
    analyze_folder_structure(target_path, max_depth=3, show_files=True)
    
    print("\n" + "="*80 + "\n")
    
    # 폴더만 분석 (더 깊이까지)
    print("2️⃣ 폴더 구조만 분석 (깊이 5까지):")
    analyze_folder_structure(target_path, max_depth=5, show_files=False)
    
    print("\n" + "="*80 + "\n")
    
    # 사용자 옵션
    print("3️⃣ 사용자 정의 분석:")
    try:
        depth = input("최대 깊이를 입력하세요 (Enter = 무제한): ")
        max_depth = int(depth) if depth.strip() else None
        
        show_files = input("파일도 표시하시겠습니까? (y/n, 기본값: y): ").lower()
        show_files = show_files != 'n'
        
        analyze_folder_structure(target_path, max_depth=max_depth, show_files=show_files)
        
    except KeyboardInterrupt:
        print("\n분석이 중단되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()