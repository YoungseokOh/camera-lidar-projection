import os
import re
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

class CameraSyncAnalyzer:
    def __init__(self):
        pass

    def read_frame_offset(self, dataset_path):
        """frame_offset.txt에서 프레임 오프셋 정보 읽기"""
        offset_file = Path(dataset_path) / "frame_offset.txt"
        if not offset_file.exists():
            print(f"❌ frame_offset.txt 파일을 찾을 수 없습니다: {offset_file}")
            return None
        offsets = {}
        with open(offset_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    offsets[key] = int(value)
        print("📄 frame_offset.txt 읽기 완료:", offsets)
        return offsets

    def get_frame_pairs(self, offsets, num_samples=10, random_mode=False):
        """A6 기준으로 A5와 매핑되는 프레임 쌍 반환 (랜덤/균등)"""
        a6_start, a6_end = offsets['a6_start'], offsets['a6_end']
        a5_start, a5_end = offsets['a5_start'], offsets['a5_end']
        a6_total = a6_end - a6_start + 1
        a5_total = a5_end - a5_start + 1
        ratio = a5_total / a6_total

        if random_mode:
            # 실제 프레임 번호를 랜덤으로 선택
            a6_frames = sorted(random.sample(range(a6_start, a6_end + 1), num_samples))
        else:
            # 균등하게 실제 프레임 번호 선택
            a6_frames = np.linspace(a6_start, a6_end, num_samples, dtype=int)

        pairs = []
        for idx, a6_frame in enumerate(a6_frames):
            # A6의 상대 인덱스
            rel_idx = a6_frame - a6_start
            # 비율로 A5 프레임 계산
            a5_frame = a5_start + int(rel_idx * ratio)
            if a5_frame <= a5_end:
                pairs.append({'idx': idx, 'a6': a6_frame, 'a5': a5_frame})
        print(f"\n🎯 매핑된 프레임 쌍 ({len(pairs)}개):")
        for p in pairs:
            print(f"  샘플 {p['idx']:2d}: A6[{p['a6']}] <-> A5[{p['a5']}]")
        return pairs

    def get_frame_pairs_reverse(self, offsets, num_samples=10, random_mode=False):
        """A5 기준으로 A6와 매핑되는 프레임 쌍 반환 (비율을 그대로 적용)"""
        a6_start, a6_end = offsets['a6_start'], offsets['a6_end']
        a5_start, a5_end = offsets['a5_start'], offsets['a5_end']
        a6_total = a6_end - a6_start + 1
        a5_total = a5_end - a5_start + 1
        ratio = a5_total / a6_total  # 기존 비율 그대로 사용

        if random_mode:
            a5_frames = sorted(random.sample(range(a5_start, a5_end + 1), num_samples))
        else:
            a5_frames = np.linspace(a5_start, a5_end, num_samples, dtype=int)

        pairs = []
        for idx, a5_frame in enumerate(a5_frames):
            rel_idx = a5_frame - a5_start
            # 비율로 나누어서 A6 프레임 계산
            a6_frame = a6_start + int(rel_idx / ratio)
            if a6_frame <= a6_end:
                pairs.append({'idx': idx, 'a5': a5_frame, 'a6': a6_frame})
        print(f"\n🎯 역방향 매핑된 프레임 쌍 ({len(pairs)}개):")
        for p in pairs:
            print(f"  샘플 {p['idx']:2d}: A5[{p['a5']}] <-> A6[{p['a6']}]")
        return pairs

    def find_image(self, folder, frame_num):
        """프레임 번호에 맞는 이미지 파일 찾기 (확장자 무관)"""
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            path = folder / f"{frame_num:010d}{ext}"
            if path.exists():
                return path
        return None

    def load_image(self, image_path, size=(400, 300)):
        """이미지 로드 및 리사이즈"""
        if image_path is None or not image_path.exists():
            return None
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, size)

    def show_comparison(self, dataset_path, frame_pairs):
        """프레임 쌍 이미지 비교 시각화"""
        a5_folder = Path(dataset_path) / "image_a5"
        a6_folder = Path(dataset_path) / "image_a6"
        rows = len(frame_pairs)
        fig, axes = plt.subplots(rows, 2, figsize=(12, 3*rows))
        fig.suptitle(f'Camera Sync Comparison - {Path(dataset_path).name}', fontsize=16)
        if rows == 1:
            axes = axes.reshape(1, -1)
        for i, pair in enumerate(frame_pairs):
            a5_img = self.load_image(self.find_image(a5_folder, pair['a5']))
            a6_img = self.load_image(self.find_image(a6_folder, pair['a6']))
            # A5
            if a5_img is not None:
                axes[i, 0].imshow(a5_img)
                axes[i, 0].set_title(f'A5 Frame {pair["a5"]}', fontsize=12)
            else:
                axes[i, 0].text(0.5, 0.5, 'Image Not Found', ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 0].set_title(f'A5 Frame {pair["a5"]} (Missing)', fontsize=12, color='red')
            axes[i, 0].axis('off')
            # A6
            if a6_img is not None:
                axes[i, 1].imshow(a6_img)
                axes[i, 1].set_title(f'A6 Frame {pair["a6"]}', fontsize=12)
            else:
                axes[i, 1].text(0.5, 0.5, 'Image Not Found', ha='center', va='center', transform=axes[i, 1].transAxes)
                axes[i, 1].set_title(f'A6 Frame {pair["a6"]} (Missing)', fontsize=12, color='red')
            axes[i, 1].axis('off')
            # 매핑 정보
            fig.text(0.5, 1 - (i + 0.8) / rows, f"A6[{pair['a6']}] <-> A5[{pair['a5']}]", ha='center', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.tight_layout()
        output_path = Path(dataset_path) / f"sync_comparison_{rows}samples.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 비교 이미지 저장: {output_path}")
        plt.show()

    def analyze(self, dataset_path, num_samples=10, random_mode=False, reverse=False):
        """전체 분석 실행"""
        print(f"\n{'='*60}\n🔍 데이터셋 분석: {Path(dataset_path).name}\n{'='*60}")
        offsets = self.read_frame_offset(dataset_path)
        if offsets is None:
            return
        if reverse:
            pairs = self.get_frame_pairs_reverse(offsets, num_samples, random_mode)
        else:
            pairs = self.get_frame_pairs(offsets, num_samples, random_mode)
        self.show_comparison(dataset_path, pairs)

def main():
    print("📷 Camera Sync Tool")
    print("=" * 50)
    analyzer = CameraSyncAnalyzer()
    base_path = r"Y:\adasip\Temp\20250711_LC_test\20250711_A6_A5_LC_test\ncdb_a6_dataset"
    dataset_path = input(f"Dataset path (Enter for default):\n{base_path}\n> ").strip().strip('"')
    if not dataset_path:
        dataset_path = base_path
    if not Path(dataset_path).exists():
        print(f"❌ Path does not exist: {dataset_path}")
        return
    try:
        num_samples = int(input("Number of samples (default 10): ") or "10")
        num_samples = max(1, min(num_samples, 20))
    except ValueError:
        num_samples = 10
    random_mode = input("Random sample? (y/n, default n): ").lower() == 'y'
    reverse = input("Reverse mapping? (y/n, default n): ").lower() == 'y'
    analyzer.analyze(dataset_path, num_samples, random_mode, reverse)

if __name__ == "__main__":
    main()