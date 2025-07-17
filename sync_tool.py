import sys
import os
import json
import shutil
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QMessageBox, QFileDialog,
                             QTableWidget, QTableWidgetItem, QHeaderView, QShortcut, QSizePolicy, QMenu)
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal

class MappingWindow(QWidget):
    """
    매핑 데이터를 테이블 형태로 보여주는 독립적인 창 클래스.
    """
    row_selected = pyqtSignal(int)
    delete_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("매핑 데이터")
        self.setGeometry(800, 100, 600, 700)
        
        layout = QVBoxLayout(self)
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(4)
        self.mapping_table.setHorizontalHeaderLabels(["ID", "파일명", "A5 원본", "A6 원본"])
        self.mapping_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mapping_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mapping_table.cellClicked.connect(self.on_cell_clicked)
        self.mapping_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mapping_table.customContextMenuRequested.connect(self.open_menu)
        self.mapping_table.setStyleSheet('''
            QTableWidget::item:selected {
                background-color: #FFFFE0; /* light yellow */
                color: black;
            }
        ''')
        layout.addWidget(self.mapping_table)

    def open_menu(self, position):
        """컨텍스트 메뉴를 엽니다."""
        row = self.mapping_table.indexAt(position).row()
        if row < 0:
            return
            
        menu = QMenu()
        delete_action = menu.addAction("선택한 쌍 삭제")
        action = menu.exec_(self.mapping_table.mapToGlobal(position))
        
        if action == delete_action:
            self.delete_requested.emit(row)

    def on_cell_clicked(self, row, column):
        """테이블 셀 클릭 시 row_selected 신호를 발생시킵니다."""
        self.row_selected.emit(row)

    def update_table(self, data):
        """테이블 내용을 새로운 데이터로 업데이트합니다."""
        self.mapping_table.setRowCount(len(data))
        for i, item in enumerate(data):
            self.mapping_table.setItem(i, 0, QTableWidgetItem(str(item['id'])))
            self.mapping_table.setItem(i, 1, QTableWidgetItem(item['new_filename']))
            self.mapping_table.setItem(i, 2, QTableWidgetItem(os.path.basename(item['a5_original_path'])))
            self.mapping_table.setItem(i, 3, QTableWidgetItem(os.path.basename(item['a6_original_path'])))
        self.mapping_table.scrollToBottom()

class ImageSyncTool(QMainWindow):
    """
    두 카메라(A5, A6)의 이미지 프레임을 시각적으로 동기화하고 저장하는 GUI 애플리케이션.
    """
    def __init__(self, parent_folder):
        super().__init__()
        self.parent_folder = os.path.abspath(parent_folder)
        self.valid = False

        self._setup_paths()
        if not self._load_frame_offsets(): return
        if not self._load_and_map_images(): return
        
        self._initialize_state()
        self._load_mapping_data()
        self._setup_gui()
        self._create_shortcuts()
        
        self.mapping_window.update_table(self.mapping_data)
        self._update_display()
        
        self.valid = True

    def showEvent(self, event):
        """메인 윈도우가 표시될 때 매핑 윈도우의 위치를 조정합니다."""
        super().showEvent(event)
        if self.valid and not self.mapping_window.isVisible():
            main_geo = self.geometry()
            self.mapping_window.setGeometry(main_geo.right() + 10, main_geo.top(), 600, main_geo.height())
            self.mapping_window.show()

    def closeEvent(self, event):
        """메인 윈도우가 닫힐 때 매핑 윈도우도 함께 닫습니다."""
        self.mapping_window.close()
        super().closeEvent(event)

    def _setup_paths(self):
        self.a5_dir = os.path.join(self.parent_folder, 'image_a5')
        self.a6_dir = os.path.join(self.parent_folder, 'image_a6')
        self.pcd_dir = os.path.join(self.parent_folder, 'pcd') # pcd 폴더 경로 추가
        self.synced_data_dir = os.path.join(self.parent_folder, 'synced_data')
        self.synced_a5_dir = os.path.join(self.synced_data_dir, 'image_a5')
        self.synced_a6_dir = os.path.join(self.synced_data_dir, 'image_a6')
        self.mapping_file = os.path.join(self.synced_data_dir, 'mapping_data.json')
        self.offset_file = os.path.join(self.parent_folder, 'frame_offset.txt')

    def _load_frame_offsets(self):
        if not os.path.exists(self.offset_file):
            QMessageBox.critical(None, "오류", f"필수 파일 'frame_offset.txt'를 찾을 수 없습니다:\n{self.offset_file}")
            return False
        try:
            with open(self.offset_file, 'r') as f: content = f.read()
            self.a5_start_frame = int(re.search(r"a5_start\s+(\d+)", content).group(1))
            self.a6_start_frame = int(re.search(r"a6_start\s+(\d+)", content).group(1))
            self.a5_end_frame = int(re.search(r"a5_end\s+(\d+)", content).group(1))
            self.a6_end_frame = int(re.search(r"a6_end\s+(\d+)", content).group(1))

            a5_total = self.a5_end_frame - self.a5_start_frame
            a6_total = self.a6_end_frame - self.a6_start_frame
            self.frame_ratio = a5_total / a6_total if a6_total != 0 else 1.0
            
            return True
        except (AttributeError, ValueError, FileNotFoundError) as e:
            QMessageBox.critical(None, "오류", f"'frame_offset.txt' 파일 분석 중 오류: {e}")
            return False

    def _parse_frame_number(self, filename):
        match = re.search(r'\d+', filename)
        return int(match.group(0)) if match else -1

    def _load_and_map_images(self):
        """pcd 폴더 기준으로 image_a5 목록을 필터링하고, 모든 이미지 목록을 로드 및 매핑합니다."""
        if not all(os.path.isdir(d) for d in [self.a5_dir, self.a6_dir, self.pcd_dir]):
            QMessageBox.critical(None, "오류", "필수 폴더('image_a5', 'image_a6', 'pcd') 중 하나를 찾을 수 없습니다.")
            return False

        # pcd 파일의 기본 이름 목록 생성
        pcd_basenames = {os.path.splitext(f)[0] for f in os.listdir(self.pcd_dir)}

        # pcd 목록 기준으로 image_a5 필터링
        all_a5_images = sorted(os.listdir(self.a5_dir), key=self._parse_frame_number)
        self.a5_images = [f for f in all_a5_images if os.path.splitext(f)[0] in pcd_basenames]
        
        self.a6_images = sorted([f for f in os.listdir(self.a6_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))], key=self._parse_frame_number)
        
        if not self.a5_images or not self.a6_images:
            QMessageBox.critical(None, "오류", "이미지 폴더가 비어있거나 pcd와 매칭되는 a5 이미지가 없습니다.")
            return False
            
        self.a5_frame_to_index = {self._parse_frame_number(f): i for i, f in enumerate(self.a5_images)}
        self.a6_frame_to_index = {self._parse_frame_number(f): i for i, f in enumerate(self.a6_images)}
        
        self.a5_start_index = self.a5_frame_to_index.get(self.a5_start_frame)
        self.a5_end_index = self.a5_frame_to_index.get(self.a5_end_frame)
        self.a6_start_index = self.a6_frame_to_index.get(self.a6_start_frame)
        self.a6_end_index = self.a6_frame_to_index.get(self.a6_end_frame)
        
        if any(i is None for i in [self.a5_start_index, self.a5_end_index, self.a6_start_index, self.a6_end_index]):
            QMessageBox.critical(None, "오류", "'frame_offset.txt'에 지정된 프레임 번호에 해당하는 이미지를 (필터링 된 목록에서) 찾을 수 없습니다.")
            return False
        return True

    def _initialize_state(self):
        self.a5_index = self.a5_start_index
        self.a6_index = self.a6_start_index
        self.mapping_data = []
        self.sync_counter = 0
        self.last_status_message = ""
        self.view_mode = False
        self.view_index = 0
        self.synced_original_paths = set()

    def _load_mapping_data(self):
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    self.mapping_data = json.load(f)
                if self.mapping_data:
                    self.sync_counter = max(item.get('id', -1) for item in self.mapping_data) + 1
                    for item in self.mapping_data:
                        self.synced_original_paths.add(item['a5_original_path'])
                        self.synced_original_paths.add(item['a6_original_path'])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                QMessageBox.warning(None, "JSON 오류", f"mapping_data.json 파일 분석 중 오류: {e}")
                self.mapping_data, self.sync_counter = [], 0

    def _setup_gui(self):
        self.setWindowTitle("이미지 동기화 툴")
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        image_layout = QHBoxLayout(central_widget)
        
        left_layout = QVBoxLayout()
        self.a5_filename_label = QLabel("A5 Filename")
        self.a5_filename_label.setAlignment(Qt.AlignCenter)
        self.a5_image_label = QLabel()
        self.a5_image_label.setAlignment(Qt.AlignCenter)
        self.a5_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        left_layout.addWidget(self.a5_filename_label)
        left_layout.addWidget(self.a5_image_label, 1)
        image_layout.addLayout(left_layout, 1)

        right_layout = QVBoxLayout()
        self.a6_filename_label = QLabel("A6 Filename")
        self.a6_filename_label.setAlignment(Qt.AlignCenter)
        self.a6_image_label = QLabel()
        self.a6_image_label.setAlignment(Qt.AlignCenter)
        self.a6_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        right_layout.addWidget(self.a6_filename_label)
        right_layout.addWidget(self.a6_image_label, 1)
        image_layout.addLayout(right_layout, 1)
        
        self.mapping_window = MappingWindow()
        self.mapping_window.row_selected.connect(self.on_mapping_table_row_selected)
        self.mapping_window.delete_requested.connect(self._request_delete_pair)
        self.statusBar().showMessage("준비 완료")

    def on_mapping_table_row_selected(self, row_index):
        """매핑 테이블에서 행이 선택되었을 때 호출되는 슬롯."""
        if not self.mapping_data or not (0 <= row_index < len(self.mapping_data)):
            return

        # 보기 모드가 아니면 보기 모드로 전환
        if not self.view_mode:
            self.view_mode = True
            self._update_shortcut_states()

        self.view_index = row_index
        item_id = self.mapping_data[row_index].get('id', 'N/A')
        self._update_display(f"ID {item_id}번 쌍으로 이동")

    def _create_shortcuts(self):
        self.shortcuts = {
            'toggle_view': QShortcut(QKeySequence("V"), self, self._toggle_view_mode),
            'go_home': QShortcut(QKeySequence("H"), self, self._go_to_home_state),
            'sync_a5_prev': QShortcut(QKeySequence("Q"), self, self._prev_a5),
            'sync_a5_next': QShortcut(QKeySequence("E"), self, self._next_a5),
            'sync_a6_prev': QShortcut(QKeySequence("A"), self, self._prev_a6),
            'sync_a6_next': QShortcut(QKeySequence("D"), self, self._next_a6),
            'sync_save': QShortcut(QKeySequence(Qt.Key_Return), self, self._save_pair),
            'sync_save_enter': QShortcut(QKeySequence(Qt.Key_Enter), self, self._save_pair),
            'sync_delete': QShortcut(QKeySequence(Qt.Key_Backspace), self, self._delete_last_pair),
            'view_prev': QShortcut(QKeySequence("A"), self, self._prev_view),
            'view_next': QShortcut(QKeySequence("D"), self, self._next_view),
        }
        self._update_shortcut_states()

    def _update_shortcut_states(self):
        is_sync_mode = not self.view_mode
        self.shortcuts['sync_a5_prev'].setEnabled(is_sync_mode)
        self.shortcuts['sync_a5_next'].setEnabled(is_sync_mode)
        self.shortcuts['sync_a6_prev'].setEnabled(is_sync_mode)
        self.shortcuts['sync_a6_next'].setEnabled(is_sync_mode)
        self.shortcuts['sync_save'].setEnabled(is_sync_mode)
        self.shortcuts['sync_save_enter'].setEnabled(is_sync_mode)
        self.shortcuts['sync_delete'].setEnabled(is_sync_mode)
        self.shortcuts['view_prev'].setEnabled(self.view_mode)
        self.shortcuts['view_next'].setEnabled(self.view_mode)

    def _update_display(self, status_message=None):
        if self.view_mode:
            if not self.mapping_data:
                self.statusBar().showMessage("표시할 저장된 데이터가 없습니다. (V 키로 동기화 모드로 전환)")
                self.mapping_window.mapping_table.clearSelection()
                return
            
            # view_index가 유효한지 확인 및 조정
            if self.view_index >= len(self.mapping_data):
                self.view_index = len(self.mapping_data) - 1
            if self.view_index < 0:
                self.view_index = 0

            entry = self.mapping_data[self.view_index]
            filename = entry['new_filename']

            # 매핑 테이블의 해당 행을 선택하여 하이라이트
            self.mapping_window.mapping_table.selectRow(self.view_index)

            self._update_panel(self.synced_a5_dir, [item['new_filename'] for item in self.mapping_data], self.view_index, self.a5_image_label, self.a5_filename_label)
            self._update_panel(self.synced_a6_dir, [item['new_filename'] for item in self.mapping_data], self.view_index, self.a6_image_label, self.a6_filename_label)
            status_text = f"보기 모드: {self.view_index + 1}/{len(self.mapping_data)} ({filename})"
        else:
            self._update_panel(self.a5_dir, self.a5_images, self.a5_index, self.a5_image_label, self.a5_filename_label)
            self._update_panel(self.a6_dir, self.a6_images, self.a6_index, self.a6_image_label, self.a6_filename_label)
            a5_frame = self._parse_frame_number(self.a5_images[self.a5_index])
            a6_frame = self._parse_frame_number(self.a6_images[self.a6_index])
            status_text = f"저장된 쌍: {len(self.mapping_data)} | A5: {self.a5_index + 1}/{len(self.a5_images)} ({a5_frame}) | A6: {self.a6_index + 1}/{len(self.a6_images)} ({a6_frame})"

        if status_message: self.last_status_message = status_message
        if self.last_status_message: status_text += f" | {self.last_status_message}"
        self.statusBar().showMessage(status_text)

    def _update_panel(self, img_dir, img_list, index, image_label, filename_label):
        filename = img_list[index]
        filename_label.setText(filename)
        image_path = os.path.join(img_dir, filename)
        
        image_label.setStyleSheet("border: 3px solid transparent; padding: 3px;")
        if not self.view_mode and os.path.abspath(image_path) in self.synced_original_paths:
            image_label.setStyleSheet("border: 3px solid green; padding: 3px;")

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            image_label.setText(f"이미지 로드 실패:\n{filename}")
            image_label.setPixmap(QPixmap())
        else:
            scaled_pixmap = pixmap.scaled(image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)

    def _update_a5_based_on_a6(self):
        """현재 A6 프레임에 따라 A5 프레임을 자동으로 계산하고 업데이트합니다."""
        current_a6_frame = self._parse_frame_number(self.a6_images[self.a6_index])
        
        rel_idx = current_a6_frame - self.a6_start_frame
        target_a5_frame = self.a5_start_frame + int(rel_idx * self.frame_ratio)

        # 계산된 프레임 번호와 가장 가까운 실제 A5 이미지를 찾습니다.
        closest_a5_image = min(self.a5_images, key=lambda img: abs(self._parse_frame_number(img) - target_a5_frame))
        
        try:
            # 찾은 이미지의 인덱스로 a5_index를 업데이트합니다.
            self.a5_index = self.a5_images.index(closest_a5_image)
        except ValueError:
            # 만약 이미지를 찾지 못하면(이론상 발생하지 않아야 함), 시작 인덱스로 되돌립니다.
            self.a5_index = self.a5_start_index

    def _navigate_index(self, current_index_attr, start_index_attr, end_index_attr, step, message, update_a5=False):
        current_index = getattr(self, current_index_attr)
        start_index = getattr(self, start_index_attr)
        end_index = getattr(self, end_index_attr)

        if step == -1: # Previous
            new_index = current_index - 1 if current_index > start_index else end_index
        else: # Next
            new_index = current_index + 1 if current_index < end_index else start_index
        
        setattr(self, current_index_attr, new_index)
        
        if update_a5:
            self._update_a5_based_on_a6()
        self._update_display(message)

    def _navigate_circular_index(self, current_index_attr, data_list_attr, step, message):
        current_index = getattr(self, current_index_attr)
        data_list = getattr(self, data_list_attr)
        
        if not data_list: # Handle empty list case for view mode
            self._update_display("표시할 저장된 데이터가 없습니다.")
            return

        new_index = (current_index + step + len(data_list)) % len(data_list)
        setattr(self, current_index_attr, new_index)
        self._update_display(message)

    def _prev_a5(self): self._navigate_index('a5_index', 'a5_start_index', 'a5_end_index', -1, "A5 이전")
    def _next_a5(self): self._navigate_index('a5_index', 'a5_start_index', 'a5_end_index', 1, "A5 다음")

    def _prev_a6(self): self._navigate_index('a6_index', 'a6_start_index', 'a6_end_index', -1, "A6 이전", update_a5=True)

    def _next_a6(self): self._navigate_index('a6_index', 'a6_start_index', 'a6_end_index', 1, "A6 다음", update_a5=True)

    def _prev_view(self): self._navigate_circular_index('view_index', 'mapping_data', -1, "이전 쌍 보기")
    def _next_view(self): self._navigate_circular_index('view_index', 'mapping_data', 1, "다음 쌍 보기")

    def _go_to_home_state(self):
        if self.view_mode:
            if self.mapping_data:
                self.view_index = len(self.mapping_data) - 1
                self._update_display("마지막 저장 쌍으로 이동")
        else:
            if self.mapping_data:
                last_entry = self.mapping_data[-1]
                a5_filename = os.path.basename(last_entry['a5_original_path'])
                a6_filename = os.path.basename(last_entry['a6_original_path'])
                try:
                    self.a5_index = self.a5_images.index(a5_filename)
                    self.a6_index = self.a6_images.index(a6_filename)
                    self._update_display("마지막 저장 위치로 이동")
                except ValueError:
                    self.a5_index = self.a5_start_index
                    self.a6_index = self.a6_start_index
                    self._update_display("마지막 저장 파일을 찾을 수 없어 초기 상태로 복귀")
            else:
                self.a5_index = self.a5_start_index
                self.a6_index = self.a6_start_index
                self._update_display("초기 상태로 복귀")

    def _toggle_view_mode(self):
        if not self.mapping_data and not self.view_mode:
            QMessageBox.information(self, "정보", "보기 모드로 전환하려면 먼저 하나 이상의 쌍을 저장해야 합니다.")
            return
        self.view_mode = not self.view_mode
        self.view_index = len(self.mapping_data) - 1 if self.mapping_data else 0
        self._update_shortcut_states()
        self._update_display("보기 모드 전환" if self.view_mode else "동기화 모드 전환")

    def _save_pair(self):
        src_a5_path = os.path.join(self.a5_dir, self.a5_images[self.a5_index])
        src_a6_path = os.path.join(self.a6_dir, self.a6_images[self.a6_index])
        
        if os.path.abspath(src_a5_path) in self.synced_original_paths or \
           os.path.abspath(src_a6_path) in self.synced_original_paths:
            QMessageBox.warning(self, "중복 저장", "선택된 이미지 중 하나 이상이 이미 다른 쌍에 포함되어 있습니다.")
            return

        os.makedirs(self.synced_a5_dir, exist_ok=True)
        os.makedirs(self.synced_a6_dir, exist_ok=True)

        new_filename = f"{self.sync_counter:010d}.png"
        dest_a5_path = os.path.join(self.synced_a5_dir, new_filename)
        dest_a6_path = os.path.join(self.synced_a6_dir, new_filename)

        try:
            shutil.copy2(src_a5_path, dest_a5_path)
            shutil.copy2(src_a6_path, dest_a6_path)
        except Exception as e:
            QMessageBox.critical(self, "파일 복사 오류", f"이미지 저장 중 오류 발생: {e}")
            return

        abs_src_a5 = os.path.abspath(src_a5_path)
        abs_src_a6 = os.path.abspath(src_a6_path)
        mapping_entry = {"id": self.sync_counter, "new_filename": new_filename, "a5_original_path": abs_src_a5, "a6_original_path": abs_src_a6}
        
        self.mapping_data.append(mapping_entry)
        self.synced_original_paths.add(abs_src_a5)
        self.synced_original_paths.add(abs_src_a6)
        
        # ID 정렬 후 저장
        self.mapping_data.sort(key=lambda x: x['id'])
        self._save_mapping_data()
        self.mapping_window.update_table(self.mapping_data)

        self.sync_counter += 1
        self._update_display(f"성공: ID {self.sync_counter-1}번 쌍 저장!")

    def _delete_last_pair(self):
        """가장 마지막에 저장된 쌍을 삭제합니다."""
        if not self.mapping_data:
            QMessageBox.information(self, "정보", "삭제할 저장된 쌍이 없습니다.")
            return
        
        # ID를 기준으로 마지막 항목을 찾습니다.
        last_entry = max(self.mapping_data, key=lambda x: x['id'])
        index_to_delete = self.mapping_data.index(last_entry)
        
        reply = QMessageBox.question(self, "삭제 확인", "가장 마지막 쌍을 삭제하시겠습니까?", 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._perform_delete(index_to_delete, last_entry)

    def _request_delete_pair(self, index):
        """매핑 테이블에서 요청된 특정 쌍의 삭제를 처리합니다."""
        if not (0 <= index < len(self.mapping_data)):
            return
            
        entry_to_delete = self.mapping_data[index]
        entry_id = entry_to_delete.get('id', 'N/A')

        reply = QMessageBox.question(self, "삭제 확인", f"ID {entry_id}번 쌍을 삭제하시겠습니까?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._perform_delete(index, entry_to_delete)

    def _perform_delete(self, index, entry_to_delete):
        """실제 파일 및 데이터 삭제를 수행하는 공통 메소드."""
        # 데이터 목록에서 항목 제거
        del self.mapping_data[index]

        # 원본 경로 집합에서 제거
        if entry_to_delete['a5_original_path'] in self.synced_original_paths:
            self.synced_original_paths.remove(entry_to_delete['a5_original_path'])
        if entry_to_delete['a6_original_path'] in self.synced_original_paths:
            self.synced_original_paths.remove(entry_to_delete['a6_original_path'])
        
        # 실제 파일 삭제
        for dir_path in [self.synced_a5_dir, self.synced_a6_dir]:
            file_to_delete = os.path.join(dir_path, entry_to_delete['new_filename'])
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)

        # 다음 저장 ID를 가장 높은 ID + 1로 재설정 (더 안전한 방식)
        if self.mapping_data:
            self.sync_counter = max(item.get('id', -1) for item in self.mapping_data) + 1
        else:
            self.sync_counter = 0

        self._save_mapping_data()
        self.mapping_window.update_table(self.mapping_data)
        
        status_message = f"성공: ID {entry_to_delete['id']}번 쌍 삭제 완료!"
        
        if self.view_mode and not self.mapping_data:
            self.view_mode = False
            self._update_shortcut_states()
            status_message += " 동기화 모드로 전환합니다."
        
        self._update_display(status_message)

    def _save_mapping_data(self):
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.mapping_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            QMessageBox.critical(self, "JSON 저장 오류", f"mapping_data.json 파일 저장 중 오류: {e}")

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    if len(sys.argv) > 1:
        parent_folder = sys.argv[1]
    else:
        default_path = r"Y:\adasip\Temp\20250711_LC_test\20250711_A6_A5_LC_test\ncdb_a6_dataset\2025_07_11"
        start_path = default_path if os.path.isdir(default_path) else "."
        parent_folder = QFileDialog.getExistingDirectory(None, "동기화할 이미지들의 상위 폴더를 선택하세요", start_path)
        if not parent_folder:
            sys.exit("폴더가 선택되지 않았습니다.")

    main_win = ImageSyncTool(parent_folder)
    if main_win.valid:
        main_win.show()
        sys.exit(app.exec_())
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()