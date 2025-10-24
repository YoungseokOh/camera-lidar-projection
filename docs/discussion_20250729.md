### `sync_tool.py` 데이터 관리 및 ID 재사용 논의 (2025년 7월 29일)

**1. 현재 `sync_tool.py`의 데이터 관리 방식 분석:**

*   **`mapping_data`:** `ImageSyncTool` 클래스 내 `self.mapping_data`는 동기화된 이미지-PCD 쌍의 정보를 담는 Python 리스트입니다. 이 리스트는 `mapping_data.json` 파일로 저장됩니다.
*   **`id` 부여 방식:** 새로운 쌍을 저장할 때 사용되는 `id`는 `sync_counter` 변수에 의해 결정됩니다. `sync_counter`는 `self.mapping_data`에 현재 존재하는 항목들의 `id` 중 최댓값에 1을 더한 값으로 설정됩니다 (`max(item.get('id', -1) for item in self.mapping_data) + 1`).
*   **저장 위치:** 새로운 매핑 항목은 `self.mapping_data.append(mapping_entry)`를 통해 리스트의 **맨 끝에 추가**됩니다. 이후 `self.mapping_data.sort(key=lambda x: x['id'])`를 통해 `id`를 기준으로 정렬되지만, 새로 부여된 `id`가 항상 기존의 모든 `id`보다 크기 때문에, 사실상 항상 리스트의 맨 끝에 위치하게 됩니다.
*   **파일 이름 (`new_filename`):** 동기화된 파일들(예: `0000000001.png`, `0000000001.pcd`, `0000000001.jpg`)의 이름은 이 `id`를 기반으로 생성됩니다 (`f"{self.sync_counter:010d}"`).
*   **삭제 동작:** `_request_delete_pairs` 및 `_perform_delete_actions` 메서드를 통해 매핑 테이블에서 항목을 삭제하면, 해당 항목의 정보가 `mapping_data` 리스트에서 제거될 뿐만 아니라, `synced_data` 폴더 내의 관련 물리적 파일들(동기화된 A6 이미지, PCD 파일, 투영 결과 이미지)도 함께 삭제됩니다. 삭제된 `id`는 현재 재사용되지 않습니다.

**2. 사용자 요구사항:**

*   사용자는 매핑 테이블에서 중간에 특정 `id` (예: 8번)를 가진 항목을 삭제한 후, 새로운 항목을 저장할 때 삭제된 그 `id` (8번)를 재사용하여 해당 "빈 자리"에 새로운 항목이 삽입되기를 원합니다.
*   이는 `new_filename`도 해당 재사용된 `id`를 반영하여 생성되기를 의미합니다.
*   순서가 엄격하게 맞아야 하는 것은 아니지만, 삭제된 자리가 채워져서 전체 목록이 깔끔하게 정리되기를 선호합니다.

**3. 현재 방식의 한계:**

*   현재 `id` 부여 방식(`max(id) + 1`)과 `append()` 방식으로는 삭제된 `id`를 재사용하거나 리스트의 중간에 항목을 물리적으로 삽입하는 것이 불가능합니다. 새로운 항목은 항상 가장 큰 `id`를 부여받고 리스트의 맨 끝에 추가됩니다.

**4. 제안된 해결 방안: "사용 가능한 ID 풀(Pool) 관리 및 `new_filename`과 ID의 직접 연동"**

이 방안은 삭제된 `id`를 추적하고, 새로운 항목을 저장할 때 이 `id`를 재사용하여 파일명도 해당 `id`로 생성하는 방식입니다.

*   **`available_ids` 풀 도입:**
    *   `ImageSyncTool` 클래스에 `self.available_ids = []`와 같은 자료구조를 추가합니다. Python의 `heapq` 모듈을 사용하여 `min-heap`으로 구현하면, 항상 가장 작은 사용 가능한 `id`를 효율적으로 가져올 수 있습니다.
*   **삭제 시 `id` 풀에 추가:**
    *   `_perform_delete_actions` 메서드에서 항목을 삭제할 때, 해당 항목의 `id`를 `self.available_ids` 힙에 추가합니다.
*   **저장 시 `id` 할당 로직 변경:**
    *   `_save_pair` 메서드에서 새로운 `id`를 할당할 때, 먼저 `self.available_ids` 힙이 비어 있는지 확인합니다.
    *   **힙이 비어 있지 않으면:** `heapq.heappop(self.available_ids)`를 통해 힙에서 가장 작은 `id`를 가져와 새로운 항목에 할당합니다.
    *   **힙이 비어 있으면:** 기존 `sync_counter` 로직(`max(id) + 1`)을 사용하여 새로운 최대 `id`를 생성합니다.
*   **`new_filename` 연동:**
    *   새로운 항목의 `new_filename`은 이 할당된 `id`를 기반으로 생성됩니다 (`f"{assigned_id:010d}"`).
*   **`mapping_data` 정렬 유지:**
    *   `self.mapping_data` 리스트에 새로운 항목을 추가한 후, `self.mapping_data.sort(key=lambda x: x['id'])`를 통해 `id`를 기준으로 다시 정렬합니다.
*   **JSON 저장:**
    *   정렬된 `self.mapping_data` 리스트가 `mapping_data.json` 파일에 저장됩니다.
*   **초기 로드 시 `available_ids` 초기화:**
    *   `_load_mapping_data` 시점에 `mapping_data`를 로드한 후, 현재 사용 중인 `id`들을 파악하고, 전체 가능한 `id` 범위에서 사용 중이지 않은 `id`들을 `available_ids`에 채워 넣는 로직이 필요합니다. (예: `mapping_data`의 `id`들을 `set`에 넣고, `0`부터 `max_id`까지 반복하며 `set`에 없는 `id`를 `available_ids`에 추가).

**5. 방안의 효과 및 사용자 요구사항 충족 여부:**

*   이 방안을 사용하면 삭제된 `id`가 재사용되어 새로운 항목에 할당됩니다.
*   `new_filename`도 재사용된 `id`를 반영하므로, 파일 시스템에서도 해당 `id`의 파일이 새로 생성됩니다. (기존 파일은 이미 삭제되었으므로 덮어쓰기 문제는 발생하지 않습니다.)
*   `mapping_data` 리스트는 항상 `id`를 기준으로 정렬되므로, `mapping_data.json` 파일과 매핑 테이블은 `id` 순서대로 깔끔하게 정리된 상태를 유지합니다.
*   결과적으로, JSON 파일의 물리적인 중간에 직접 삽입하는 것은 아니지만, `id` 순서상 삭제된 "빈 자리"에 새로운 항목이 논리적으로 채워지는 효과를 얻을 수 있습니다.

**6. 추가 논의점 (해결됨):**

*   **파일 삭제 여부:** 사용자가 "지금 파일을 따로 삭제해주고 있지는 않나? json에서만 삭제가 되는건가?"라고 질문하셨고, 코드 분석 결과 `_perform_delete_actions` 메서드에서 JSON 데이터뿐만 아니라 관련 물리적 파일들도 함께 삭제하고 있음을 확인했습니다. 따라서 `id` 재사용 시 파일 덮어쓰기 문제는 발생하지 않습니다.
