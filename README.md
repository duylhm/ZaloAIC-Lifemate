# LifeMate v2 — zero-shot baseline

Pipeline hiện tại: **Gipformer 1.5 68M → Qwen3-4B → ToolEnv → JSONL**.
Sau log Kaggle cho thấy Qwen3.5 dùng kernel PyTorch dự phòng và 10/11 mẫu đầu
timeout, mặc định chuyển sang Qwen3-4B để đo lại. Đây là phương án thử tiếp,
chưa có số liệu GPU chứng minh đã đạt tốc độ/điểm yêu cầu. Bản v2.1 chọn FP16
trên GPU trước Ampere, chặn pilot khi smoke lỗi và dừng sớm TRAIN khi 3/5 mẫu
đầu timeout. Không tăng giới hạn 30 giây để che lỗi hiệu năng.
Bản cũ được giữ trong `rebuild-backup-20261115/`; chưa commit/push vì `.git`
không cho ghi trong phiên làm việc. Adapter cũ và dữ liệu không bị xóa.

## Chạy trên Kaggle

1. Giải nén `LifeMate_v2_Kit.zip`. Import `LifeMate_v2_Kaggle.ipynb`.
2. Upload **LifeMate_v2_Source.zip bản mới** thành private dataset, gắn dataset đó
   và dataset BTC qua Add Input. Dataset source cũ không dùng với notebook v2.
3. Bật GPU và Internet. Chạy 8 cell code theo bảng dưới; chỉ 5 cell đầu tự chạy
   công việc. Để SOURCE/DATA_ROOT trống nếu chỉ có một bộ source/data; hoặc điền
   path file ZIP hay thư mục đã giải nén. Không sửa cell setup.

| Cell | Công việc | Output cần xem |
|---|---|---|
| 1 config | Chọn source/data/model, mặc định Qwen3 + Gipformer | Biến cấu hình |
| 2 setup | Nhận ZIP hoặc thư mục, cài thư viện | `Setup complete` |
| 3 preflight | Kiểm tra GPU, đủ audio, pin revision, tạo split | Tên GPU, SHA, số mẫu mỗi tập |
| 4 smoke | 5 mẫu pilot chạy audio → JSONL | Thời gian, timeout, trace |
| 5 pilot | Cùng mẫu: gold vs ASR | score, state_match, tool_f1, chênh lệch |
| 6 confirm | Bật RUN_CONFIRM sau khi xem pilot | Kết quả tập độc lập |
| 7 public | Bật RUN_PUBLIC khi đã chốt | `submission.jsonl` |
| 8 export | Bật EXPORT_OFFLINE để tải weights cho Docker | Thư mục deploy |

Kết quả: `/kaggle/working/lifemate-v2-results/`. Lưu cả thư mục Output sau phiên.
Nếu bị ngắt, chạy lại cell đó. Phiên Kaggle mới: gắn output cũ, điền RESUME_FROM,
giữ nguyên source/cấu hình/model revision. Nếu thư viện hoặc cấu hình thay đổi,
runner từ chối trộn kết quả; đổi RUN_NAME để bắt đầu phép thử mới.

Trước khi chạy tiếp, xem `*.summary.json` và `*.traces.jsonl`. Nếu nhiều timeout
hoặc context overflow, xử lý tốc độ/context trước. Nếu thử Qwen3-4B làm đối chứng,
đổi MODEL_ID, để MODEL_REVISION trống, đổi RUN_NAME rồi chạy lại từ cell 3.

## Lệnh chính

```bash
python scripts/prepare_splits.py --root /data --output outputs/splits
python run_inference.py --root /data --split train --ids outputs/splits/pilot.json --asr gold --output outputs/pilot_gold.jsonl --evaluate
python run_inference.py --root /data --split train --ids outputs/splits/pilot.json --asr gipformer --output outputs/pilot_asr.jsonl --evaluate
python scripts/compare.py outputs/pilot_gold.jsonl outputs/pilot_asr.jsonl
python run_inference.py --root /data --split public_test --asr gipformer --output outputs/submission.jsonl
python evaluate.py --root /data --split public_test --prediction outputs/submission.jsonl
```

Để tái lập, truyền `--revision` và `--asr-revision` theo file pins của notebook.
Inference chỉ nhập ToolEnv của đúng split. Label chỉ được đọc trong evaluator,
split và oracle test. Không đọc trực tiếp state.json từ code Agent.

## Kiểm tra và giới hạn

```bash
python -m unittest discover -s tests -v
python scripts/verify_train_replay.py --root /data --output outputs/oracle-check.json
```

Oracle cấp sẵn gold calls để kiểm tra Agent/runtime; **không phải điểm model**.
Xem [docs/VALIDATION.md](docs/VALIDATION.md) cho kết quả local thực tế và phần
chưa chạy. GPU inference, so sánh ASR và Docker offline chưa được xác nhận trên
máy này. Chưa có submission mới hay cải thiện leaderboard được đo.

QLoRA vẫn đóng theo kế hoạch đã duyệt: chỉ mở khi có lỗi Agent cụ thể và đủ
ngân sách xác nhận. Không k-fold, ensemble hoặc model soup ở baseline này.
Xem [kế hoạch thử nghiệm](docs/EXPERIMENT_PLAN.md) và [Docker](docker/README.md).
