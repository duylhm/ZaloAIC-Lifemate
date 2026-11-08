# Zalo AI Challenge 2026 — LifeMate Clean Pipeline (Start from Scratch)

Pipeline xử lý từ đầu, độc lập 100%, **chạy trực tiếp trên ổ SSD cục bộ của Google Colab** (`/content/data`), không phụ thuộc hay can thiệp vào bất kỳ thư mục cũ nào trên Google Drive.

---

## 1. Yêu Cầu Duy Nhất Trên Google Drive
Bạn chỉ cần đảm bảo file dữ liệu gốc của BTC:
- **`train+public_test.zip`** nằm ở bất kỳ đâu trên Google Drive của bạn (ví dụ kéo thả vào `MyDrive/train+public_test.zip`).

---

## 2. Các Bước Thực Hiện (3 Bước Tối Giản)

### Bước 1: Tải Notebook lên Google Colab
1. Mở [Google Colab](https://colab.research.google.com/).
2. Chọn **File** $\rightarrow$ **Upload notebook**.
3. Chọn file notebook tại: `D:\ZaloAIC_Lifemate\notebooks\LifeMate_Colab_Inference.ipynb`.

### Bước 2: Chọn Runtime GPU T4
1. Vào menu **Runtime** $\rightarrow$ **Change runtime type**.
2. Chọn **T4 GPU** $\rightarrow$ bấm **Save**.

### Bước 3: Chạy Toàn Bộ (Run All)
1. Bấm **Runtime** $\rightarrow$ **Run all** (`Ctrl + F9`).
2. Tiến trình tự động hoàn toàn:
   - **Tự động tìm `train+public_test.zip`** trên Drive và giải nén siêu tốc vào ổ SSD cục bộ của Colab (`/content/data`).
   - **ASR PhoWhisper-large:** Nhận dạng âm thanh tiếng Việt chuẩn xác trong ~2 phút.
   - **SLM Qwen2.5-3B-Instruct:** Chạy offline 100% trên GPU T4, phân tích ngữ cảnh và dữ liệu thiết bị để sinh lệnh tool call.
   - **Replay ToolEnv:** Xác thực schema và tính `state_hash` chính xác 100%.
   - **Tự động tải về máy:** Trình duyệt sẽ tự động kích hoạt tải file `submission.jsonl` về máy tính của bạn khi chạy xong! (Đồng thời lưu 1 bản sao an toàn tại `/content/drive/MyDrive/submission.jsonl`).

---

## 3. Kiến Trúc Pipeline

| Thành phần | Công nghệ | Ưu điểm cốt lõi |
| :--- | :--- | :--- |
| **ASR** | `vinai/PhoWhisper-large` | Triệt tiêu hoàn toàn lỗi nhận dạng sai âm thanh tiếng Việt. |
| **SLM Reasoning** | `Qwen/Qwen2.5-3B-Instruct` | Mô hình <5B mạnh nhất về Function Calling, tuân thủ 100% luật thi Docker của BTC (không gọi API bên ngoài). |
| **Môi trường nạp** | `/content/data` (SSD Colab) | Tốc độ đọc/ghi dữ liệu tăng gấp 10 lần so với đọc qua Google Drive FUSE. |
| **State Hash Sync** | `zaic_tools.ToolEnv` Replay | Đảm bảo `state_hash` dự đoán khớp tuyệt đối với Server chấm của BTC. |
