"""Generate the small, ordered Kaggle workflow from the tested bootstrap implementation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []

def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

def code(name, text):
    cells.append({"cell_type": "code", "id": name, "metadata": {"tags": [name]},
                  "execution_count": None, "outputs": [], "source": text.strip().splitlines(True)})

markdown('''# LifeMate v2 — baseline trước, đo điểm sau
Notebook mới, độc lập với bộ source cũ. Gắn **LifeMate_v2_Source.zip** và dữ liệu BTC bằng Add Input.
Kaggle có thể giải nén ZIP: cell setup nhận cả file và thư mục.

**Chạy cell 1 → 2 → 3 → 4 → 5.** Cell 4 chỉ kiểm tra 5 mẫu từ audio đến JSONL; cell 5 so gold với ASR trên pilot.
Cell 6 xác nhận, cell 7 chạy public, cell 8 đóng gói model offline — mặc định tắt.
Nếu chưa có GPU, dừng ở setup. Chưa có kết quả để quyết định train; notebook này không tự bật huấn luyện.
''')
markdown('## 1. Cấu hình')
code('config', '''from pathlib import Path
SOURCE = ""  # File LifeMate_v2_Source.zip HOẶC thư mục dataset chứa lifemate/runner.py bản v2.
DATA_ROOT = ""  # Thư mục chứa train/ và public_test/, hoặc train+public_test.zip. Trống = tự tìm.
OUTPUT_DIR = "/kaggle/working/lifemate-v2-results"
RESUME_FROM = ""  # Thư mục kết quả đã lưu từ phiên trước, gắn lại qua Add Input.
MODEL_ID = "Qwen/Qwen3-4B"  # Runtime fallback after Qwen3.5 PyTorch-kernel timeouts.
MODEL_REVISION = ""  # Cell 3 chọn SHA trước cutoff của BTC và in ra.
ASR_MODEL = "g-group-ai-lab/gipformer1.5-68M-rnnt"
ASR_REVISION = ""
ASR_BACKEND = "gipformer"  # Phương án đối chứng: "phowhisper" + ASR_MODEL="vinai/PhoWhisper-large".
PILOT_SIZE = 30
SECONDS_PER_SAMPLE = 30
RUN_NAME = "qwen3_gip_runtimefix"  # Đổi tên nếu đổi model, ASR hoặc cấu hình.
''')
markdown('## 2. Tìm source/dữ liệu và cài thư viện\nKhông cần sửa dòng bên trong cell này. Source cũ sẽ bị từ chối với thông báo rõ ràng.')
bootstrap = (ROOT / 'lifemate/bootstrap.py').read_text(encoding='utf-8')
code('setup', bootstrap + '''
import os, subprocess, sys, json
work = Path("/kaggle/working")
inputs = Path("/kaggle/input")
work.mkdir(parents=True, exist_ok=True)
if not SOURCE:
    found = source_roots(inputs)
    if not found:
        found = sorted(inputs.rglob("LifeMate_v2_Source.zip"))
    if len(found) != 1:
        raise ValueError(f"Gắn source v2 hoặc điền SOURCE bằng đúng path file/thư mục. Tìm thấy: {found}")
    SOURCE = str(found[0])
repo = stage_source(SOURCE, work)
if not DATA_ROOT:
    roots = sorted({p.parent.parent for p in inputs.rglob("train.jsonl") if p.parent.name == "train"})
    if not roots:
        roots = sorted(inputs.rglob("train+public_test.zip"))
    if len(roots) != 1:
        raise ValueError(f"Điền DATA_ROOT bằng path dataset BTC. Tìm thấy: {roots}")
    DATA_ROOT = str(roots[0])
DATA_ROOT = str(find_data(DATA_ROOT, work))
out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)
if RESUME_FROM:
    saved = Path(RESUME_FROM)
    if not saved.is_dir():
        raise FileNotFoundError(saved)
    for item in saved.rglob("*"):
        if item.is_file():
            target = out / item.relative_to(saved)
            if target.exists() and target.read_bytes() != item.read_bytes():
                raise ValueError(f"Kết quả đang có khác bản resume: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(item, target)
os.environ["HF_HOME"] = str(work / "hf-cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
print("Source:", repo, "\\nData:", DATA_ROOT, "\\nOutput:", out)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)
def run_script(script, *args):
    subprocess.run([sys.executable, str(repo / script), *map(str, args)], cwd=repo, check=True)
print("Setup complete — chạy cell 3.")
''')
markdown('## 3. Kiểm tra GPU/dữ liệu, chốt revision và tạo split\nGiữ nguyên 30 giây/mẫu để biết sớm cấu hình có quá chậm hay không. Model chỉ được tải ở cell 4.')
code('preflight', '''subprocess.run([sys.executable, "-c", "import torch; from transformers import Qwen3_5ForCausalLM; import sherpa_onnx; assert torch.cuda.is_available(), 'Enable Kaggle GPU'; print(torch.cuda.get_device_name(0), torch.__version__, 'capability=', torch.cuda.get_device_capability(0))"], check=True)
for split in ("train", "public_test"):
    rows = [json.loads(s) for s in (Path(DATA_ROOT) / split / f"{split}.jsonl").read_text().splitlines() if s.strip()]
    for row in rows:
        for turn in [row, *row.get("history", [])]:
            if turn.get("audio") and not (Path(DATA_ROOT) / split / turn["audio"]).is_file():
                raise FileNotFoundError(f"Thiếu audio: {split}/{turn['audio']}")
    print(split, len(rows), "samples; all referenced audio found")
pin_args = ["--model", MODEL_ID, "--asr-model", ASR_MODEL, "--output", out / f"{RUN_NAME}.pins.json"]
if MODEL_REVISION: pin_args += ["--revision", MODEL_REVISION]
if ASR_REVISION: pin_args += ["--asr-revision", ASR_REVISION]
run_script("scripts/pin_models.py", *pin_args)
pins = json.loads((out / f"{RUN_NAME}.pins.json").read_text())
MODEL_REVISION = pins["llm"]["revision"]
ASR_REVISION = pins["asr"]["revision"]
run_script("scripts/prepare_splits.py", "--root", DATA_ROOT, "--output", out / "splits", "--pilot-size", PILOT_SIZE)
frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(out / "environment.txt").write_text(frozen)
def infer(name, mode, split="train", ids=None, limit=None):
    args = ["--root", DATA_ROOT, "--split", split, "--output", out / f"{RUN_NAME}_{name}.jsonl",
            "--model", MODEL_ID, "--revision", MODEL_REVISION, "--asr", mode,
            "--seconds-per-sample", SECONDS_PER_SAMPLE]
    if mode != "gold": args += ["--asr-model", ASR_MODEL, "--asr-revision", ASR_REVISION]
    if ids: args += ["--ids", out / "splits" / f"{ids}.json"]
    if limit: args += ["--limit", limit]
    if split == "train": args += ["--evaluate"]
    run_script("run_inference.py", *args)
def summary(name):
    return json.loads((out / f"{RUN_NAME}_{name}.summary.json").read_text())
def check_smoke():
    # Run in a fresh interpreter so source modules from a previous notebook run cannot be reused.
    run_script("scripts/check_smoke.py", "--output", out / f"{RUN_NAME}_smoke.jsonl",
               "--model", MODEL_ID, "--revision", MODEL_REVISION,
               "--asr", ASR_BACKEND, "--asr-model", ASR_MODEL,
               "--asr-revision", ASR_REVISION, "--seconds-per-sample", SECONDS_PER_SAMPLE)
''')
markdown('## 4. Smoke: 5 mẫu pilot từ audio → Agent → JSONL\nNếu đa số timeout hoặc lỗi context, dừng đọc trace trước khi chạy 30 mẫu. File JSONL chỉ xuất sau khi replay khớp hash.')
code('smoke', '''infer("smoke", ASR_BACKEND, ids="pilot", limit=5)
print(summary("smoke"))
check_smoke()
''')
markdown('## 5. Pilot: cùng Agent, transcript chuẩn so với ASR\nChạy lại cell này sẽ tiếp tục phần chưa xong. Không đổi cấu hình giữa chừng; nếu đổi thì dùng RUN_NAME mới.')
code('pilot', '''check_smoke()
infer("pilot_gold", "gold", ids="pilot")
infer("pilot_asr", ASR_BACKEND, ids="pilot")
for name in ("pilot_gold", "pilot_asr"):
    print(name, summary(name))
run_script("scripts/compare.py", out / f"{RUN_NAME}_pilot_gold.jsonl", out / f"{RUN_NAME}_pilot_asr.jsonl")
''')
markdown('## 6. Xác nhận trên các nhóm còn lại\nBật sau khi xem pilot. Dùng tập này để xác nhận quyết định, không lặp tìm prompt liên tục trên cùng tập.')
code('confirm', '''RUN_CONFIRM = False
if RUN_CONFIRM:
    check_smoke()
    infer("confirm_gold", "gold", ids="confirm")
    infer("confirm_asr", ASR_BACKEND, ids="confirm")
    run_script("scripts/compare.py", out / f"{RUN_NAME}_confirm_gold.jsonl", out / f"{RUN_NAME}_confirm_asr.jsonl")
else:
    print("Confirm skipped. Set RUN_CONFIRM=True when ready.")
''')
markdown('## 7. Chạy public và kiểm tra file nộp\nBật khi đã chọn cấu hình và kiểm tra thời gian. Không có nhãn public trong pipeline.')
code('public', '''RUN_PUBLIC = False
if RUN_PUBLIC:
    check_smoke()
    infer("public", ASR_BACKEND, split="public_test")
    file = out / f"{RUN_NAME}_public.jsonl"
    run_script("evaluate.py", "--root", DATA_ROOT, "--split", "public_test", "--prediction", file)
    shutil.copy2(file, out / "submission.jsonl")
    print("File nộp:", out / "submission.jsonl")
else:
    print("Public skipped. Set RUN_PUBLIC=True after selecting the configuration.")
''')
markdown('## 8. Xuất weights và gói Docker chạy offline\nTải thêm khoảng dung lượng checkpoint LLM. Cell không tự build hoặc nộp Docker; xem docker/README.md để chạy smoke với mạng tắt.')
code('export', '''EXPORT_OFFLINE = False
if EXPORT_OFFLINE:
    deploy = Path("/kaggle/working/lifemate-v2-deploy")
    shutil.copytree(repo, deploy, dirs_exist_ok=True)
    run_script("scripts/export_models.py", "--pins", out / f"{RUN_NAME}.pins.json", "--output", deploy / "models")
    (deploy / "runtime_config.json").write_text(json.dumps({"asr": ASR_BACKEND, "model": "/code/models/llm", "asr_model": "/code/models/asr"}))
    print("Docker build context ready:", deploy)
else:
    print("Offline export skipped.")
print("Lưu thư mục kết quả trong Output trước khi kết thúc phiên:", out)
''')

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"isGpuEnabled": True, "isInternetEnabled": True}}, "nbformat": 4, "nbformat_minor": 5}
path = ROOT / "notebooks/LifeMate_v2_Kaggle.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
for cell in cells:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), cell["id"], "exec")
print(path)
