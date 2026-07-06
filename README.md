# 🏥 Hệ thống Trợ lý Y tế Thông minh (Intelligent Medical Assistant RAG)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0%2B-61DAFB.svg)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX--Runtime-Inference-green.svg)](https://onnxruntime.ai/)

Hệ thống Trợ lý Y tế Thông minh hỗ trợ tư vấn sức khỏe bằng tiếng Việt, được xây dựng trên kiến trúc **Full-stack React + FastAPI**, tích hợp mô hình định tuyến **PhoBERT ONNX**, viết lại câu hỏi bằng **ViT5-base**, và trả lời dựa trên kỹ thuật **Hybrid RAG** kết hợp suy luận từ **Local LLM (GGUF)**.

> [!IMPORTANT]
> **Tuyên bố Miễn trừ Trách nhiệm Y tế (Medical Disclaimer):**
> Đây là nguyên mẫu nghiên cứu & giáo dục hỗ trợ tư vấn y tế. Hệ thống **KHÔNG** thay thế chẩn đoán, đơn thuốc hay lời khuyên điều trị từ bác sĩ chuyên khoa. Đối với các triệu chứng cấp cứu hoặc chuyển biến nặng, người dùng phải lập tức liên hệ cơ sở y tế gần nhất hoặc gọi tổng đài cấp cứu (115).

---

## 🌟 Điểm nhấn & Định hướng Công nghệ (Current Direction)

Dự án được xây dựng và tối ưu hóa theo xu hướng hiện đại nhất của các hệ thống AI tác tử (Agentic AI) cục bộ:

1. **Định tuyến Siêu tốc với PhoBERT ONNX Classifier:**
   - Thay thế việc phân loại bằng LLM chậm chạp (Zero-shot) hoặc quy tắc cứng nhắc (Regex) bằng một mô hình **PhoBERT-base-v2** được fine-tune chuyên biệt và đóng gói dưới dạng **ONNX Runtime**.
   - Định tuyến câu hỏi chuẩn xác thành 4 nhóm chính: `medical` (y tế), `emergency` (cấp cứu), `out-of-scope` (ngoài luồng), và `faq` (hỏi đáp chatbot) chỉ với độ trễ cực thấp từ **10-30ms**.

2. **Viết lại câu hỏi thông minh với ViT5 Seq2Seq:**
   - Tích hợp mô hình **ViT5-base** chuyên viết lại câu hỏi tiếng Việt (`Query Rewrite`).
   - Tự động chuyển đổi các câu hỏi không dấu, viết sai chính tả, hoặc hành văn tự nhiên của người dùng (ví dụ: *"bo toi bi dau nguc du doi cuu voi"*) thành các truy vấn y khoa chuẩn hóa để cải thiện độ chính xác khi tìm kiếm RAG.
   - Có thể xuất sang **ONNX** (`optimum-cli`) để chạy mượt mà trên CPU.

3. **Điều phối Workflow linh hoạt bằng LangGraph:**
   - Hệ thống được cấu trúc dưới dạng Đồ thị trạng thái (`StateGraph`), bắt đầu từ `intent_router` đến `query_rewrite`, `hybrid_retrieval`, `evidence_grading` và `answer_generation`.
   - Nếu phát hiện `emergency` hoặc `out-of-scope`, hệ thống lập tức đi tới node kếtthuốc sớm (`early_exit`) mà không chạy qua luồng RAG hay LLM sinh chữ, bảo đảm an toàn y tế và phản hồi tức thì.

4. **Tìm kiếm Lai (Hybrid Retrieval) & Thẩm định Bằng chứng:**
   - Kết hợp tìm kiếm ngữ nghĩa (**ChromaDB Vector Store**) và tìm kiếm từ khóa (**BM25**), sau đó dung hợp kết quả bằng thuật toán **RRF (Reciprocal Rank Fusion)**.
   - **Evidence Grader** tự động chấm điểm độ tin cậy của tài liệu. Nếu dữ liệu nội bộ không đủ thông tin, hệ thống sẽ kích hoạt **Web Crawler** để cào bổ sung từ các nguồn y khoa uy tín hàng đầu (`MedlinePlus`, `FDA`, `WHO`).

5. **Inference Cục bộ Hiệu năng cao với llama-server (GGUF):**
   - Không sử dụng API cloud đắt đỏ, backend tự động quản lý vòng đời của **`llama-server`** để chạy suy luận mô hình lượng hóa **Qwen GGUF** (ví dụ: `models/qwen3-4b-thinking.gguf`).
   - Khởi động và tắt tiến trình nền `llama-server` đồng bộ theo Server FastAPI qua cơ chế `lifespan`.

---

## 🏗️ Kiến trúc Hệ thống (System Architecture)

Sơ đồ mô tả luồng xử lý thực tế trong Đồ thị trạng thái LangGraph của hệ thống:

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	intent_router(intent_router)
	query_rewrite(query_rewrite)
	hybrid_retrieval(hybrid_retrieval)
	retrieval_assessment(retrieval_assessment)
	web_retrieval(web_retrieval)
	direct_llm(direct_llm)
	rag_generation(rag_generation)
	response_validation(response_validation)
	early_exit(early_exit)
	__end__([<p>__end__</p>]):::last
	__start__ --> intent_router;
	direct_llm --> response_validation;
	hybrid_retrieval --> retrieval_assessment;
	intent_router -.-> direct_llm;
	intent_router -.-> early_exit;
	intent_router -.-> query_rewrite;
	query_rewrite --> hybrid_retrieval;
	rag_generation --> response_validation;
	retrieval_assessment -.-> early_exit;
	retrieval_assessment -. &nbsp;high_score&nbsp; .-> rag_generation;
	retrieval_assessment -. &nbsp;low_score&nbsp; .-> web_retrieval;
	web_retrieval --> retrieval_assessment;
	early_exit --> __end__;
	response_validation --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

---

## 📂 Cấu trúc Thư mục Dự án

```text
.
├── backend/                       # Backend API Server & Xử lý RAG lõi
│   ├── api.py                     # Máy chủ FastAPI (lifespan quản lý llama-server)
│   ├── main.py                    # Giao diện CLI hỗ trợ kiểm thử & Ingest dữ liệu
│   ├── config.py                  # Cấu hình tham số và đường dẫn hệ thống
│   ├── requirements.txt           # Thư viện Python (optimum, onnxruntime, langgraph...)
│   ├── data/                      # Dữ liệu y khoa đã xử lý (processed)
│   └── src/                       # Các module xử lý nghiệp vụ và tiện ích
│       ├── pipeline/              # Phân hệ Điều phối & Đồ thị (LangGraph)
│       │   ├── graph.py           # Định nghĩa đồ thị trạng thái LangGraph
│       │   └── registry.py        # Quản lý Prompt và Data tập trung
│       ├── nodes/                 # Phân hệ Nghiệp vụ y khoa (Cognitive Nodes)
│       │   ├── router.py          # Định tuyến sử dụng PhoBERT ONNX
│       │   ├── rewriter.py        # Viết lại câu bằng ViT5-base
│       │   ├── generator.py       # Soạn thảo câu trả lời RAG
│       │   └── ...
│       ├── store/                 # Phân hệ Truy xuất dữ liệu (RAG Store)
│       │   ├── retriever.py       # Tìm kiếm lai BM25 + Vector DB
│       │   └── ...
│       └── utils/                 # Tiện ích & Quản trị tiến trình
│           └── server.py          # Quản lý tiến trình nền llama-server
├── frontend/                      # Giao diện Web App (React + Vite)
│   ├── src/                       # Mã nguồn giao diện và xử lý SSE Stream
│   └── ...
├── notebooks/                     # Nghiên cứu và Huấn luyện mô hình
│   ├── 04_train_phobert_intent.ipynb # Train mô hình Intent Classifier
│   └── 05_train_vit5_rewrite.ipynb    # Train mô hình Query Rewrite
├── models/                        # Vector DB và Trọng số mô hình
│   ├── phobert-intent-onnx/       # Mô hình định tuyến PhoBERT dạng ONNX
│   ├── vit5-rewrite-onnx/         # Mô hình viết lại câu ViT5 dạng ONNX (nếu có)
│   └── qwen3-4b-thinking.gguf     # Trọng số LLM phục vụ RAG
└── README.md                      # Tài liệu hướng dẫn hệ thống
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

Hệ thống được phát triển và chạy mượt mà trên môi trường **Windows PowerShell** hoặc **Linux/macOS**.

### 1. Cấu hình Backend & Cài đặt Thư viện

Di chuyển vào thư mục dự án, tạo môi trường ảo và cài đặt các phụ thuộc cần thiết (đặc biệt là ONNX Runtime):

```powershell
# Di chuyển tới dự án
cd Intelligent-Medical-Assistant-for-Healthcare-Consultation

# Tạo và kích hoạt virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 2. Chuẩn bị Trọng số Mô hình

Hệ thống yêu cầu các file mô hình được đặt chính xác trong thư mục `models/` tại thư mục gốc:

1. **LLM (GGUF):** Tải file mô hình lượng hóa `qwen3-4b-thinking.gguf` (hoặc tương đương) và lưu vào `models/qwen3-4b-thinking.gguf`.
2. **PhoBERT ONNX:** Đảm bảo thư mục mô hình định tuyến ONNX đã được đặt tại `models/phobert-intent-onnx/` (chứa `model.onnx`, `config.json`...).

### 3. Nạp Cơ sở dữ liệu Tri thức (Vector DB Ingestion)

Tiến hành xử lý dữ liệu thô, sinh chỉ mục tìm kiếm lai (Vector + BM25):

```powershell
cd backend
python main.py --ingest
```

### 4. Khởi chạy Ứng dụng Full-stack

Mở hai terminal song song để chạy cả API backend và giao diện UI frontend.

#### 🌐 Terminal 1: Khởi chạy FastAPI Backend
```powershell
.\.venv\Scripts\Activate.ps1
cd backend
uv run api.py
```
*Lưu ý:* Khi khởi động, backend sẽ tự động gọi file thực thi `llama-server` chạy ngầm ở cổng 8080 để load mô hình GGUF. Đồng thời log `ONNX Model Loaded Successfully!` sẽ báo hiệu PhoBERT ONNX đã sẵn sàng.

#### 🖥️ Terminal 2: Khởi chạy React Web App
```powershell
cd frontend
npm install
npm run dev
```
Truy cập trình duyệt theo địa chỉ: `http://localhost:5173`.

---

## 🛡️ Tiêu chuẩn Định dạng & An toàn Y khoa

- **Ngắt dòng tự nhiên:** Câu trả lời từ LLM được tinh chỉnh bằng các hướng dẫn hệ thống chi tiết giúp tối ưu khả năng hiển thị Markdown (sử dụng danh sách thụt dòng, chia đoạn bằng khoảng trắng và bôi đậm từ khóa y học quan trọng).
- **Phòng ngừa Khẩn cấp:** Mô hình định tuyến PhoBERT ONNX nhận diện các tình huống khẩn cấp chỉ trong vài mili-giây để đưa ra cảnh báo khẩn cấp hướng dẫn người dùng kết nối ngay với số điện thoại 115.
- **Ràng buộc Chứng cứ:** Nếu thông tin truy xuất RAG không đủ hoặc bị chấm điểm thấp bởi *Evidence Grader*, hệ thống sẽ trả về câu từ chối chuẩn mực nhằm tránh hiện tượng ảo giác (hallucination) gây nguy hiểm trong tư vấn y khoa.
- **Gắn Disclaimer:** Mọi câu trả lời liên quan tới bệnh lý đều tự động được đính kèm tuyên bố miễn trừ trách nhiệm pháp lý phù hợp với mức độ rủi ro đã phân loại.

---

## 💡 Hướng dẫn Xử lý Sự cố (Troubleshooting)

- **Lỗi `Failed to load ONNX model: ...`:** Kiểm tra xem bạn đã cài đặt `optimum[onnxruntime]` và `onnxruntime` chưa. Đồng thời kiểm tra xem thư mục `models/phobert-intent-onnx` có tồn tại ngay tại root của dự án không.
- **Lỗi liên quan tới `llama-server`:** Đảm bảo file thực thi `llama-server` (hoặc `llama-server.exe` trên Windows) nằm trong PATH hệ thống hoặc được đặt trong thư mục `backend/` để manager có thể tìm thấy và tự động khởi chạy. Bạn có thể kiểm tra log tiến trình này trong file `backend/llama_log.txt`.
