# Financial Report Analysis with LLM & RAG

Hệ thống hỏi đáp báo cáo tài chính sử dụng **Large Language Model (LLM)** kết hợp **Retrieval-Augmented Generation (RAG)**.

Project được thử nghiệm trên **Apple Form 10-K FY2025**. Hệ thống cho phép đặt câu hỏi bằng ngôn ngữ tự nhiên, truy xuất các đoạn liên quan trong báo cáo và sử dụng LLM để sinh câu trả lời dựa trên evidence được tìm thấy.

## 1. Pipeline

```text
Apple 10-K PDF
      ↓
Text / Table Extraction
      ↓
Chunking + Metadata
      ↓
Embedding
      ↓
FAISS + BM25
      ↓
Hybrid Retrieval (RRF, Top-5)
      ↓
Gemini LLM
      ↓
Answer / Calculation / Evidence
      ↓
Evaluation
```

Retrieval sử dụng:

- Dense Retrieval: `BAAI/bge-small-en-v1.5` + FAISS
- Lexical Retrieval: BM25
- Hybrid Retrieval: Dense + BM25, hợp nhất bằng Reciprocal Rank Fusion (RRF)
- Top-K: 5

LLM sử dụng:

- `gemini-3.5-flash`

## 2. Ba cấu hình thí nghiệm

Project so sánh ba cấu hình trên cùng bộ 30 câu hỏi:

### Configuration A — LLM Only

```text
Question → LLM → Answer
```

Dùng làm baseline, không cung cấp nội dung 10-K cho LLM.

### Configuration B — LLM + RAG

```text
Question
   ↓
Hybrid Retrieval Top-5
   ↓
LLM
   ↓
Answer
```

Dùng để đo tác động của retrieval.

### Configuration C — LLM + RAG + Evidence

```text
Question
   ↓
Hybrid Retrieval Top-5
   ↓
LLM
   ↓
Answer + Calculation + Evidence
```

Cấu hình B và C sử dụng cùng frozen Top-5 retrieved contexts để đảm bảo so sánh công bằng.

Ground truth chỉ được sử dụng ở bước evaluation và không được đưa vào retrieval hoặc generation.

## 3. Cấu trúc project

```text
financial-report-rag/
├── data/
│   ├── raw/
│   │   └── apple_2025_10k.pdf
│   ├── processed/
│   │   └── chunks.json
│   └── evaluation/
│       ├── questions.json
│       ├── retrieval_results.json
│       ├── llm_only.json
│       ├── rag.json
│       └── rag_evidence.json
│
├── indexes/
│   └── faiss.index
│
├── src/
│   ├── extract.py
│   ├── retrieval.py
│   ├── llm.py
│   ├── run_experiments.py
│   ├── evaluate.py
│   └── demo.py
│
├── results/
├── requirements.txt
├── .env.example
└── README.md
```

## 4. Cài đặt

Clone repository:

```bash
git clone <REPOSITORY_URL>
cd financial-report-rag
```

Khuyến nghị tạo virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cài dependencies:

```bash
pip install -r requirements.txt
```

## 5. Cấu hình Gemini API

Tạo file:

```text
.env
```

tại thư mục gốc của project.

Ví dụ:

```env
GEMINI_API_KEY=your_api_key_here
```

Không commit `.env` hoặc API key lên GitHub.

## 6. Chạy demo

Từ thư mục gốc của project:

```bash
python src/demo.py
```

Demo cho phép thử ba cấu hình:

```text
A - LLM Only
B - LLM + RAG
C - LLM + RAG + Evidence
```

Với cấu hình B/C, hệ thống thực hiện Hybrid Retrieval để tìm Top-5 chunks liên quan trước khi gọi LLM.

Lưu ý: lần chạy retrieval đầu tiên có thể chậm hơn do model embedding cần được load vào bộ nhớ.

## 7. Chạy thí nghiệm A/B/C

### Configuration A

```bash
python src/run_experiments.py --config a
```

### Configuration B

```bash
python src/run_experiments.py --config b
```

### Configuration C

```bash
python src/run_experiments.py --config c
```

Kết quả được lưu tại:

```text
data/evaluation/llm_only.json
data/evaluation/rag.json
data/evaluation/rag_evidence.json
```

## 8. Chạy evaluation

```bash
python src/evaluate.py
```

Các metric chính bao gồm:

- Answer Accuracy
- Numerical Accuracy
- Retrieval Recall@5
- Evidence Accuracy
- Reasoning Accuracy
- Trend Accuracy
- Abstention Accuracy
- Hallucination Rate
- Latency
- Token usage

## 9. Kết quả chính

Trên benchmark gồm 30 câu hỏi từ Apple Form 10-K FY2025:

| Configuration | Answer Accuracy |
|---|---:|
| A — LLM Only | 10.0% |
| B — LLM + RAG | 90.0% |
| C — RAG + Evidence | 86.7% |

Retrieval benchmark:

| Method | Recall@5 |
|---|---:|
| BM25 | 70.4% |
| Dense | 81.5% |
| Hybrid (RRF) | **88.9%** |

Hybrid Retrieval được chọn và freeze cho cấu hình B/C vì đạt Recall@5 tổng thể cao nhất.

## 10. Phạm vi

Đây là **research prototype / MVP**, được xây dựng để đánh giá LLM + RAG trên báo cáo tài chính.

Benchmark hiện tại giới hạn ở:

- Apple Form 10-K FY2025
- 30 câu hỏi
- 6 nhóm: Fact Retrieval, Comparison, Calculation, Reasoning, Trend và Unanswerable

Kết quả không nên được khái quát trực tiếp sang mọi doanh nghiệp hoặc mọi loại báo cáo tài chính.

Hệ thống là công cụ **tra cứu và kiểm chứng thông tin**, không phải hệ thống tư vấn đầu tư.