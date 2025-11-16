# 項目文件指南 (File Guide)

本文檔說明項目中每個文件的用途。

## 📁 根目錄文件

| 文件名 | 用途 | 重要性 |
|--------|------|--------|
| `README.md` | 項目主文檔，包含使用說明、架構圖、示例 | ⭐⭐⭐ |
| `LICENSE` | MIT開源許可證 | ⭐⭐ |
| `pyproject.toml` | 現代Python項目配置（Black、pytest等） | ⭐⭐⭐ |
| `requirements.txt` | 生產環境Python依賴 | ⭐⭐⭐ |
| `requirements-dev.txt` | 開發環境額外依賴 | ⭐⭐ |
| `config.yaml` | 模型配置示例文件 | ⭐ |
| `.env.example` | 環境變量模板 | ⭐⭐ |
| `.gitignore` | Git忽略文件規則 | ⭐⭐⭐ |
| `.pre-commit-config.yaml` | Pre-commit hooks配置 | ⭐⭐ |
| `Makefile` | 常用命令快捷方式 | ⭐⭐ |
| `Dockerfile` | Docker容器配置 | ⭐⭐ |
| `docker-compose.yml` | Docker Compose多容器配置 | ⭐⭐ |
| `train.py` | 快速訓練腳本 | ⭐⭐⭐ |
| `setup.sh` | Linux/Mac快速設置腳本 | ⭐⭐ |
| `setup.bat` | Windows快速設置腳本 | ⭐⭐ |
| `CHECKLIST.md` | 部署前檢查清單 | ⭐⭐ |
| `PROJECT_SUMMARY.md` | 項目完成總結 | ⭐⭐ |
| `FILE_GUIDE.md` | 本文件，文件說明指南 | ⭐ |

## 📂 src/ - 源代碼

### src/
| 文件名 | 用途 | 說明 |
|--------|------|------|
| `__init__.py` | 包初始化 | 定義公開的API |
| `config.py` | 配置管理 | CNNConfig、RAGConfig類 |

### src/models/
| 文件名 | 用途 | 說明 |
|--------|------|------|
| `__init__.py` | 模型包初始化 | 導出BaseMLModel |
| `base_model.py` | **抽象基類** | 統一的模型介面 ⭐⭐⭐ |
| `cnn_classifier.py` | CNN實現 | TinyConvNet + CNNClassifier ⭐⭐⭐ |
| `rag_system.py` | RAG實現 | ChromaDB + Claude ⭐⭐⭐ |

### src/utils/
| 文件名 | 用途 | 說明 |
|--------|------|------|
| `__init__.py` | 工具包初始化 | 導出工具函數 |
| `metrics.py` | 工具函數 | 驗證、格式化等 |

## 🧪 tests/ - 測試

| 文件名 | 用途 | 測試內容 |
|--------|------|----------|
| `__init__.py` | 測試包初始化 | - |
| `conftest.py` | pytest配置和fixtures | 共享測試數據 ⭐⭐ |
| `test_cnn.py` | CNN測試 | 單元測試 + 性能驗證 ⭐⭐⭐ |
| `test_rag.py` | RAG測試 | 單元測試 + API集成 ⭐⭐⭐ |
| `test_integration.py` | 整合測試 | 跨模型測試 + 端到端 ⭐⭐⭐ |

## 📚 docs/ - 文檔

| 文件名 | 用途 | 說明 |
|--------|------|------|
| `QUICKSTART.md` | 快速開始指南 | 5-10分鐘入門 |
| `architecture.md` | 架構文檔 | 詳細設計說明 |

## 🐳 .github/ - CI/CD

| 文件路徑 | 用途 | 說明 |
|----------|------|------|
| `.github/workflows/ci.yml` | **CI/CD Pipeline** | GitHub Actions配置 ⭐⭐⭐ |

## 📊 data/ - 數據

| 目錄 | 用途 | 說明 |
|------|------|------|
| `data/sample_images/` | CNN樣本圖像 | MNIST會自動下載到這裡 |
| `data/knowledge_base/` | RAG知識庫 | 示例文檔 |
| `data/knowledge_base/sample_documents.md` | 示例知識庫 | 用於RAG演示 |

## 💾 models/ - 模型存儲

| 文件 | 用途 | 說明 |
|------|------|------|
| `models/README.md` | 目錄說明 | 模型文件使用指南 |
| `models/.gitkeep` | Git追蹤空目錄 | - |

## 📓 notebooks/ - Jupyter筆記本

| 目錄 | 用途 | 說明 |
|------|------|------|
| `notebooks/` | （預留）探索性分析 | 可添加演示筆記本 |

## 🔍 文件用途說明

### ⭐⭐⭐ 核心文件（必須理解）

1. **`src/models/base_model.py`**
   - 項目的核心設計
   - 定義統一介面
   - **面試重點**：展示設計模式理解

2. **`src/models/cnn_classifier.py`**
   - CNN實現
   - 展示PyTorch使用
   - **面試重點**：模型訓練流程

3. **`src/models/rag_system.py`**
   - RAG實現
   - 展示LLM集成
   - **面試重點**：向量檢索 + 生成

4. **`.github/workflows/ci.yml`**
   - CI/CD核心配置
   - **面試重點**：MLOps實踐

5. **測試文件（test_*.py）**
   - 測試驅動開發
   - **面試重點**：代碼質量保證

### ⭐⭐ 重要配置文件

- `pyproject.toml`: 現代化Python配置
- `requirements.txt`: 依賴管理
- `.gitignore`: 版本控制
- `Dockerfile`: 容器化

### ⭐ 輔助文件

- 文檔（README, QUICKSTART等）
- 腳本（train.py, setup.sh等）
- 示例配置

## 📝 代碼行數統計

```
核心代碼（src/）:      ~800行
測試代碼（tests/）:    ~900行
配置+文檔:             ~1000+行
總計:                  ~2700+行
```

## 🎯 學習路徑建議

### 第一步：理解架構
1. 閱讀 `README.md`
2. 查看 `docs/architecture.md`
3. 理解 `src/models/base_model.py`

### 第二步：運行代碼
1. 使用 `setup.sh` 快速設置
2. 運行 `train.py`
3. 查看測試 `pytest tests/ -v`

### 第三步：深入代碼
1. 研究 `cnn_classifier.py` 和 `rag_system.py`
2. 理解測試結構
3. 查看CI/CD配置

### 第四步：自定義擴展
1. 嘗試添加新模型
2. 修改配置
3. 添加新測試

## 🔧 常用文件組合

### 開發時常用
- `src/models/*.py` - 修改模型
- `tests/test_*.py` - 添加測試
- `train.py` - 快速驗證

### 調試時常用
- `pyproject.toml` - pytest配置
- `.env` - 環境變量
- `Makefile` - 快捷命令

### 部署時關注
- `.github/workflows/ci.yml` - CI/CD
- `Dockerfile` - 容器化
- `requirements.txt` - 依賴

### 展示時重點
- `README.md` - 項目說明
- `docs/architecture.md` - 設計文檔
- CI/CD通過記錄 - 質量證明

## 💡 提示

- **不要修改**：`.gitkeep` 文件（用於追蹤空目錄）
- **必須修改**：README中的用戶名、LICENSE中的版權信息
- **推薦修改**：添加個人風格到文檔
- **可選修改**：配置參數、模型架構

## 🎓 面試準備

準備解釋這些文件的作用：
1. `base_model.py` - 設計模式
2. `ci.yml` - DevOps實踐
3. `test_integration.py` - 測試策略
4. `Dockerfile` - 容器化知識

---

**記住**：不需要記住每個文件，但要理解整體架構！

如果面試官問："你能解釋一下這個項目的結構嗎？"

**最佳回答**：
> "這個項目採用分層架構：
> 1. **核心層**（src/models/）實現統一的模型介面
> 2. **測試層**（tests/）保證代碼質量
> 3. **CI/CD層**（.github/）自動化流程
> 4. **容器層**（Docker）支持部署
> 
> 關鍵設計是BaseMLModel抽象類，讓不同類型的模型可以用同一套CI/CD流程。"

簡潔、專業、重點突出！🎯
