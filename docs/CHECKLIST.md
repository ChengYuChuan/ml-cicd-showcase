# 部署前檢查清單 (Pre-Deployment Checklist)

在將項目推送到GitHub之前，請確保完成以下檢查：

## 📋 基本設置

- [ ] 項目已在本地運行成功
- [ ] 所有測試通過 (`pytest tests/ -v`)
- [ ] 代碼格式正確 (`make lint` 或 `black --check src/ tests/`)
- [ ] 沒有硬編碼的API密鑰或敏感信息

## 🔧 配置文件

- [ ] `.env.example` 包含所有必要的環境變量（但沒有實際值）
- [ ] `.gitignore` 正確配置（排除 `.env`, `*.pth`, `__pycache__/` 等）
- [ ] `requirements.txt` 包含所有依賴
- [ ] `pyproject.toml` 配置正確

## 📝 文檔

- [ ] README.md 已更新（替換 `yourusername` 為你的GitHub用戶名）
- [ ] README.md 中的徽章鏈接正確
- [ ] LICENSE 文件中的版權信息正確
- [ ] 所有文檔中的示例代碼可運行

## 🧪 測試

- [ ] 所有單元測試通過
- [ ] 整合測試通過
- [ ] CNN模型達到預期準確率 (>85%)
- [ ] RAG系統可以正常檢索（如果有API key）
- [ ] Docker構建成功 (`docker-compose build`)

## 🐳 Docker

- [ ] Dockerfile 可以成功構建
- [ ] Docker容器中測試通過
- [ ] docker-compose.yml 配置正確
- [ ] 環境變量在Docker中正確傳遞

## 🚀 GitHub設置

- [ ] 在GitHub創建新倉庫（可以是private或public）
- [ ] 倉庫名稱：`ml-cicd-showcase` 或自定義
- [ ] 添加倉庫描述
- [ ] 添加topics標籤：`machine-learning`, `mlops`, `ci-cd`, `pytorch`, `rag`

## 🔑 GitHub Secrets

需要在GitHub倉庫設置中添加的Secrets：

- [ ] `ANTHROPIC_API_KEY` - 你的Anthropic API密鑰
- [ ] （可選）`CODECOV_TOKEN` - 如果使用Codecov

設置路徑：GitHub倉庫 → Settings → Secrets and variables → Actions → New repository secret

## 📊 CI/CD

- [ ] `.github/workflows/ci.yml` 文件存在
- [ ] Workflow配置正確（YAML語法無誤）
- [ ] 所有job名稱清晰
- [ ] 超時設置合理

## 🎨 個性化

### 必須修改的地方：

1. **README.md**
   - [ ] 替換 `yourusername` 為你的GitHub用戶名（多處）
   - [ ] 更新徽章URL
   - [ ] 添加你的聯繫信息

2. **pyproject.toml**
   - [ ] 更新 `authors` 中的名字和郵箱

3. **LICENSE**
   - [ ] 更新版權持有人名稱

4. **setup.sh / setup.bat**
   - [ ] 測試腳本在你的系統上運行

### 可選修改：

- [ ] 添加個人品牌/風格到README
- [ ] 自定義配置文件
- [ ] 添加額外的模型示例

## 🔍 最終檢查

運行以下命令確保一切正常：

```bash
# 1. 代碼質量
make lint

# 2. 快速測試
make test-fast

# 3. 完整測試（包括覆蓋率）
make test-cov

# 4. Docker測試
make docker-test

# 5. 訓練快速驗證
python train.py --model cnn --epochs 1
```

所有命令都應該成功運行！

## 📤 推送到GitHub

一切檢查完畢後：

```bash
# 初始化Git（如果還沒有）
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: ML CI/CD Showcase project"

# 添加遠程倉庫
git remote add origin https://github.com/你的用戶名/ml-cicd-showcase.git

# 推送
git push -u origin main
```

## ✅ 推送後驗證

- [ ] GitHub倉庫頁面顯示正確
- [ ] README正確渲染
- [ ] GitHub Actions開始運行
- [ ] 檢查Actions標籤，確保CI/CD pipeline運行
- [ ] 所有jobs都通過（綠色✓）

## 🎯 如果CI/CD失敗

常見問題：

1. **API Key問題**
   - 檢查GitHub Secrets是否正確設置
   - 密鑰名稱必須完全匹配

2. **依賴安裝失敗**
   - 檢查requirements.txt
   - 可能需要添加版本約束

3. **測試超時**
   - 調整workflow中的timeout-minutes
   - 考慮跳過耗時的測試

4. **Docker構建失敗**
   - 檢查Dockerfile語法
   - 確保所有文件都已提交

## 📈 後續優化（可選）

- [ ] 設置Codecov徽章
- [ ] 添加更多文檔（Jupyter notebooks）
- [ ] 創建GitHub Pages展示
- [ ] 添加更多模型類型
- [ ] 實現模型版本控制（DVC）

## 🎓 求職應用

準備好展示項目：

- [ ] LinkedIn個人資料中添加此項目
- [ ] 準備5分鐘的演示
- [ ] 準備回答技術問題
- [ ] 簡歷中添加項目描述

## 💡 提示

- 保持README清晰簡潔
- 確保所有鏈接可點擊
- 徽章會在第一次Actions運行後顯示
- 定期更新依賴版本

---

**完成所有檢查後，你的項目就可以展示給潛在雇主了！** 🚀

記住：這個項目的目的是**展示你的技能**，所以要確保：
1. ✅ 代碼質量高
2. ✅ 文檔清晰
3. ✅ 可以實際運行
4. ✅ CI/CD正常工作

祝你求職順利！💼
