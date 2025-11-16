from anthropic import Anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from .base_model import BaseMLModel

class RAGSystem(BaseMLModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Embedding模型 (80MB, 快速)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 向量數據庫 (本地)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("knowledge_base")
        
        # Anthropic Claude
        self.claude = Anthropic(api_key=config.get('api_key'))
    
    def ingest_documents(self, documents: list[str]):
        """吸收文檔到向量庫"""
        embeddings = self.embedder.encode(documents)
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
    
    def retrieve(self, query: str, top_k: int = 3):
        """檢索相關文檔"""
        query_embedding = self.embedder.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        return results['documents'][0]
    
    def generate(self, query: str, context: list[str]):
        """使用Claude生成答案"""
        prompt = f"""Based on the following context, answer the question.

Context:
{chr(10).join(f'- {doc}' for doc in context)}

Question: {query}

Answer:"""
        
        message = self.claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    def query(self, question: str):
        """完整的RAG流程"""
        context = self.retrieve(question)
        answer = self.generate(question, context)
        return {
            'answer': answer,
            'context': context
        }
    
    def evaluate(self, test_queries):
        """評估檢索質量"""
        # 使用預定義的測試集
        precision_scores = []
        for query, expected_docs in test_queries:
            retrieved = self.retrieve(query)
            # 計算精確度
            precision = len(set(retrieved) & set(expected_docs)) / len(retrieved)
            precision_scores.append(precision)
        
        return {
            'retrieval_precision': sum(precision_scores) / len(precision_scores),
            'avg_latency_ms': self._measure_latency()
        }