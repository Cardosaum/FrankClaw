use async_trait::async_trait;
use frankclaw_core::error::{
    EmbeddingProvider as EmbeddingProviderError, FrankClawError, MemoryStore as MemoryStoreError,
    Result,
};
use secrecy::{ExposeSecret, SecretString};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tracing::debug;

fn embedding_provider_err(msg: impl Into<String>) -> FrankClawError {
    EmbeddingProviderError { msg: msg.into() }.build()
}

fn memory_store_err(msg: impl Into<String>) -> FrankClawError {
    MemoryStoreError { msg: msg.into() }.build()
}

/// Abstract embedding provider.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync + 'static {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
}

/// OpenAI-compatible embedding provider.
pub struct OpenAiEmbeddingProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: SecretString,
    model: String,
    dim: usize,
}

impl OpenAiEmbeddingProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: SecretString,
        model: impl Into<String>,
    ) -> Self {
        let model = model.into();
        let dim = match model.as_str() {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        };
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key,
            model,
            dim,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| embedding_provider_err("empty embedding response"))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // Batch up to 100 texts per request.
        let mut all_results = Vec::with_capacity(texts.len());
        for batch in texts.chunks(100) {
            let body = serde_json::json!({
                "model": self.model,
                "input": batch,
            });
            let resp = self
                .client
                .post(format!(
                    "{}/v1/embeddings",
                    self.base_url.trim_end_matches('/')
                ))
                .header(
                    "Authorization",
                    format!("Bearer {}", self.api_key.expose_secret()),
                )
                .json(&body)
                .send()
                .await
                .map_err(|e| embedding_provider_err(format!("HTTP request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(embedding_provider_err(format!(
                    "API error {status}: {text}"
                )));
            }

            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| embedding_provider_err(format!("failed to parse response: {e}")))?;

            let data = json["data"]
                .as_array()
                .ok_or_else(|| embedding_provider_err("missing 'data' array in response"))?;

            for item in data {
                let embedding: Vec<f32> = item["embedding"]
                    .as_array()
                    .ok_or_else(|| embedding_provider_err("missing embedding array"))?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                all_results.push(embedding);
            }
        }
        Ok(all_results)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Ollama embedding provider (sequential, no native batch).
pub struct OllamaEmbeddingProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dim: usize,
}

impl OllamaEmbeddingProvider {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, dim: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            dim,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let body = serde_json::json!({
            "model": self.model,
            "prompt": text,
        });
        let resp = self
            .client
            .post(format!(
                "{}/api/embeddings",
                self.base_url.trim_end_matches('/')
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| embedding_provider_err(format!("Ollama request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(embedding_provider_err(format!(
                "Ollama error {status}: {text}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| embedding_provider_err(format!("failed to parse Ollama response: {e}")))?;

        json["embedding"]
            .as_array()
            .ok_or_else(|| embedding_provider_err("missing 'embedding' in Ollama response"))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<_>>()
            .pipe_ok()
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

trait PipeOk: Sized {
    fn pipe_ok(self) -> Result<Self> {
        Ok(self)
    }
}
impl<T> PipeOk for T {}

/// Caching wrapper that stores embeddings in SQLite keyed by SHA-256 of the text.
pub struct CachedEmbeddingProvider<P> {
    inner: P,
    db: Arc<std::sync::Mutex<rusqlite::Connection>>,
}

impl<P: EmbeddingProvider> CachedEmbeddingProvider<P> {
    pub fn new(inner: P, db_path: &std::path::Path) -> Result<Self> {
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| memory_store_err(format!("failed to open embedding cache: {e}")))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )
        .map_err(|e| memory_store_err(format!("failed to create cache table: {e}")))?;
        Ok(Self {
            inner,
            db: Arc::new(std::sync::Mutex::new(conn)),
        })
    }

    fn text_hash(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn get_cached(&self, hash: &str) -> Option<Vec<f32>> {
        let db = self.db.lock().ok()?;
        let mut stmt = db
            .prepare("SELECT embedding FROM embedding_cache WHERE text_hash = ?1 AND model = ?2")
            .ok()?;
        let blob: Vec<u8> = stmt
            .query_row(rusqlite::params![hash, self.inner.model_name()], |row| {
                row.get(0)
            })
            .ok()?;
        Some(bytes_to_f32(&blob))
    }

    fn put_cached(&self, hash: &str, embedding: &[f32]) {
        if let Ok(db) = self.db.lock() {
            let blob = f32_to_bytes(embedding);
            let _ = db.execute(
                "INSERT OR REPLACE INTO embedding_cache (text_hash, model, embedding) VALUES (?1, ?2, ?3)",
                rusqlite::params![hash, self.inner.model_name(), blob],
            );
        }
    }
}

#[async_trait]
impl<P: EmbeddingProvider> EmbeddingProvider for CachedEmbeddingProvider<P> {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let hash = Self::text_hash(text);
        if let Some(cached) = self.get_cached(&hash) {
            debug!(hash = %hash, "embedding cache hit");
            return Ok(cached);
        }
        let result = self.inner.embed(text).await?;
        self.put_cached(&hash, &result);
        Ok(result)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let hash = Self::text_hash(text);
            if let Some(cached) = self.get_cached(&hash) {
                results.push(cached);
            } else {
                uncached_indices.push(i);
                uncached_texts.push(*text);
                results.push(Vec::new()); // placeholder
            }
        }

        if !uncached_texts.is_empty() {
            let refs: Vec<&str> = uncached_texts.clone();
            let fresh = self.inner.embed_batch(&refs).await?;
            for (j, embedding) in fresh.into_iter().enumerate() {
                let idx = uncached_indices[j];
                let hash = Self::text_hash(texts[idx]);
                self.put_cached(&hash, &embedding);
                results[idx] = embedding;
            }
        }

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
}

/// Gemini embedding provider using the batchEmbedContents API.
pub struct GeminiEmbeddingProvider {
    client: reqwest::Client,
    api_key: SecretString,
    model: String,
    dim: usize,
}

impl GeminiEmbeddingProvider {
    pub fn new(api_key: SecretString, model: impl Into<String>) -> Self {
        let model = model.into();
        let dim = match model.as_str() {
            "text-embedding-004" => 768,
            "text-embedding-005" => 768,
            "embedding-001" => 768,
            _ => 768,
        };
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            dim,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for GeminiEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| embedding_provider_err("empty Gemini embedding response"))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut all_results = Vec::with_capacity(texts.len());
        for batch in texts.chunks(100) {
            let requests: Vec<serde_json::Value> = batch
                .iter()
                .map(|text| {
                    serde_json::json!({
                        "model": format!("models/{}", self.model),
                        "content": { "parts": [{ "text": text }] }
                    })
                })
                .collect();

            let body = serde_json::json!({ "requests": requests });
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
                self.model,
                self.api_key.expose_secret()
            );

            let resp = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| embedding_provider_err(format!("Gemini request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(embedding_provider_err(format!(
                    "Gemini API error {status}: {text}"
                )));
            }

            let json: serde_json::Value = resp.json().await.map_err(|e| {
                embedding_provider_err(format!("failed to parse Gemini response: {e}"))
            })?;

            let embeddings = json["embeddings"].as_array().ok_or_else(|| {
                embedding_provider_err("missing 'embeddings' array in Gemini response")
            })?;

            for emb in embeddings {
                let values: Vec<f32> = emb["values"]
                    .as_array()
                    .ok_or_else(|| embedding_provider_err("missing 'values' in Gemini embedding"))?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                all_results.push(values);
            }
        }
        Ok(all_results)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Voyage AI embedding provider (OpenAI-compatible API).
pub struct VoyageEmbeddingProvider {
    client: reqwest::Client,
    api_key: SecretString,
    model: String,
    dim: usize,
}

impl VoyageEmbeddingProvider {
    pub fn new(api_key: SecretString, model: impl Into<String>) -> Self {
        let model = model.into();
        let dim = match model.as_str() {
            "voyage-3" => 1024,
            "voyage-3-lite" => 512,
            "voyage-large-2" => 1536,
            "voyage-2" => 1024,
            "voyage-code-2" => 1536,
            _ => 1024,
        };
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            dim,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for VoyageEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| embedding_provider_err("empty Voyage embedding response"))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut all_results = Vec::with_capacity(texts.len());
        for batch in texts.chunks(100) {
            let body = serde_json::json!({
                "model": self.model,
                "input": batch,
            });

            let resp = self
                .client
                .post("https://api.voyageai.com/v1/embeddings")
                .header(
                    "Authorization",
                    format!("Bearer {}", self.api_key.expose_secret()),
                )
                .json(&body)
                .send()
                .await
                .map_err(|e| embedding_provider_err(format!("Voyage request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(embedding_provider_err(format!(
                    "Voyage API error {status}: {text}"
                )));
            }

            let json: serde_json::Value = resp.json().await.map_err(|e| {
                embedding_provider_err(format!("failed to parse Voyage response: {e}"))
            })?;

            let data = json["data"]
                .as_array()
                .ok_or_else(|| embedding_provider_err("missing 'data' array in Voyage response"))?;

            for item in data {
                let embedding: Vec<f32> = item["embedding"]
                    .as_array()
                    .ok_or_else(|| {
                        embedding_provider_err("missing embedding array in Voyage response")
                    })?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                all_results.push(embedding);
            }
        }
        Ok(all_results)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| {
            let mut bytes = [0; 4];
            bytes.copy_from_slice(chunk);
            f32::from_le_bytes(bytes)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_roundtrip() {
        let data = vec![1.0f32, -2.5, std::f32::consts::PI, 0.0];
        let bytes = f32_to_bytes(&data);
        let restored = bytes_to_f32(&bytes);
        assert_eq!(data, restored);
    }

    #[test]
    fn gemini_dimension_mapping() {
        let key = SecretString::from("test-key");
        let p = GeminiEmbeddingProvider::new(key.clone(), "text-embedding-004");
        assert_eq!(p.dimension(), 768);
        assert_eq!(p.model_name(), "text-embedding-004");

        let p2 = GeminiEmbeddingProvider::new(key, "text-embedding-005");
        assert_eq!(p2.dimension(), 768);
    }

    #[test]
    fn voyage_dimension_mapping() {
        let key = SecretString::from("test-key");
        let p = VoyageEmbeddingProvider::new(key.clone(), "voyage-3");
        assert_eq!(p.dimension(), 1024);
        assert_eq!(p.model_name(), "voyage-3");

        let p2 = VoyageEmbeddingProvider::new(key.clone(), "voyage-3-lite");
        assert_eq!(p2.dimension(), 512);

        let p3 = VoyageEmbeddingProvider::new(key.clone(), "voyage-large-2");
        assert_eq!(p3.dimension(), 1536);

        let p4 = VoyageEmbeddingProvider::new(key, "voyage-code-2");
        assert_eq!(p4.dimension(), 1536);
    }

    #[test]
    fn gemini_unknown_model_defaults_768() {
        let key = SecretString::from("test-key");
        let p = GeminiEmbeddingProvider::new(key, "unknown-model");
        assert_eq!(p.dimension(), 768);
    }

    #[test]
    fn voyage_unknown_model_defaults_1024() {
        let key = SecretString::from("test-key");
        let p = VoyageEmbeddingProvider::new(key, "unknown-model");
        assert_eq!(p.dimension(), 1024);
    }

    #[test]
    fn text_hash_deterministic() {
        let h1 = CachedEmbeddingProvider::<OpenAiEmbeddingProvider>::text_hash("hello");
        let h2 = CachedEmbeddingProvider::<OpenAiEmbeddingProvider>::text_hash("hello");
        assert_eq!(h1, h2);
        let h3 = CachedEmbeddingProvider::<OpenAiEmbeddingProvider>::text_hash("world");
        assert_ne!(h1, h3);
    }
}
