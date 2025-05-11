/**
 * Defines the detailed specification of an AI model, including its architecture,
 * metadata parsed from the model file, and provenance information.
 */
export interface ModelSpecification {
  // Intrinsic Model Details (parsed from model file, e.g., GGUF headers)
  architecture?: string;       // e.g., 'llama', 'qwen', 'phi2'
  quantization?: string;       // e.g., 'Q4_K_M', 'Q8_0', 'F16'
  layerCount?: number;         // Number of transformer layers
  embeddingLength?: number;    // Dimension of the embedding layer
  contextLength?: number;      // Maximum context window size supported by the model
  vocabSize?: number;          // Vocabulary size
  headCount?: number;          // Number of attention heads
  headCountKv?: number;        // Number of KV heads (for GQA/MQA)
  ropeFrequencyBase?: number;  // RoPE base frequency
  ropeFrequencyScale?: number; // RoPE frequency scale

  // Creator/Licensing Information (if available in model metadata)
  creator?: string;
  license?: string;
  modelName?: string;          // Name of the model as specified in its metadata
  ggufVersion?: number;        // Version of the GGUF format used

  // Tensor Info (optional, could be extensive, maybe a summary or specific tensors)
  // tensorInfo?: Record<string, { type: string; shape: number[] }>;

  // Provenance Data (how and when this specific instance was obtained)
  sourceURL?: string;          // The original URL from which the model was downloaded
  downloadDate?: string;       // ISO 8601 string of when the model was downloaded/added
  fileName?: string;           // Original filename if loaded from a File object
  fileSize?: number;           // Original file size in bytes

  // Other potentially useful metadata fields from GGUF
  // Example: 'general.architecture', 'general.name', 'tokenizer.ggml.tokens'
  // We can add more specific fields as GGUF parsing becomes more detailed.
  [key: string]: any; // Allow for other GGUF metadata fields not explicitly listed
} 