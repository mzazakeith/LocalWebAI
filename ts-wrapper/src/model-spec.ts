/**
 * Defines the detailed specification of an AI model, including its architecture,
 * metadata parsed from the model file, and provenance information.
 * 
 * This interface serves multiple purposes:
 * 1. Stores metadata extracted from GGUF headers during model parsing
 * 2. Provides configuration information used for model validation
 * 3. Maintains provenance tracking (source, timestamps, file details)
 * 4. Enables display of model capabilities and properties in the UI
 */
export interface ModelSpecification {
  // Intrinsic Model Details (parsed from model file, e.g., GGUF headers)
  /** 
   * The model architecture type (e.g., 'llama', 'qwen', 'phi2')
   * Used for identifying model family and compatibility
   */
  architecture?: string;
  
  /** 
   * Quantization format used by the model (e.g., 'Q4_K_M', 'Q8_0', 'F16')
   * Affects memory usage, inference speed, and model accuracy
   */
  quantization?: string;
  
  /** 
   * Number of transformer layers in the model
   * Correlates with model depth and computational requirements
   */
  layerCount?: number;
  
  /** 
   * Dimension of the embedding layer
   * Affects model capacity and memory requirements
   */
  embeddingLength?: number;
  
  /** 
   * Maximum context window size supported by the model
   * Determines how much text the model can process in a single inference
   */
  contextLength?: number;
  
  /** 
   * Size of vocabulary in the model's tokenizer
   * Affects token generation and model compatibility
   */
  vocabSize?: number;
  
  /** 
   * Number of attention heads in the model
   * Relates to parallel processing capacity and model architecture
   */
  headCount?: number;
  
  /** 
   * Number of key-value attention heads (for GQA/MQA architectures)
   * May differ from headCount in models using grouped-query attention 
   */
  headCountKv?: number;
  
  /** 
   * RoPE base frequency parameter
   * Affects positional encoding and context utilization
   */
  ropeFrequencyBase?: number;
  
  /** 
   * RoPE frequency scaling factor
   * Used for context window extensions in some models
   */
  ropeFrequencyScale?: number;

  // Creator/Licensing Information (if available in model metadata)
  /** Creator or organization that produced the model */
  creator?: string;
  
  /** License information for model usage */
  license?: string;
  
  /** Name of the model as specified in its metadata */
  modelName?: string;
  
  /** 
   * Version of the GGUF format used
   * Critical for compatibility checking
   */
  ggufVersion?: number;

  // Provenance Data (how and when this specific instance was obtained)
  /** 
   * The original URL from which the model was downloaded
   * Used for provenance tracking and attribution
   */
  sourceURL?: string;
  
  /** 
   * ISO 8601 string of when the model was downloaded/added
   * Used for versioning and provenance tracking
   */
  downloadDate?: string;
  
  /** 
   * Original filename if loaded from a File object
   * Used for reference and UI display
   */
  fileName?: string;
  
  /** 
   * Original file size in bytes
   * Used for storage management and UI display
   */
  fileSize?: number;

  /**
   * Allows for other GGUF metadata fields not explicitly listed
   * This provides flexibility to store any additional metadata
   * from the GGUF format without modifying the interface
   */
  [key: string]: any;
} 