/**
 * GGUF Parser
 * 
 * Parses GGUF file headers to extract model metadata and specifications.
 * Based on the GGUF format specifications from llama.cpp.
 */

import { ModelSpecification } from './model-spec.js';

// GGUF Magic Numbers and Constants
const GGUF_MAGIC = 0x46554747; // "GGUF" in ASCII (little-endian)
const GGUF_VERSION_MIN_SUPPORTED = 2;
const GGUF_VERSION_MAX_SUPPORTED = 3; // Update as newer versions become supported

// Basic GGUF data types
enum GGUFDataType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

/**
 * Parses the GGUF header from an ArrayBuffer and extracts model metadata
 * @param buffer The ArrayBuffer containing the model file data
 * @returns ModelSpecification with parsed metadata
 * @throws Error if the header is invalid or unsupported
 */
export function parseGGUFHeader(buffer: ArrayBuffer): ModelSpecification {
  const dataView = new DataView(buffer);
  let offset = 0;

  // Check magic number
  const magic = dataView.getUint32(offset, true); // true for little-endian
  offset += 4;
  
  if (magic !== GGUF_MAGIC) {
    throw new Error(`Invalid GGUF file format: Magic number mismatch. Expected ${GGUF_MAGIC}, got ${magic}`);
  }

  // Read version
  const version = dataView.getUint32(offset, true);
  offset += 4;

  if (version < GGUF_VERSION_MIN_SUPPORTED || version > GGUF_VERSION_MAX_SUPPORTED) {
    throw new Error(`Unsupported GGUF version: ${version}. Supported versions: ${GGUF_VERSION_MIN_SUPPORTED}-${GGUF_VERSION_MAX_SUPPORTED}`);
  }

  // Initialize model specification
  const spec: ModelSpecification = {
    ggufVersion: version
  };

  // Read tensor count
  const tensorCount = version >= 3 ? 
    Number(readUint64(dataView, offset)) : 
    dataView.getUint32(offset, true);
  offset += version >= 3 ? 8 : 4;

  // Skip tensor data for now as we only need metadata
  
  // Read key-value metadata count
  const metadataCount = version >= 3 ? 
    Number(readUint64(dataView, offset)) : 
    dataView.getUint32(offset, true);
  offset += version >= 3 ? 8 : 4;

  // Parse metadata
  for (let i = 0; i < metadataCount; i++) {
    // Read key
    const keyLength = version >= 3 ? 
      Number(readUint64(dataView, offset)) : 
      dataView.getUint32(offset, true);
    offset += version >= 3 ? 8 : 4;
    
    const keyBytes = new Uint8Array(buffer, offset, keyLength);
    const key = new TextDecoder().decode(keyBytes);
    offset += keyLength;

    // Read value type
    const valueType = dataView.getUint32(offset, true);
    offset += 4;

    // Parse value based on type
    let value;
    switch (valueType) {
      case GGUFDataType.UINT32:
        value = dataView.getUint32(offset, true);
        offset += 4;
        break;
      case GGUFDataType.INT32:
        value = dataView.getInt32(offset, true);
        offset += 4;
        break;
      case GGUFDataType.FLOAT32:
        value = dataView.getFloat32(offset, true);
        offset += 4;
        break;
      case GGUFDataType.BOOL:
        value = dataView.getUint8(offset) !== 0;
        offset += 1;
        break;
      case GGUFDataType.STRING:
        const strLength = version >= 3 ? 
          Number(readUint64(dataView, offset)) : 
          dataView.getUint32(offset, true);
        offset += version >= 3 ? 8 : 4;
        
        const strBytes = new Uint8Array(buffer, offset, strLength);
        value = new TextDecoder().decode(strBytes);
        offset += strLength;
        break;
      // For simplicity, skip other types for now
      default:
        console.warn(`Skipping unsupported metadata type ${valueType} for key "${key}"`);
        // Skip the value bytes
        if (valueType === GGUFDataType.UINT8 || valueType === GGUFDataType.INT8 || valueType === GGUFDataType.BOOL) {
          offset += 1;
        } else if (valueType === GGUFDataType.UINT16 || valueType === GGUFDataType.INT16) {
          offset += 2;
        } else if (valueType === GGUFDataType.UINT32 || valueType === GGUFDataType.INT32 || valueType === GGUFDataType.FLOAT32) {
          offset += 4;
        } else if (valueType === GGUFDataType.UINT64 || valueType === GGUFDataType.INT64 || valueType === GGUFDataType.FLOAT64) {
          offset += 8;
        } else {
          // For array and other complex types, we would need to skip properly, but for now
          // we'll just break out of the metadata loop to avoid parsing errors
          console.warn(`Encountered complex type, stopping metadata parsing at key "${key}"`);
          i = metadataCount; // exit the loop
          break;
        }
        continue;
    }

    // Map key to model specification properties
    mapKeyToModelSpec(spec, key, value);
  }

  return spec;
}

/**
 * Maps a GGUF metadata key to the appropriate property in ModelSpecification
 */
function mapKeyToModelSpec(spec: ModelSpecification, key: string, value: any): void {
  // Map common GGUF metadata keys to our ModelSpecification properties
  switch (key) {
    case "general.architecture":
      spec.architecture = value;
      break;
    case "general.name":
      spec.modelName = value;
      break;
    case "llama.context_length":
    case "phi2.context_length":
    case "context_length":
      spec.contextLength = value;
      break;
    case "llama.embedding_length":
    case "phi2.embedding_length":
    case "embedding_length":
      spec.embeddingLength = value;
      break;
    case "general.quantization_version":
    case "quantization.version":
      spec.quantization = value;
      break;
    case "llama.attention.head_count":
    case "head_count":
      spec.headCount = value;
      break;
    case "llama.attention.head_count_kv":
    case "head_count_kv":
      spec.headCountKv = value;
      break;
    case "general.file_type":
      // Just for logging, not stored
      console.log(`GGUF file type: ${value}`);
      break;
    case "tokenizer.ggml.model":
      // This might indicate the model type/family
      if (!spec.architecture) {
        spec.architecture = value;
      }
      break;
    case "general.quantization":
      spec.quantization = value;
      break;
    default:
      // Store any other metadata in the general key-value store
      spec[key] = value;
      break;
  }
}

/**
 * Helper function to read uint64 values from DataView
 * Note: JavaScript numbers can safely represent integers up to 2^53-1
 */
function readUint64(dataView: DataView, offset: number): bigint {
  const low = dataView.getUint32(offset, true);
  const high = dataView.getUint32(offset + 4, true);
  return BigInt(low) + (BigInt(high) << 32n);
} 