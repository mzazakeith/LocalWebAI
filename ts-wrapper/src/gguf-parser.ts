/**
 * GGUF Parser
 * 
 * Parses GGUF file headers to extract model metadata and specifications.
 * Based on the GGUF format specifications from llama.cpp.
 */

import { ModelSpecification } from './model-spec.js';
import { GGUFParsingError, ModelCompatibilityError } from './errors.js';

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
 * @throws GGUFParsingError if the header is invalid
 * @throws ModelCompatibilityError if the version is unsupported
 */
export function parseGGUFHeader(buffer: ArrayBuffer): ModelSpecification {
  if (!buffer || buffer.byteLength < 8) {
    throw new GGUFParsingError(
      "Insufficient data to parse GGUF header (minimum 8 bytes required)",
      { byteLength: buffer?.byteLength ?? 0 }
    );
  }

  const dataView = new DataView(buffer);
  let offset = 0;

  try {
  // Check magic number
  const magic = dataView.getUint32(offset, true); // true for little-endian
  offset += 4;
  
  if (magic !== GGUF_MAGIC) {
      throw new GGUFParsingError(
        `Invalid GGUF file format: Magic number mismatch`,
        { expected: GGUF_MAGIC, actual: magic, hexActual: `0x${magic.toString(16).toUpperCase()}` }
      );
  }

  // Read version
  const version = dataView.getUint32(offset, true);
  offset += 4;

  if (version < GGUF_VERSION_MIN_SUPPORTED || version > GGUF_VERSION_MAX_SUPPORTED) {
      throw new ModelCompatibilityError(
        `Unsupported GGUF version`,
        version,
        GGUF_VERSION_MIN_SUPPORTED,
        GGUF_VERSION_MAX_SUPPORTED
      );
  }

  // Initialize model specification
  const spec: ModelSpecification = {
    ggufVersion: version
  };

    try {
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

      // Sanity check for metadata count to avoid potentially malformed headers
      if (metadataCount < 0 || metadataCount > 1000000) {
        throw new GGUFParsingError(
          `Invalid metadata count`,
          { count: metadataCount }
        );
      }

  // Parse metadata
  for (let i = 0; i < metadataCount; i++) {
    // Read key
    const keyLength = version >= 3 ? 
      Number(readUint64(dataView, offset)) : 
      dataView.getUint32(offset, true);
        
        // Sanity check for key length
        if (keyLength <= 0 || keyLength > 1000) {
          throw new GGUFParsingError(
            `Invalid key length`,
            { length: keyLength, index: i }
          );
        }
        
    offset += version >= 3 ? 8 : 4;
        
        // Check bounds before reading key
        if (offset + keyLength > buffer.byteLength) {
          throw new GGUFParsingError(
            `Buffer overflow while reading key`,
            { offset, keyLength, bufferLength: buffer.byteLength }
          );
        }
    
    const keyBytes = new Uint8Array(buffer, offset, keyLength);
    const key = new TextDecoder().decode(keyBytes);
    offset += keyLength;

    // Read value type
        if (offset + 4 > buffer.byteLength) {
          throw new GGUFParsingError(
            `Buffer overflow while reading value type`,
            { offset, bufferLength: buffer.byteLength }
          );
        }
        
    const valueType = dataView.getUint32(offset, true);
    offset += 4;

        // Validate value type is within enum range
        if (valueType < 0 || valueType > 12) {
          throw new GGUFParsingError(
            `Invalid value type`,
            { type: valueType, key }
          );
        }

    // Parse value based on type
    let value;
    switch (valueType) {
      case GGUFDataType.UINT32:
            if (offset + 4 > buffer.byteLength) {
              throw new GGUFParsingError(
                `Buffer overflow while reading UINT32 value`,
                { offset, bufferLength: buffer.byteLength, key }
              );
            }
        value = dataView.getUint32(offset, true);
        offset += 4;
        break;
      case GGUFDataType.INT32:
            if (offset + 4 > buffer.byteLength) {
              throw new GGUFParsingError(
                `Buffer overflow while reading INT32 value`,
                { offset, bufferLength: buffer.byteLength, key }
              );
            }
        value = dataView.getInt32(offset, true);
        offset += 4;
        break;
      case GGUFDataType.FLOAT32:
            if (offset + 4 > buffer.byteLength) {
              throw new GGUFParsingError(
                `Buffer overflow while reading FLOAT32 value`,
                { offset, bufferLength: buffer.byteLength, key }
              );
            }
        value = dataView.getFloat32(offset, true);
        offset += 4;
        break;
      case GGUFDataType.BOOL:
            if (offset + 1 > buffer.byteLength) {
              throw new GGUFParsingError(
                `Buffer overflow while reading BOOL value`,
                { offset, bufferLength: buffer.byteLength, key }
              );
            }
        value = dataView.getUint8(offset) !== 0;
        offset += 1;
        break;
      case GGUFDataType.STRING:
        const strLength = version >= 3 ? 
          Number(readUint64(dataView, offset)) : 
          dataView.getUint32(offset, true);
            
            if (strLength < 0 || strLength > 1000000) {
              throw new GGUFParsingError(
                `Invalid string length`,
                { length: strLength, key }
              );
            }
            
        offset += version >= 3 ? 8 : 4;
            
            // Check bounds before reading string
            if (offset + strLength > buffer.byteLength) {
              throw new GGUFParsingError(
                `Buffer overflow while reading string value`,
                { offset, strLength, bufferLength: buffer.byteLength, key }
              );
            }
        
        const strBytes = new Uint8Array(buffer, offset, strLength);
            try {
        value = new TextDecoder().decode(strBytes);
            } catch (error) {
              throw new GGUFParsingError(
                `Error decoding string value`,
                { key, error }
              );
            }
        offset += strLength;
        break;
      // For simplicity, skip other types for now
      default:
        console.warn(`Skipping unsupported metadata type ${valueType} for key "${key}"`);
            // Skip the value bytes with proper bounds checking
        if (valueType === GGUFDataType.UINT8 || valueType === GGUFDataType.INT8 || valueType === GGUFDataType.BOOL) {
              if (offset + 1 > buffer.byteLength) {
                throw new GGUFParsingError(
                  `Buffer overflow while skipping small value`,
                  { offset, bufferLength: buffer.byteLength, key, type: valueType }
                );
              }
          offset += 1;
        } else if (valueType === GGUFDataType.UINT16 || valueType === GGUFDataType.INT16) {
              if (offset + 2 > buffer.byteLength) {
                throw new GGUFParsingError(
                  `Buffer overflow while skipping 16-bit value`,
                  { offset, bufferLength: buffer.byteLength, key, type: valueType }
                );
              }
          offset += 2;
        } else if (valueType === GGUFDataType.UINT32 || valueType === GGUFDataType.INT32 || valueType === GGUFDataType.FLOAT32) {
              if (offset + 4 > buffer.byteLength) {
                throw new GGUFParsingError(
                  `Buffer overflow while skipping 32-bit value`,
                  { offset, bufferLength: buffer.byteLength, key, type: valueType }
                );
              }
          offset += 4;
        } else if (valueType === GGUFDataType.UINT64 || valueType === GGUFDataType.INT64 || valueType === GGUFDataType.FLOAT64) {
              if (offset + 8 > buffer.byteLength) {
                throw new GGUFParsingError(
                  `Buffer overflow while skipping 64-bit value`,
                  { offset, bufferLength: buffer.byteLength, key, type: valueType }
                );
              }
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
    } catch (error) {
      // If we got a parsing error after successfully reading the magic number and version,
      // we still return a basic specification with the version info
      console.warn("Error parsing GGUF metadata, returning basic specification:", error);
      if (!(error instanceof GGUFParsingError)) {
        throw new GGUFParsingError(
          `Error parsing GGUF metadata: ${error instanceof Error ? error.message : String(error)}`,
          { version }
        );
      }
      throw error;
  }

  return spec;
  } catch (error) {
    // Re-throw LocalWebAIError types, but convert other errors to GGUFParsingError
    if (error instanceof GGUFParsingError || error instanceof ModelCompatibilityError) {
      throw error;
    }
    throw new GGUFParsingError(
      `Unexpected error parsing GGUF header: ${error instanceof Error ? error.message : String(error)}`
    );
  }
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
 * @throws GGUFParsingError if reading past buffer bounds
 */
function readUint64(dataView: DataView, offset: number): bigint {
  if (offset + 8 > dataView.byteLength) {
    throw new GGUFParsingError(
      `Buffer overflow while reading uint64 value`,
      { offset, bufferLength: dataView.byteLength }
    );
  }
  
  const low = dataView.getUint32(offset, true);
  const high = dataView.getUint32(offset + 4, true);
  return BigInt(low) + (BigInt(high) << 32n);
} 