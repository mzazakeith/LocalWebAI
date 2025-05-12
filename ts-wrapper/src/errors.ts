/**
 * Custom error types for the LocalWebAI library
 * Provides specific error classes for different failure scenarios during model loading and operation
 */

/**
 * Base error class for all LocalWebAI errors
 * Allows for categorizing errors and providing structured error information
 */
export class LocalWebAIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
    // Ensures proper prototype chain when targeting ES5
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Network-related errors (fetch failures, timeouts, etc.)
 */
export class NetworkError extends LocalWebAIError {
  constructor(message: string, public readonly status?: number, public readonly url?: string) {
    super(`Network error: ${message}${status ? ` (status ${status})` : ''}${url ? ` for ${url}` : ''}`);
  }
}

/**
 * File-related errors (read failures, format issues, etc.)
 */
export class FileError extends LocalWebAIError {
  constructor(message: string, public readonly fileName?: string) {
    super(`File error: ${message}${fileName ? ` (${fileName})` : ''}`);
  }
}

/**
 * Errors related to the GGUF file format and parsing
 */
export class GGUFParsingError extends LocalWebAIError {
  constructor(message: string, public readonly details?: any) {
    super(`GGUF parsing error: ${message}`);
    this.details = details;
  }
}

/**
 * Errors related to model compatibility
 */
export class ModelCompatibilityError extends LocalWebAIError {
  constructor(
    message: string, 
    public readonly actualVersion?: number, 
    public readonly minSupported?: number,
    public readonly maxSupported?: number
  ) {
    super(`Model compatibility error: ${message}`);
  }
}

/**
 * Errors related to caching (IndexedDB) operations
 */
export class CacheError extends LocalWebAIError {
  constructor(message: string, public readonly modelId?: string) {
    super(`Cache error: ${message}${modelId ? ` for model ${modelId}` : ''}`);
  }
}

/**
 * Errors related to the virtual file system (VFS) in the Wasm worker
 */
export class VFSError extends LocalWebAIError {
  constructor(message: string, public readonly path?: string) {
    super(`VFS error: ${message}${path ? ` (${path})` : ''}`);
  }
}

/**
 * Errors related to Wasm initialization or execution
 */
export class WasmError extends LocalWebAIError {
  constructor(message: string) {
    super(`WebAssembly error: ${message}`);
  }
}

/**
 * Errors raised when an operation is cancelled by the user
 */
export class OperationCancelledError extends LocalWebAIError {
  constructor(message: string = "Operation cancelled by user") {
    super(message);
  }
}

/**
 * Errors occurring during model initialization or execution
 */
export class ModelInitializationError extends LocalWebAIError {
  constructor(message: string) {
    super(`Model initialization error: ${message}`);
  }
}

/**
 * Function to determine the appropriate error class based on error details
 * Useful for converting generic errors to more specific types
 */
export function classifyError(error: any): LocalWebAIError {
  // If the error is already a LocalWebAIError, return it as is
  if (error instanceof LocalWebAIError) {
    return error;
  }
  
  // Handle DOM AbortError as OperationCancelledError
  if (error instanceof DOMException && error.name === 'AbortError') {
    return new OperationCancelledError();
  }

  // Extract message if error is a standard Error
  const message = error instanceof Error ? error.message : String(error);
  
  // Check for GGUF version/format errors first (highest priority)
  if (message.includes('GGUF version') || 
      (message.includes('version') && message.includes('supported')) ||
      (message.includes('magic number') && message.includes('GGUF'))) {
    // Try to extract version numbers if available
    const actualVersionMatch = message.match(/actual(?:Version)?[:\s]+(\d+)/i);
    const minSupportedMatch = message.match(/min(?:imum)?(?:Supported)?[:\s]+(\d+)/i);
    const maxSupportedMatch = message.match(/max(?:imum)?(?:Supported)?[:\s]+(\d+)/i);
    
    const actualVersion = actualVersionMatch ? parseInt(actualVersionMatch[1], 10) : undefined;
    const minSupported = minSupportedMatch ? parseInt(minSupportedMatch[1], 10) : undefined;
    const maxSupported = maxSupportedMatch ? parseInt(maxSupportedMatch[1], 10) : undefined;
    
    return new ModelCompatibilityError(
      message.replace(/^VFS error: /, ''), 
      actualVersion, 
      minSupported, 
      maxSupported
    );
  }
  
  // Network errors
  if (message.includes('Failed to fetch') || 
      message.includes('fetch failed') ||
      message.includes('network request') ||
      message.includes('HTTP status')) {
    // Try to extract status code and URL if available
    const statusMatch = message.match(/status[:\s]+(\d+)/i);
    const urlMatch = message.match(/(?:for|url)[:\s]+(https?:\/\/[^\s]+)/i);
    
    return new NetworkError(
      message, 
      statusMatch ? parseInt(statusMatch[1], 10) : undefined,
      urlMatch ? urlMatch[1] : undefined
    );
  }
  
  // GGUF parsing errors (structural issues with the file format)
  if (message.includes('GGUF') && 
     (message.includes('parsing') || 
      message.includes('invalid') || 
      message.includes('corrupt') ||
      message.includes('header') ||
      message.includes('buffer overflow'))) {
    return new GGUFParsingError(message.replace(/^VFS error: /, ''));
  }
  
  // VFS errors related to model files - may need to be re-classified
  if (message.includes('VFS error') || message.includes('virtual filesystem')) {
    // Check if the VFS error is actually due to a file format issue
    if (message.includes('model file') && 
       (message.includes('invalid') || message.includes('format') || message.includes('not found'))) {
      return new FileError(message.replace(/^VFS error: /, ''));
    }
    
    // Extract path if available
    const pathMatch = message.match(/\(([^)]+)\)/);
    return new VFSError(message, pathMatch ? pathMatch[1] : undefined);
  }
  
  // File errors 
  if ((message.includes('file') && 
      (message.includes('error') || message.includes('failed') || message.includes('invalid'))) ||
      message.includes('Error reading') ||
      message.includes('stat.isFile')) {
    // Extract filename if available
    const filenameMatch = message.match(/\(([^)]+\.(?:gguf|bin))\)/);
    return new FileError(message, filenameMatch ? filenameMatch[1] : undefined);
  }
  
  // WebAssembly errors
  if (message.includes('WebAssembly') || 
      message.includes('Wasm') || 
      message.includes('Module') && message.includes('initialize')) {
    return new WasmError(message.replace(/^WebAssembly error: /, ''));
  }
  
  // Cache errors
  if (message.includes('cache') || 
      message.includes('IndexedDB') || 
      message.includes('storage')) {
    // Extract model ID if available
    const modelIdMatch = message.match(/model[:\s]+([a-zA-Z0-9_\-.]+)/);
    return new CacheError(message, modelIdMatch ? modelIdMatch[1] : undefined);
  }
  
  // Model initialization errors
  if (message.includes('model') && 
     (message.includes('initialize') || message.includes('start') || message.includes('loading'))) {
    return new ModelInitializationError(message);
  }
  
  // Default case - return a generic LocalWebAIError
  return new LocalWebAIError(message);
}