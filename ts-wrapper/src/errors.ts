/**
 * Custom error types for the LocalWebAI library
 * Provides specific error classes for different failure scenarios during model loading and operation
 * 
 * This module implements a comprehensive error hierarchy for the LocalWebAI system.
 * It enables precise error classification, propagation, and user-friendly reporting
 * across various failure scenarios - from network issues to model compatibility problems.
 */

/**
 * Base error class for all LocalWebAI errors
 * 
 * This class serves as the foundation for the error hierarchy, enabling
 * type-checking and consistent error handling throughout the application.
 * All specific error types extend this base class for consistency.
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
 * 
 * Used when there are problems downloading models or other resources
 * from remote URLs, including connection failures, HTTP errors,
 * timeouts, and CORS issues.
 */
export class NetworkError extends LocalWebAIError {
  /**
   * @param message The error message
   * @param status Optional HTTP status code if available
   * @param url Optional URL that caused the error
   */
  constructor(message: string, public readonly status?: number, public readonly url?: string) {
    super(`Network error: ${message}${status ? ` (status ${status})` : ''}${url ? ` for ${url}` : ''}`);
  }
}

/**
 * File-related errors (read failures, format issues, etc.)
 * 
 * Used when there are problems reading from or writing to files,
 * particularly when dealing with user-uploaded model files.
 * This includes file access issues, corruption, or format mismatches.
 */
export class FileError extends LocalWebAIError {
  /**
   * @param message The error message
   * @param fileName Optional name of the file that caused the error
   */
  constructor(message: string, public readonly fileName?: string) {
    super(`File error: ${message}${fileName ? ` (${fileName})` : ''}`);
  }
}

/**
 * Errors related to the GGUF file format and parsing
 * 
 * Used when the model file is recognized as GGUF but contains
 * parsing issues, invalid structures, or corrupted data within the format.
 * This is different from format mismatches (which use FileError)
 * or version incompatibilities (which use ModelCompatibilityError).
 */
export class GGUFParsingError extends LocalWebAIError {
  /**
   * @param message The error message
   * @param details Optional object with additional details about the parsing error
   */
  constructor(message: string, public readonly details?: any) {
    super(`GGUF parsing error: ${message}`);
    this.details = details;
  }
}

/**
 * Errors related to model compatibility
 * 
 * Used when a model is not compatible with the current system,
 * typically due to version mismatches in the GGUF format or
 * other compatibility constraints.
 */
export class ModelCompatibilityError extends LocalWebAIError {
  /**
   * @param message The error message
   * @param actualVersion Optional version found in the model
   * @param minSupported Optional minimum supported version
   * @param maxSupported Optional maximum supported version
   */
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
 * 
 * Used when there are issues storing, retrieving, or managing
 * cached models in the browser's IndexedDB storage. This includes
 * quota errors, permission issues, or data corruption in the cache.
 */
export class CacheError extends LocalWebAIError {
  /**
   * @param message The error message
   * @param modelId Optional identifier of the model that caused the cache error
   */
  constructor(message: string, public readonly modelId?: string) {
    super(`Cache error: ${message}${modelId ? ` for model ${modelId}` : ''}`);
  }
}

/**
 * Errors related to the virtual file system (VFS) in the Wasm worker
 * 
 * Used when there are problems with the Emscripten virtual filesystem
 * used by the WebAssembly module, such as file creation failures,
 * permission issues, or memory constraints.
 */
export class VFSError extends LocalWebAIError {
  /**
   * @param message The error message
   * @param path Optional VFS path that caused the error
   */
  constructor(message: string, public readonly path?: string) {
    super(`VFS error: ${message}${path ? ` (${path})` : ''}`);
  }
}

/**
 * Errors related to Wasm initialization or execution
 * 
 * Used when there are problems with the WebAssembly module itself,
 * such as initialization failures, memory limitations, or
 * incompatibility with the browser environment.
 */
export class WasmError extends LocalWebAIError {
  /**
   * @param message The error message about the WebAssembly issue
   */
  constructor(message: string) {
    super(`WebAssembly error: ${message}`);
  }
}

/**
 * Errors raised when an operation is cancelled by the user
 * 
 * Used when the user explicitly cancels an operation,
 * such as model loading, through the UI or programmatically
 * via an AbortController.
 */
export class OperationCancelledError extends LocalWebAIError {
  /**
   * @param message Optional custom message about the cancellation
   */
  constructor(message: string = "Operation cancelled by user") {
    super(message);
  }
}

/**
 * Errors occurring during model initialization or execution
 * 
 * Used when there are problems initializing or running the model
 * after it has been successfully loaded, such as insufficient memory,
 * invalid inference parameters, or internal model errors.
 */
export class ModelInitializationError extends LocalWebAIError {
  /**
   * @param message The error message about the initialization issue
   */
  constructor(message: string) {
    super(`Model initialization error: ${message}`);
  }
}

/**
 * Function to determine the appropriate error class based on error details
 * 
 * This function examines arbitrary errors and converts them to specific
 * LocalWebAIError subtypes based on their characteristics. It's used to
 * ensure consistent error handling even when errors originate from
 * third-party code or browser APIs.
 * 
 * @param error Any error object or string to classify
 * @returns A properly typed LocalWebAIError instance
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
      message.includes('invalid format') || 
      message.includes('corrupted'))) {
    return new GGUFParsingError(message.replace(/^VFS error: /, ''));
  }
  
  // Errors related to data processing within Wasm, like memory allocation from bad data
  if (message.includes('Invalid typed array length') || 
      (error.name === 'RangeError' && message.toLowerCase().includes('size'))) { // Broader RangeError for size issues
    // This often indicates the WASM module tried to process invalid GGUF data
    // leading to an attempt to allocate memory or create an array with an invalid size.
    return new GGUFParsingError(`Failed to process model data: ${message}`);
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