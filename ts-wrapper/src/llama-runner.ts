import { ModelCache } from './model-cache.js';
import { ModelSpecification } from './model-spec.js'; // Import ModelSpecification
import { ProgressCallback, ProgressInfo, LoadingStage, getStageDescription } from './loading-progress.js'; // Import progress types
import { 
  NetworkError, 
  FileError, 
  GGUFParsingError, 
  ModelCompatibilityError, 
  CacheError,
  VFSError,
  WasmError,
  OperationCancelledError,
  LocalWebAIError,
  ModelInitializationError,
  classifyError
} from './errors.js';

// Define the types for callbacks
export type TokenCallback = (token: string) => void;
export type CompletionCallback = () => void;

// Define the structure for generation parameters, aligning with llama.cpp options
export interface GenerateTextParams {
  n_predict?: number;       // Max tokens to predict. -1 for infinity, -2 for till context limit.
  ctx_size?: number;        // Context size for the model.
  batch_size?: number;      // Batch size for prompt processing.
  temp?: number;            // Temperature for sampling.
  n_gpu_layers?: number;    // Number of layers to offload to GPU (if supported by Wasm build).
  top_k?: number;           // Top-K sampling.
  top_p?: number;           // Top-P (nucleus) sampling.
  no_display_prompt?: boolean; // Whether to include the prompt in the output stream.
  chatml?: boolean;         // Use ChatML prompt format.
  // We can add more parameters here as needed, e.g., repeat_penalty, seed, etc.
}

// Replicate actions from worker for type safety
const workerActions = {
  LOAD: 'LOAD',
  INITIALIZED: 'INITIALIZED',
  RUN_MAIN: 'RUN_MAIN',
  WRITE_RESULT: 'WRITE_RESULT',
  RUN_COMPLETED: 'RUN_COMPLETED',
  LOAD_MODEL_DATA: 'LOAD_MODEL_DATA',
  MODEL_METADATA: 'MODEL_METADATA', // Add the new action
  PROGRESS_UPDATE: 'PROGRESS_UPDATE', // Add progress update action
  CANCEL_LOAD: 'CANCEL_LOAD', // Add cancel action
};

// GGUF version constraints for validation
const MIN_SUPPORTED_GGUF_VERSION = 2;
const MAX_SUPPORTED_GGUF_VERSION = 3;

// Required model metadata fields
const REQUIRED_METADATA_FIELDS = [
  // Critical fields that must be present for compatibility verification
  'ggufVersion'
];

// Optional but important fields (we'll warn if missing)
const IMPORTANT_METADATA_FIELDS = [
  'architecture',
  'contextLength',
  'embeddingLength'
];

export class LlamaRunner {
  private worker: Worker | null = null;
  private modelCache: ModelCache;
  private isInitialized = false;
  private isLoadingModel = false;
  private onModelLoadedCallback: (() => void) | null = null;
  private onModelLoadErrorCallback: ((error: Error) => void) | null = null;
  private currentProgressCallback: ProgressCallback | null = null;
  private currentTokenCallback: TokenCallback | null = null;
  private currentCompletionCallback: CompletionCallback | null = null;
  private currentModelId: string | null = null; // Keep track of the current model ID for metadata
  private currentModelMetadata: ModelSpecification | null = null; // Store current model metadata
  private lastProgressInfo: ProgressInfo | null = null; // Track the last progress update
  private abortController: AbortController | null = null; // For tracking active abort controller

  /**
   * @param workerPath Path to the compiled worker.ts (e.g., 'worker.js')
   * @param wasmModulePath Path to the Emscripten JS glue file (e.g., from llama-cpp-wasm/dist/.../main.js)
   * @param wasmPath Path to the .wasm file (e.g., from llama-cpp-wasm/dist/.../main.wasm)
   */
  constructor(
    private workerPath: string,
    private wasmModulePath: string,
    private wasmPath: string
  ) {
    this.modelCache = new ModelCache();
    this.initWorker();
  }

  private initWorker(): void {
    this.worker = new Worker(this.workerPath, { type: 'module' });

    this.worker.onmessage = (event) => {
      const { event: action, text, error, errorDetails, metadata, stage, ...progressDetails } = event.data;
      switch (action) {
        case workerActions.INITIALIZED:
          this.isInitialized = true;
          if (this.isLoadingModel && this.onModelLoadedCallback) {
            this.onModelLoadedCallback();
          }
          this.isLoadingModel = false;
          this.onModelLoadedCallback = null;
          this.onModelLoadErrorCallback = null;
          break;
        case workerActions.WRITE_RESULT:
          if (this.currentTokenCallback && typeof text === 'string') {
            this.currentTokenCallback(text);
          }
          break;
        case workerActions.RUN_COMPLETED:
          if (this.currentCompletionCallback) {
            this.currentCompletionCallback();
          }
          this.currentTokenCallback = null;
          this.currentCompletionCallback = null;
          break;
        case workerActions.MODEL_METADATA:
          // Handle the new metadata message
          this.handleModelMetadata(metadata as ModelSpecification);
          break;
        case workerActions.PROGRESS_UPDATE:
          // Handle detailed progress updates
          if (stage && this.currentProgressCallback) {
            const progressInfo: ProgressInfo = {
              stage: stage as LoadingStage,
              ...progressDetails
            };
            this.lastProgressInfo = progressInfo;
            this.currentProgressCallback(progressInfo);
          }
          break;
        // Handle potential errors from worker
        case 'ERROR': // Assuming worker posts { event: 'ERROR', message: '...'}
            console.error('Error from worker:', error);
            
            // Create a proper error instance based on errorDetails if available
            let errorInstance: Error;
            
            if (errorDetails) {
              // Create specific error type based on the name
              switch (errorDetails.name) {
                case 'GGUFParsingError':
                  errorInstance = new GGUFParsingError(errorDetails.message, errorDetails.details);
                  break;
                case 'ModelCompatibilityError':
                  errorInstance = new ModelCompatibilityError(
                    errorDetails.message,
                    errorDetails.actualVersion,
                    errorDetails.minSupported,
                    errorDetails.maxSupported
                  );
                  break;
                case 'VFSError':
                  errorInstance = new VFSError(errorDetails.message, errorDetails.path);
                  break;
                case 'WasmError':
                  errorInstance = new WasmError(errorDetails.message);
                  break;
                case 'OperationCancelledError':
                  errorInstance = new OperationCancelledError(errorDetails.message);
                  break;
                case 'ModelInitializationError':
                  errorInstance = new ModelInitializationError(errorDetails.message);
                  break;
                default:
                  // Generic LocalWebAIError for unknown types
                  errorInstance = new LocalWebAIError(errorDetails.message || error);
              }
            } else {
              // Fallback to the error message string
              errorInstance = classifyError(error || 'Unknown worker error');
            }
            
            // Report error through progress callback if available
            if (this.currentProgressCallback) {
              this.currentProgressCallback({
                stage: LoadingStage.ERROR,
                message: errorInstance.message,
                error: errorInstance.message
              });
            }
            
            if (this.isLoadingModel && this.onModelLoadErrorCallback) {
                this.onModelLoadErrorCallback(errorInstance);
            }
            this.isLoadingModel = false;
            this.onModelLoadedCallback = null;
            this.onModelLoadErrorCallback = null;
            // Potentially notify other error listeners if any
            break;
      }
    };

    this.worker.onerror = (event: ErrorEvent) => {
      console.error('Error in LlamaRunner worker:', event);
      // Log more details from the ErrorEvent
      let detailedErrorMessage = 'Worker onerror';
      if (event.message) {
        detailedErrorMessage = event.message;
      } else if (typeof event === 'string') {
        detailedErrorMessage = event;
      }
      console.error(`Worker Error Details: Message: ${event.message}, Filename: ${event.filename}, Lineno: ${event.lineno}, Colno: ${event.colno}`);

      // Create a specific error instance for worker errors
      const error = new WasmError(`Worker error: ${detailedErrorMessage}`);

      // Report error through progress callback if available
      if (this.currentProgressCallback) {
        this.currentProgressCallback({
          stage: LoadingStage.ERROR,
          message: error.message,
          error: error.message
        });
      }

      if (this.isLoadingModel && this.onModelLoadErrorCallback) {
        this.onModelLoadErrorCallback(error);
      }
      this.isLoadingModel = false;
      this.onModelLoadedCallback = null;
      this.onModelLoadErrorCallback = null;
    };

    // Initial message to worker to load Wasm module
    // The worker will then fetch the model if a modelUrl is also passed, or wait for model data.
    this.worker.postMessage({
      event: workerActions.LOAD,
      wasmModulePath: new URL(this.wasmModulePath, window.location.href).href,
      wasmPath: new URL(this.wasmPath, window.location.href).href,
      // modelUrl: initialModelUrl, // Optionally load a default model URL on init
    });
  }

  /**
   * Reports progress to the callback with appropriate stage and details
   */
  private reportProgress(info: Partial<ProgressInfo>): void {
    if (!this.currentProgressCallback) return;
    
    const progressInfo: ProgressInfo = {
      stage: info.stage || (this.lastProgressInfo?.stage || LoadingStage.PREPARING_MODEL_DATA),
      message: info.message || getStageDescription(info.stage || (this.lastProgressInfo?.stage || LoadingStage.PREPARING_MODEL_DATA)),
      loaded: info.loaded,
      total: info.total,
      metadata: info.metadata || this.currentModelMetadata || undefined,
      error: info.error
    };
    
    this.lastProgressInfo = progressInfo;
    this.currentProgressCallback(progressInfo);
  }

  /**
   * Validates and handles the model metadata received from the worker
   * @param metadata The parsed model metadata
   */
  private async handleModelMetadata(metadata: ModelSpecification): Promise<void> {
    // Store the metadata locally
    this.currentModelMetadata = metadata;

    // Validate the metadata
    const validationResult = this.validateModelMetadata(metadata);
    if (validationResult.error) {
      console.error('Model metadata validation failed:', validationResult.error);
      
      // Create the appropriate error instance
      const error = validationResult.errorInstance || 
                    new ModelCompatibilityError(validationResult.error);
      
      this.reportProgress({
        stage: LoadingStage.ERROR,
        message: error.message,
        error: error.message
      });
      
      if (this.isLoadingModel && this.onModelLoadErrorCallback) {
        this.onModelLoadErrorCallback(error);
        this.isLoadingModel = false;
        this.onModelLoadedCallback = null;
        this.onModelLoadErrorCallback = null;
      }
      return;
    }

    // Log any warnings
    if (validationResult.warnings && validationResult.warnings.length > 0) {
      validationResult.warnings.forEach(warning => {
        console.warn('Model metadata warning:', warning);
      });
    }

    // Report that we have metadata
    this.reportProgress({
      metadata: metadata,
      message: 'Model metadata parsed successfully'
    });

    // If we have a current modelId, update the model's metadata in the cache
    if (this.currentModelId) {
      try {
        // Retrieve the model data from cache
        const cachedModelData = await this.modelCache.getModelFromCache(this.currentModelId);
        
        if (cachedModelData) {
          // Add provenance information
          if (typeof this.currentModelId === 'string' && this.currentModelId.startsWith('http')) {
            metadata.sourceURL = this.currentModelId;
          }
          metadata.downloadDate = new Date().toISOString();
          
          // Update the cache with the new metadata
          await this.modelCache.cacheModel(this.currentModelId, cachedModelData, metadata);
          console.log('Updated model cache with metadata for:', this.currentModelId);
        }
      } catch (err) {
        console.warn('Failed to update model cache with metadata:', err);
        // Non-fatal error, continue with model load
      }
    }
  }

  /**
   * Validates the model metadata to ensure compatibility
   * @param metadata The model metadata to validate
   * @returns Object with error message if validation fails, warnings array, and error instance.
   */
  private validateModelMetadata(metadata: ModelSpecification): { 
    error: string | null, 
    warnings: string[], 
    errorInstance?: LocalWebAIError 
  } {
    const warnings: string[] = [];
    
    // Check if metadata object exists
    if (!metadata) {
      return {
        error: 'No metadata provided',
        warnings,
        errorInstance: new ModelCompatibilityError('No metadata provided')
      };
    }
    
    // Check required fields
    for (const field of REQUIRED_METADATA_FIELDS) {
      if (metadata[field] === undefined) {
        return {
          error: `Missing required metadata field: ${field}`,
          warnings,
          errorInstance: new ModelCompatibilityError(`Missing required metadata field: ${field}`)
        };
      }
    }
    
    // Check GGUF version compatibility
    if (metadata.ggufVersion === undefined) {
      return {
        error: 'Missing GGUF version information',
        warnings,
        errorInstance: new ModelCompatibilityError('Missing GGUF version information')
      };
    }
    
    if (metadata.ggufVersion < MIN_SUPPORTED_GGUF_VERSION || 
        metadata.ggufVersion > MAX_SUPPORTED_GGUF_VERSION) {
      return {
        error: `Unsupported GGUF version: ${metadata.ggufVersion}`,
        warnings,
        errorInstance: new ModelCompatibilityError(
          `Unsupported GGUF version`,
          metadata.ggufVersion,
          MIN_SUPPORTED_GGUF_VERSION,
          MAX_SUPPORTED_GGUF_VERSION
        )
      };
    }
    
    // Check for important but non-required fields
    for (const field of IMPORTANT_METADATA_FIELDS) {
      if (metadata[field] === undefined) {
        warnings.push(`Missing recommended metadata field: ${field}`);
      }
    }
    
    // Validate context length if present (e.g., must be a reasonable value)
    if (metadata.contextLength !== undefined) {
      if (typeof metadata.contextLength !== 'number') {
        return {
          error: `Invalid context length: ${metadata.contextLength}. Must be a number.`,
          warnings,
          errorInstance: new ModelCompatibilityError(`Invalid context length: ${metadata.contextLength}. Must be a number.`)
        };
      }
      
      if (metadata.contextLength < 1 || metadata.contextLength > 32768) {
        return {
          error: `Invalid context length: ${metadata.contextLength}. Expected a value between 1 and 32768.`,
          warnings,
          errorInstance: new ModelCompatibilityError(`Invalid context length: ${metadata.contextLength}. Expected a value between 1 and 32768.`)
        };
      }
    }
    
    // Validate embedding length if present
    if (metadata.embeddingLength !== undefined) {
      if (typeof metadata.embeddingLength !== 'number') {
        return {
          error: `Invalid embedding length: ${metadata.embeddingLength}. Must be a number.`,
          warnings,
          errorInstance: new ModelCompatibilityError(`Invalid embedding length: ${metadata.embeddingLength}. Must be a number.`)
        };
      }
      
      if (metadata.embeddingLength < 1) {
        return {
          error: `Invalid embedding length: ${metadata.embeddingLength}. Must be greater than 0.`,
          warnings,
          errorInstance: new ModelCompatibilityError(`Invalid embedding length: ${metadata.embeddingLength}. Must be greater than 0.`)
        };
      }
    }
    
    // All validations passed
    return { error: null, warnings };
  }

  /**
   * Retrieves the current model's metadata
   * @returns The current model's metadata or null if no model is loaded
   */
  public getModelMetadata(): ModelSpecification | null {
    return this.currentModelMetadata;
  }

  /**
   * Retrieves the last reported progress information
   * @returns The last progress info or null if no progress has been reported
   */
  public getLastProgressInfo(): ProgressInfo | null {
    return this.lastProgressInfo;
  }

  /**
   * Cancels the current model loading operation if one is in progress
   * @returns True if a cancellation was initiated, false if no loading operation to cancel
   */
  public cancelLoading(): boolean {
    if (!this.isLoadingModel || !this.abortController) {
      return false;
    }

    // Signal cancellation
    this.abortController.abort();
    
    // Notify the worker to cancel any operations in progress
    if (this.worker) {
      this.worker.postMessage({
        event: workerActions.CANCEL_LOAD
      });
    }

    this.reportProgress({
      stage: LoadingStage.CANCELLED,
      message: 'Model loading cancelled by user'
    });

    return true;
  }

  /**
   * Load a GGUF model from a URL or File object.
   * @param source URL string or File object for the GGUF model.
   * @param modelId A unique ID for caching. If not provided, URL or filename+size will be used.
   * @param progressCallback Optional callback for detailed progress reporting.
   * @param signal Optional AbortSignal for cancellation support.
   * @returns Promise<void> Resolves when the model is loaded and ready for inference.
   * @throws Various error types based on the specific failure, including:
   *   - NetworkError: for URL fetch failures
   *   - FileError: for file reading issues
   *   - GGUFParsingError: for model format issues
   *   - ModelCompatibilityError: for unsupported model versions
   *   - VFSError: for virtual filesystem errors
   *   - WasmError: for WebAssembly-related issues
   */
  public async loadModel(
    source: string | File,
    modelId?: string,
    progressCallback?: ProgressCallback,
    signal?: AbortSignal
  ): Promise<void> {
    if (!this.worker) {
      throw new WasmError('Worker not initialized.');
    }
    if (this.isLoadingModel) {
        throw new ModelInitializationError('Another model is already being loaded.');
    }
    this.isLoadingModel = true;
    this.currentProgressCallback = progressCallback || null;
    this.lastProgressInfo = null;

    // Create a new AbortController if no signal is provided, or use the provided one
    this.abortController = signal ? null : new AbortController();
    const abortSignal = signal || this.abortController!.signal;

    // Set up abort handling
    const handleAbort = () => {
      if (this.isLoadingModel) {
        console.log('Model loading aborted by user');
        
        // Notify the worker 
        if (this.worker) {
          this.worker.postMessage({ 
            event: workerActions.CANCEL_LOAD 
          });
        }
        
        // Create a specific error type for cancellation
        const abortError = new OperationCancelledError('Model loading aborted by user');
        
        // Reject the promise with the cancellation error
        if (this.onModelLoadErrorCallback) {
          this.onModelLoadErrorCallback(abortError);
        }
        
        // Clean up if we have a modelId
        if (this.currentModelId) {
          // Best effort to remove any partial cache entries
          this.modelCache.deleteModel(this.currentModelId).catch(err => {
            console.warn('Error cleaning up cache after abort:', err);
          });
        }
        
        this.isLoadingModel = false;
        this.onModelLoadedCallback = null;
        this.onModelLoadErrorCallback = null;
        
        // Report through progress callback
        this.reportProgress({
          stage: LoadingStage.CANCELLED,
          message: 'Model loading aborted by user'
        });
      }
    };

    // Listen for abort events
    abortSignal.addEventListener('abort', handleAbort);

    return new Promise(async (resolve, reject) => {
      this.onModelLoadedCallback = resolve;
      this.onModelLoadErrorCallback = reject;

      // If already aborted, reject immediately
      if (abortSignal.aborted) {
        this.isLoadingModel = false;
        this.onModelLoadedCallback = null;
        this.onModelLoadErrorCallback = null;
        reject(new OperationCancelledError('Model loading aborted by user'));
        return;
      }

      const actualModelId = modelId || (typeof source === 'string' ? source : `${source.name}-${source.size}`);
      this.currentModelId = actualModelId; // Store current model ID for metadata handling
      this.currentModelMetadata = null; // Reset metadata for new model
      
      let modelData: ArrayBuffer | null = null;
      let cachedModelInfo = null;

      // 1. Try fetching from cache
      try {
        // Report starting to load
        this.reportProgress({
          stage: LoadingStage.PREPARING_MODEL_DATA,
          message: 'Checking model cache'
        });
        
        cachedModelInfo = await this.modelCache.getModelWithSpecificationFromCache(actualModelId);
        modelData = cachedModelInfo?.modelData || null;
        
        // If cache has metadata, store it immediately
        if (cachedModelInfo?.specification) {
          this.currentModelMetadata = cachedModelInfo.specification;
          
          // Report metadata available from cache
          this.reportProgress({
            stage: LoadingStage.METADATA_PARSE_COMPLETE,
            message: 'Model metadata loaded from cache',
            metadata: cachedModelInfo.specification
          });
        }
      } catch (err) {
        console.warn('Error retrieving from cache, will load from source:', err);
        // Convert to specific error type but don't throw - just fall back to loading from source
        const cacheError = new CacheError(
          `Error retrieving from cache: ${err instanceof Error ? err.message : String(err)}`,
          actualModelId
        );
        console.warn(cacheError);
        modelData = null;
      }

      // Check for abort before proceeding
      if (abortSignal.aborted) {
        handleAbort();
        return;
      }

      if (modelData) {
        // Report model found in cache
        this.reportProgress({
          stage: LoadingStage.PREPARING_MODEL_DATA,
          message: 'Model found in cache',
          loaded: modelData.byteLength,
          total: modelData.byteLength
        });
        
        console.log(`Model ${actualModelId} found in cache. Loading from cache.`);
        this.worker?.postMessage({
          event: workerActions.LOAD_MODEL_DATA,
          modelData: modelData,
        });
        // Note: The actual resolution of the promise happens when the worker confirms INITIALIZED
        return;
      }

      // 2. If not cached, fetch/read the model
      console.log(`Model ${actualModelId} not found in cache. Proceeding to load from source.`);
      try {
        if (typeof source === 'string') {
          // Report downloading
          this.reportProgress({
            stage: LoadingStage.DOWNLOADING_FROM_SOURCE,
            message: 'Downloading model from URL'
          });
          
          // Pass the abort signal to fetch
          const response = await fetch(source, { signal: abortSignal });
          if (!response.ok) {
            throw new NetworkError(
              `Failed to download model: ${response.statusText}`,
              response.status,
              source
            );
          }
          
          if (!response.body) {
            throw new NetworkError('Response body is null', undefined, source);
          }

          const contentLength = Number(response.headers.get('Content-Length') || '0');
          const reader = response.body.getReader();
          const chunks: Uint8Array[] = [];
          let receivedLength = 0;

          while (true) {
            // Check for abort before reading
            if (abortSignal.aborted) {
              handleAbort();
              return;
            }

            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            receivedLength += value.length;
            
            // Report download progress
            this.reportProgress({
              stage: LoadingStage.DOWNLOADING_FROM_SOURCE,
              loaded: receivedLength,
              total: contentLength,
              message: `Downloading model: ${contentLength > 0 ? Math.round((receivedLength / contentLength) * 100) : 0}%`
            });
          }

          // Check for abort before processing chunks
          if (abortSignal.aborted) {
            handleAbort();
            return;
          }

          modelData = new Uint8Array(receivedLength).buffer;
          const tempUint8Array = new Uint8Array(modelData);
          let position = 0;
          for (const chunk of chunks) {
            tempUint8Array.set(chunk, position);
            position += chunk.length;
          }
          modelData = tempUint8Array.buffer;

        } else {
          // Handle File object
          // Report reading from file
          this.reportProgress({
            stage: LoadingStage.READING_FROM_FILE,
            message: 'Reading model from file'
          });
          
          modelData = await new Promise<ArrayBuffer>((resolveFile, rejectFile) => {
            const reader = new FileReader();
            reader.onload = (e) => resolveFile(e.target?.result as ArrayBuffer);
            reader.onerror = (e) => {
              rejectFile(new FileError(
                reader.error?.message || 'File reading error',
                source.name
              ));
            };
            reader.onprogress = (e) => {
              if (e.lengthComputable) {
                // Report file reading progress
                this.reportProgress({
                  stage: LoadingStage.READING_FROM_FILE,
                  loaded: e.loaded,
                  total: e.total,
                  message: `Reading model file: ${Math.round((e.loaded / e.total) * 100)}%`
                });
              }
            };
            
            // Handle abort event for FileReader
            const abortHandler = () => {
              reader.abort();
              rejectFile(new OperationCancelledError('Model loading aborted by user'));
            };
            abortSignal.addEventListener('abort', abortHandler, { once: true });
            
            reader.readAsArrayBuffer(source);
          });
        }

        // Check for abort before sending to worker
        if (abortSignal.aborted) {
          handleAbort();
          return;
        }

        if (modelData) {
          // Report preparing model data
          this.reportProgress({
            stage: LoadingStage.PREPARING_MODEL_DATA,
            message: 'Preparing model data for virtual filesystem',
            loaded: modelData.byteLength,
            total: modelData.byteLength
          });
          
          // Initialize a basic specification with provenance data
          // The complete specification will be updated from worker-parsed metadata
          const initialSpec: ModelSpecification = {
            downloadDate: new Date().toISOString(),
            fileName: typeof source !== 'string' ? source.name : undefined,
            fileSize: modelData.byteLength,
            sourceURL: typeof source === 'string' ? source : undefined
          };

          // Pass modelFileName and modelContentType if available (from File object)
          const modelFileName = typeof source !== 'string' ? source.name : undefined;
          const modelContentType = typeof source !== 'string' ? source.type : undefined;
          
          try {
          // Store with initial specification - will be updated later with parsed data
          await this.modelCache.cacheModel(actualModelId, modelData, initialSpec, modelFileName, modelContentType);
          } catch (err) {
            // Log cache error but continue loading
            console.warn(new CacheError(
              `Failed to cache model: ${err instanceof Error ? err.message : String(err)}`,
              actualModelId
            ));
            // Don't throw - we can proceed even if caching fails
          }

          // Check for abort before sending to worker
          if (abortSignal.aborted) {
            handleAbort();
            return;
          }

          // Include a cancelable flag with the model data
          this.worker?.postMessage({
            event: workerActions.LOAD_MODEL_DATA,
            modelData: modelData,
            cancelable: true // Indicate this operation can be cancelled
          });
          // Again, promise resolves on INITIALIZED from worker
        } else {
            throw new FileError('Model data could not be retrieved');
        }
      } catch (err) {
        // Don't treat AbortError as an unexpected error
        if (err instanceof DOMException && err.name === 'AbortError') {
          handleAbort();
          return;
        }
        
        // Convert generic errors to specific types
        const specificError = err instanceof Error && !(err instanceof LocalWebAIError) 
          ? classifyError(err) 
          : err;
        
        console.error('Error loading model:', specificError);
        
        // Report error through progress callback
        this.reportProgress({
          stage: LoadingStage.ERROR,
          message: specificError instanceof Error ? specificError.message : String(specificError),
          error: specificError instanceof Error ? specificError.message : String(specificError)
        });
        
        this.isLoadingModel = false;
        this.onModelLoadedCallback = null;
        this.onModelLoadErrorCallback = null;
        this.currentProgressCallback = null;
        reject(specificError instanceof Error ? specificError : new Error(String(specificError)));
      } finally {
        // Clean up the abort event listener
        abortSignal.removeEventListener('abort', handleAbort);
        
        // Reset the abort controller if we created it
        if (this.abortController && !signal) {
          this.abortController = null;
        }
      }
    });
  }

  /**
   * Generate text based on a prompt with token-by-token streaming.
   * @param prompt The input prompt string.
   * @param params Optional parameters for text generation.
   * @param tokenCallback Callback for each generated token string.
   * @param completionCallback Callback for when generation is fully complete.
   * @throws ModelInitializationError if the model is not initialized
   */
  public generateText(
    prompt: string,
    params: GenerateTextParams = {},
    tokenCallback: TokenCallback,
    completionCallback: CompletionCallback
  ): void {
    if (!this.worker || !this.isInitialized) {
      throw new ModelInitializationError('LlamaRunner is not initialized or model not loaded.');
    }
    if (this.currentTokenCallback || this.currentCompletionCallback) {
        console.warn('Text generation already in progress. New request will be ignored or queued (not implemented yet).');
        // For POC, we might just throw an error or ignore
        throw new ModelInitializationError("Text generation already in progress.");
    }

    this.currentTokenCallback = tokenCallback;
    this.currentCompletionCallback = completionCallback;

    this.worker.postMessage({
      event: workerActions.RUN_MAIN,
      prompt: prompt,
      params: params,
    });
  }

  /**
   * Terminates the worker. The LlamaRunner instance should not be used after this.
   */
  public terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      this.isInitialized = false;
      this.isLoadingModel = false;
      // Clear callbacks
      this.onModelLoadedCallback = null;
      this.onModelLoadErrorCallback = null;
      this.currentProgressCallback = null;
      this.currentTokenCallback = null;
      this.currentCompletionCallback = null;
    }
  }
} 