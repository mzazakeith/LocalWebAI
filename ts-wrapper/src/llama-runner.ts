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

// Import Wllama types (adjust path as necessary if wllama is not a sibling)
import { Wllama, WllamaConfig, AssetsPathConfig, LoadModelConfig, ModelMetadata as WllamaModelMetadata, LoggerWithoutDebug, WllamaError, WllamaAbortError, SamplingConfig, ChatCompletionOptions, CompletionChunk } from '../../wllama/esm/index.js';

// Import bufToText utility or define it locally
// Assuming bufToText is re-exported or we define a simple version:
const simpleBufToText = (buf: Uint8Array) => new TextDecoder().decode(buf);

// Define a local interface compatible with wllama's DownloadOptions for progress and abort
interface WllamaDownloadOptions {
  progress_callback?: (loaded: number, total: number) => void;
  abort_signal?: AbortSignal;
  useCache?: boolean; // This option is often paired in wllama's loadModelFromUrl
  // headers?: Record<string, string>; // Not currently used by LlamaRunner
}

// Define the types for callbacks
export type TokenCallback = (token: string) => void;
export type CompletionCallback = () => void;

// Define the structure for generation parameters, aligning with llama.cpp options
export interface GenerateTextParams {
  n_predict?: number;       // Max tokens to predict. -1 for infinity, -2 for till context limit.
  // ctx_size?: number; // This will likely be set during wllama.loadModel via LoadModelConfig.n_ctx
  // batch_size?: number; // This will likely be set during wllama.loadModel via LoadModelConfig.n_batch
  temp?: number;            // Temperature for sampling.
  // n_gpu_layers?: number; // Likely not applicable for browser WASM
  top_k?: number;           // Top-K sampling.
  top_p?: number;           // Top-P (nucleus) sampling.
  // no_display_prompt?: boolean; // Will be handled during generation if wllama supports it or by filtering output
  // chatml?: boolean; // Will be handled by prompt formatting or if wllama has a specific mode
  // ---- New Wllama specific params that might be useful to expose ----
  mirostat?: number;
  mirostat_tau?: number;
  penalty_last_n?: number;
  penalty_repeat?: number;
  penalty_freq?: number;
  penalty_present?: number;
  grammar?: string; // GBNF grammar
}

// Configuration for Wllama artifact paths
export interface WllamaArtifacts {
  singleThreadWasm: string; // Path to wllama.wasm (single-thread)
  multiThreadWasm?: string; // Path to wllama-mt.wasm (multi-thread)
  // workerJs?: string; // Path to wllama.worker.js (if using multi-thread worker pattern)
                        // wllama seems to handle its own worker loading if multiThreadWasm is given
}

// GGUF version constraints for validation
const MIN_SUPPORTED_GGUF_VERSION = 2;
const MAX_SUPPORTED_GGUF_VERSION = 3;

// Required model metadata fields
const REQUIRED_METADATA_FIELDS = [
  // Critical fields that must be present for compatibility verification
  'ggufVersion'
  // 'architecture' // wllama might provide this differently, to be checked in mapping
];

// Optional but important fields (we'll warn if missing)
const IMPORTANT_METADATA_FIELDS = [
  'architecture',
  'contextLength',
  'embeddingLength'
];

export class LlamaRunner {
  // private worker: Worker | null = null; // REMOVED
  private wllamaInstance: Wllama | null = null; // ADDED
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
   * @param wllamaArtifactPaths Paths to the wllama WASM artifacts.
   * @param wllamaOptions Optional configuration for Wllama.
   */
  constructor(
    private wllamaArtifactPaths: WllamaArtifacts,
    private wllamaOptions?: WllamaConfig
  ) {
    this.modelCache = new ModelCache();
    // this.initWorker(); // REMOVED old initWorker
    this.initWllama(); // ADDED
  }

  // private initWorker(): void { // REMOVED old initWorker
    // ... old worker initialization logic ...
  // }

  private initWllama(): void { // ADDED
    try {
      const paths: AssetsPathConfig = {
        'single-thread/wllama.wasm': new URL(this.wllamaArtifactPaths.singleThreadWasm, window.location.href).href,
      };
      if (this.wllamaArtifactPaths.multiThreadWasm) {
        paths['multi-thread/wllama.wasm'] = new URL(this.wllamaArtifactPaths.multiThreadWasm, window.location.href).href;
      }

      const config: WllamaConfig = {
        logger: LoggerWithoutDebug, // Default to less verbose logging
        suppressNativeLog: true,
        ...(this.wllamaOptions || {}),
      };

      this.wllamaInstance = new Wllama(paths, config);
      this.isInitialized = true; // Assume synchronous initialization of Wllama class itself is success
      console.log("Wllama instance created successfully.");

      // If there was an onModelLoadedCallback queued from a previous attempt (unlikely here, but good practice)
      // This logic might shift depending on how loadModel handles pre-init loading.
      // For now, Wllama constructor is synchronous.
      if (this.isLoadingModel && this.onModelLoadedCallback) {
        // This state seems more relevant after a model load attempt, not Wllama instantiation.
        // For now, keeping isInitialized tied to Wllama instantiation.
      }

    } catch (error) {
      console.error('Error initializing Wllama:', error);
      this.isInitialized = false;
      // Report this critical initialization error
      const wllamaError = error instanceof WllamaError ? error : new WasmError(`Wllama initialization failed: ${error instanceof Error ? error.message : String(error)}`);
      
      if (this.currentProgressCallback) {
        this.currentProgressCallback({
          stage: LoadingStage.ERROR,
          message: wllamaError.message,
          error: wllamaError.message
        });
      }
      // This error should probably be propagated more forcefully, e.g., by throwing.
      // For now, LlamaRunner will be in a non-functional state.
      // Consider if the constructor should throw or if methods should check isInitialized.
      throw wllamaError; // Propagate critical initialization error
    }
  }

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
   * Merges worker-parsed metadata with existing provenance information.
   * 
   * @param workerMetadata The parsed model metadata received from the worker
   */
  private async handleModelMetadata(workerMetadata: ModelSpecification): Promise<void> {
    // Start with the existing metadata (which should have provenance)
    // Or initialize a new object if none exists yet
    const mergedMetadata: ModelSpecification = {
        ...(this.currentModelMetadata || {}), // Keep existing fields (like provenance)
        ...workerMetadata // Overwrite with worker-parsed fields, worker data takes precedence for GGUF fields
    };
    
    // Store the merged metadata locally
    this.currentModelMetadata = mergedMetadata;

    // Validate the merged metadata
    const validationResult = this.validateModelMetadata(mergedMetadata);
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

    // Report that we have the merged metadata
    this.reportProgress({
      metadata: mergedMetadata, // Report with the merged data
      message: 'Model metadata parsed successfully'
    });

    // If we have a current modelId, update the model's metadata in the cache
    // with the complete merged information
    if (this.currentModelId) {
      try {
        // Retrieve the model data from cache (we only need existence check here technically)
        const cachedModelData = await this.modelCache.getModelFromCache(this.currentModelId);
        
        if (cachedModelData) {
          // Ensure provenance is correctly set in the merged data before caching
          if (typeof this.currentModelId === 'string' && this.currentModelId.startsWith('http') && !mergedMetadata.sourceURL) {
            mergedMetadata.sourceURL = this.currentModelId;
          }
          if (!mergedMetadata.downloadDate) {
             mergedMetadata.downloadDate = new Date().toISOString();
          }
          
          // Update the cache with the complete merged metadata
          await this.modelCache.cacheModel(this.currentModelId, cachedModelData, mergedMetadata);
          console.log('Updated model cache with merged metadata for:', this.currentModelId);
          
          // MEMORY OPTIMIZATION: Release reference to cached data buffer if no longer needed
          // Note: This might not be necessary if getModelFromCache doesn't hold onto it,
          // but explicit cleanup can be safer depending on cache implementation.
          // cachedModelData = null; 
        }
      } catch (err) {
        console.warn('Failed to update model cache with merged metadata:', err);
        // Non-fatal error, continue with model load
      }
    }
    
    // --- UI Update Trigger --- 
    // We can trigger the UI update here *after* merging and validation
    // This ensures the UI always shows the most complete picture.
    // Note: The progress callback might have already updated the UI if called earlier,
    // but calling displayModelMetadata directly ensures it uses the final merged data.
    // Example (if displayModelMetadata was globally accessible or passed in):
    // displayModelMetadata(mergedMetadata);
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

    // NEW: If wllamaInstance exists and has a way to cancel ongoing ops, call it.
    // Wllama's loadModel and runCompletion take AbortSignal directly.
    // This external abortController is for LlamaRunner's own management.
    // Wllama will react to the signal passed to its methods.

    // REMOVED: Old worker cancellation
    // if (this.worker) {
    //   this.worker.postMessage({
    //     event: workerActions.CANCEL_LOAD
    //   });
    // }

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
    signal?: AbortSignal,
    loadConfig?: Omit<LoadModelConfig, 'progress_callback'> // progress_callback will be handled internally
  ): Promise<void> {
    if (!this.wllamaInstance || !this.isInitialized) {
      throw new WasmError('LlamaRunner is not initialized or Wllama instance is not available.');
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
        console.log('Model loading aborted by user (LlamaRunner)');

        // REMOVED: Old worker cancellation
        // if (this.worker) {
        //   this.worker.postMessage({
        //     event: workerActions.CANCEL_LOAD
        //   });
        // }

        // The AbortSignal passed to wllama.loadModel will handle wllama's internal cancellation.
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

      if (abortSignal.aborted) {
        this.isLoadingModel = false;
        this.onModelLoadedCallback = null;
        this.onModelLoadErrorCallback = null;
        const abortError = new OperationCancelledError('Model loading aborted by user prior to start');
        this.reportProgress({ stage: LoadingStage.CANCELLED, message: abortError.message, error: abortError.message });
        reject(abortError);
        return;
      }

      const actualModelId = modelId || (typeof source === 'string' ? source : `${source.name}-${source.size}`);
      this.currentModelId = actualModelId;
      this.currentModelMetadata = null;

      let modelDataFromCache: ArrayBuffer | null = null;
      let cachedSpec: ModelSpecification | undefined;

      try {
        this.reportProgress({
          stage: LoadingStage.PREPARING_MODEL_DATA,
          message: 'Checking model cache'
        });
        const cachedResult = await this.modelCache.getModelWithSpecificationFromCache(actualModelId);
        if (cachedResult) {
            modelDataFromCache = cachedResult.modelData;
            cachedSpec = cachedResult.specification;
            if (cachedSpec) {
                this.currentModelMetadata = cachedSpec;
                this.reportProgress({
                    stage: LoadingStage.METADATA_PARSE_COMPLETE, // Or a new "METADATA_FROM_CACHE"
                    message: 'Model metadata loaded from cache',
                    metadata: cachedSpec
                });
            }
        }
      } catch (err) {
        const cacheError = new CacheError(
          `Error retrieving from cache: ${err instanceof Error ? err.message : String(err)}`,
          actualModelId
        );
        console.warn(cacheError);
        modelDataFromCache = null;
      }

      if (abortSignal.aborted) {
        handleAbort();
        return;
      }

      console.log(`Preparing to load model ${actualModelId} into Wllama.`);
      try {
        let sourceInfo: { url?: string, fileName?: string, fileSize?: number } = {};

        // Ensure wllamaInstance is available
        if (!this.wllamaInstance) {
          throw new WasmError('Wllama instance is not available for loading model.');
        }

        this.reportProgress({ stage: LoadingStage.MODEL_INITIALIZATION_START, message: 'Loading model with Wllama...' });

        if (typeof source === 'string') { // Handle URL source
          sourceInfo.url = source;
          this.reportProgress({
            stage: LoadingStage.MODEL_FETCH_START,
            message: 'Fetching model from URL via Wllama'
          });

          const downloadAndLoadConfig: LoadModelConfig & WllamaDownloadOptions = {
            ...(loadConfig || {}),
            progress_callback: (loaded: number, total: number) => {
              this.reportProgress({
                stage: LoadingStage.MODEL_FETCH_PROGRESS,
                loaded,
                total,
                message: `Wllama downloading: ${total > 0 ? Math.round((loaded / total) * 100) : 0}%`
              });
            },
            // Pass AbortSignal for download if wllama supports it in DownloadOptions (as abort_signal)
            // Based on wllama.d.ts, DownloadOptions takes `abort_signal`
            abort_signal: abortSignal // Pass the signal here
          };

          await this.wllamaInstance.loadModelFromUrl(source, downloadAndLoadConfig);
          // fileSize might be known after download, wllama doesn't explicitly return it here.
          // It will be part of metadata if GGUF contains it.

        } else { // Handle File or cached ArrayBuffer source
          let modelBlob: Blob;
          if (modelDataFromCache) { // Data from LlamaRunner's cache
            this.reportProgress({
              stage: LoadingStage.PREPARING_MODEL_DATA,
              message: 'Using cached model data for Wllama',
              loaded: modelDataFromCache.byteLength,
              total: modelDataFromCache.byteLength
            });
            modelBlob = new Blob([modelDataFromCache], { type: 'application/octet-stream' });
            sourceInfo.fileName = cachedSpec?.fileName || 'cached_model.gguf';
            sourceInfo.fileSize = modelDataFromCache.byteLength;
            sourceInfo.url = cachedSpec?.sourceURL;
          } else { // Source is a File object
            this.reportProgress({
              stage: LoadingStage.READING_FROM_FILE,
              message: 'Reading model from file for Wllama'
            });
            // For File objects, we don't need to read it into ArrayBuffer ourselves if wllama.loadModel takes Blob.
            // wllama.loadModel expects Blob[] or Model. So a File (which is a Blob) is fine in an array.
            modelBlob = source as File; 
            sourceInfo.fileName = (source as File).name;
            sourceInfo.fileSize = (source as File).size;
            
            // Report file read complete (conceptual, as wllama takes the blob directly)
            this.reportProgress({
                stage: LoadingStage.PREPARING_MODEL_DATA,
                loaded: sourceInfo.fileSize,
                total: sourceInfo.fileSize,
                message: 'File prepared for Wllama.'
            });
          }

          if (abortSignal.aborted) {
            handleAbort();
            return;
          }
          
          // For loadModel(Blob[]), progress is not directly applicable for the load itself.
          // The LoadModelConfig doesn't take a progress_callback here.
          await this.wllamaInstance.loadModel([modelBlob], loadConfig || {});
        }

        if (abortSignal.aborted) {
          handleAbort();
          return;
        }

        // --- Metadata Handling ---
        const wllamaInternalMeta = this.wllamaInstance.getModelMetadata();
        if (wllamaInternalMeta) {
          this.currentModelMetadata = this.mapWllamaMetaToModelSpec(wllamaInternalMeta, sourceInfo);

          const validation = this.validateModelMetadata(this.currentModelMetadata);
          if (validation.error) {
            const errorInstance = validation.errorInstance || new ModelCompatibilityError(validation.error);
            this.reportProgress({ stage: LoadingStage.ERROR, message: errorInstance.message, error: errorInstance.message, metadata: this.currentModelMetadata });
            throw errorInstance;
          }
          if (validation.warnings.length > 0) {
            validation.warnings.forEach(w => console.warn("Metadata warning:", w));
          }

          this.reportProgress({ stage: LoadingStage.METADATA_PARSE_COMPLETE, metadata: this.currentModelMetadata, message: "Model metadata processed via Wllama." });

          // Cache the model if it was loaded from a File and not from LlamaRunner's cache initially
          if (source instanceof File && !modelDataFromCache) {
            try {
                // We need ArrayBuffer to cache in ModelCache. Read the file again or use the blob.
                const fileBufferForCache = await (source as File).arrayBuffer();
                await this.modelCache.cacheModel(actualModelId, fileBufferForCache, this.currentModelMetadata);
                this.reportProgress({ stage: LoadingStage.MODEL_FETCH_COMPLETE, message: "Model (from file) cached successfully."});
            } catch(cacheErr) {
                console.warn(new CacheError(`Failed to cache model ${actualModelId} from file: ${cacheErr instanceof Error ? cacheErr.message : String(cacheErr)}`));
                this.reportProgress({ stage: LoadingStage.MODEL_FETCH_COMPLETE, message: "Model loaded, caching (from file) failed (non-critical)."});
            }
          }
        } else {
          this.reportProgress({ stage: LoadingStage.ERROR, message: 'Wllama loaded model but getModelMetadata() returned null/undefined.' });
          throw new ModelInitializationError('Wllama loaded model but getModelMetadata() returned null/undefined.');
        }

        this.isLoadingModel = false;
        this.onModelLoadedCallback = null;
        this.onModelLoadErrorCallback = null;
        this.currentProgressCallback = null;

        this.reportProgress({ stage: LoadingStage.MODEL_READY, metadata: this.currentModelMetadata, message: 'Model ready via Wllama.' });
        resolve();

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
   * @param abortSignal Optional AbortSignal for this specific generation call
   * @throws ModelInitializationError if the model is not initialized
   */
  public async generateText(
    prompt: string,
    params: GenerateTextParams = {},
    tokenCallback: TokenCallback,
    completionCallback: CompletionCallback,
    abortSignal?: AbortSignal
  ): Promise<void> {
    if (!this.wllamaInstance || !this.isInitialized || !this.currentModelMetadata) {
      const error = new ModelInitializationError('LlamaRunner is not initialized, Wllama instance is not available, or no model is loaded.');
      // Immediately call completion callback with error indication if possible, or throw
      // For now, we'll throw, and the caller should catch.
      // Consider how to report this error to the completionCallback or a new errorCallback for generateText.
      this.currentCompletionCallback = null; // Clear completion as it won't be normally reached
      this.currentTokenCallback = null;
      throw error;
    }
    if (this.currentTokenCallback || this.currentCompletionCallback) {
      const error = new ModelInitializationError("Text generation already in progress.");
      // Similar to above, decide on error reporting. Throwing for now.
      throw error;
    }

    this.currentTokenCallback = tokenCallback;
    this.currentCompletionCallback = completionCallback;

    // 1. Create SamplingConfig
    const samplingConfig: SamplingConfig = {
      temp: params.temp,
      top_k: params.top_k,
      top_p: params.top_p,
      mirostat: params.mirostat,
      mirostat_tau: params.mirostat_tau,
      penalty_last_n: params.penalty_last_n,
      penalty_repeat: params.penalty_repeat,
      penalty_freq: params.penalty_freq,
      penalty_present: params.penalty_present,
      grammar: params.grammar
      // Other params like dynatemp_range, n_prev, n_probs, min_p, typical_p, logit_bias can be added if exposed
    };

    // 2. Create ChatCompletionOptions
    const chatOptions: ChatCompletionOptions & { stream: true } = { // Explicitly type stream as true
      stream: true,
      nPredict: params.n_predict, // Max tokens to predict
      sampling: samplingConfig,
      abortSignal: abortSignal, // Pass the AbortSignal for this generation
      // onNewToken: This will be handled by iterating the async iterable
      // stopTokens: Needs mapping if GenerateTextParams includes it
      // useCache: Could be a new param in GenerateTextParams if prompt caching is desired
    };

    try {
      if (!this.wllamaInstance) { // Should be caught by initial check, but good for type safety
        throw new ModelInitializationError('Wllama instance became null unexpectedly.');
      }

      const completionStream = await this.wllamaInstance.createCompletion(prompt, chatOptions);

      for await (const chunk of completionStream) {
        if (abortSignal?.aborted) { // Check for abort during streaming
          console.log('Text generation aborted during streaming.');
          throw new WllamaAbortError(); // WllamaAbortError or OperationCancelledError
        }
        if (this.currentTokenCallback) {
          // piece is Uint8Array. tokenCallback expects string.
          this.currentTokenCallback(simpleBufToText(chunk.piece));
        }
      }

      // Generation completed successfully (finished stream or stopped by nPredict)
      if (this.currentCompletionCallback) {
        this.currentCompletionCallback();
      }

    } catch (error) {
      console.error('Error during Wllama text generation:', error);
      let finalError: LocalWebAIError;
      if (error instanceof WllamaAbortError || (error instanceof DOMException && error.name === 'AbortError')) {
        finalError = new OperationCancelledError('Text generation cancelled.');
      } else if (error instanceof WllamaError) {
        // Map WllamaError to a more generic ModelInitializationError or a new InferenceError
        finalError = new ModelInitializationError(`Wllama inference error: ${error.message} (type: ${error.type})`);
      } else if (error instanceof LocalWebAIError) {
        finalError = error;
      } else {
        finalError = new ModelInitializationError(`Unexpected error during text generation: ${error instanceof Error ? error.message : String(error)}`);
      }
      
      // If a completionCallback exists, we should call it, perhaps with an error marker or let the throw propagate.
      // For now, letting the error propagate to the caller of generateText.
      // The UI/caller will need a try/catch around generateText.
      if (this.currentCompletionCallback) {
        // Indicate error to completion callback? Or rely on throw.
        // For now, let's assume the throw is sufficient and the caller handles it.
        // If not, the completionCallback might need an optional error argument.
         this.currentCompletionCallback(); // Call it to signify the end, even if errored.
      }
      throw finalError; // Propagate the classified error
    } finally {
      // Clear callbacks once generation is finished or errored
      this.currentTokenCallback = null;
      this.currentCompletionCallback = null;
    }
  }

  /**
   * Terminates the Wllama instance and cleans up resources.
   */
  public terminate(): void {
    // if (this.worker) { // REMOVED
    //   this.worker.terminate();
    //   this.worker = null;
    // }
    if (this.wllamaInstance) {
      try {
        this.wllamaInstance.exit(); // Assuming wllama has an exit method
        console.log("Wllama instance exited.");
      } catch (e) {
        console.error("Error during wllamaInstance.exit():", e);
      }
      this.wllamaInstance = null;
    }
    // ... existing code ...
  }

  // +++ ADDED: Helper method to map Wllama metadata to ModelSpecification +++
  private mapWllamaMetaToModelSpec(wllamaMeta: WllamaModelMetadata, sourceInfo: { url?: string, fileName?: string, fileSize?: number }): ModelSpecification {
    const spec: ModelSpecification = {
        // Provenance from sourceInfo
        sourceURL: sourceInfo.url,
        fileName: sourceInfo.fileName,
        fileSize: sourceInfo.fileSize,
        downloadDate: new Date().toISOString(),

        // From wllamaMeta.hparams
        vocabSize: wllamaMeta.hparams?.nVocab,
        contextLength: wllamaMeta.hparams?.nCtxTrain,
        embeddingLength: wllamaMeta.hparams?.nEmbd,
        layerCount: wllamaMeta.hparams?.nLayer,

        // From wllamaMeta.meta (Record<string, string>)
        architecture: wllamaMeta.meta?.['general.architecture'],
        modelName: wllamaMeta.meta?.['general.name'],
        ggufVersion: wllamaMeta.meta?.['general.version'] ? parseInt(wllamaMeta.meta['general.version'], 10) :
                       (wllamaMeta.meta?.['gguf.version'] ? parseInt(wllamaMeta.meta['gguf.version'], 10) : undefined),
        quantization: wllamaMeta.meta?.['general.file_type'], // Keep this simple for now
        headCount: wllamaMeta.meta?.['llama.attention.head_count'] ? parseInt(wllamaMeta.meta['llama.attention.head_count'], 10) : undefined,
        headCountKv: wllamaMeta.meta?.['llama.attention.head_count_kv'] ? parseInt(wllamaMeta.meta['llama.attention.head_count_kv'], 10) : undefined,
        ropeFrequencyBase: wllamaMeta.meta?.['llama.rope.freq_base'] ? parseFloat(wllamaMeta.meta['llama.rope.freq_base']) : undefined,
        ropeFrequencyScale: wllamaMeta.meta?.['llama.rope.scale_linear'] ? parseFloat(wllamaMeta.meta['llama.rope.scale_linear']) : undefined,
        creator: wllamaMeta.meta?.['general.author'],
        license: wllamaMeta.meta?.['general.license'],
    };

    // Add all other string meta values from wllamaMeta.meta
    if (wllamaMeta.meta) {
        for (const key in wllamaMeta.meta) {
            const camelKey = keyToCamelCase(key);
            // Avoid overwriting common fields already mapped or if camelKey is not a valid spec key yet
            if (!spec.hasOwnProperty(camelKey) || spec[camelKey] === undefined) { 
                 const valueStr = wllamaMeta.meta[key];
                 const numVal = parseFloat(valueStr);
                 if (!isNaN(numVal) && String(numVal) === valueStr) {
                    spec[camelKey] = numVal;
                 } else if (valueStr.toLowerCase() === 'true') {
                    spec[camelKey] = true;
                 } else if (valueStr.toLowerCase() === 'false') {
                    spec[camelKey] = false;
                 } else {
                    spec[camelKey] = valueStr;
                 }
            }
        }
    }

    // Refinement for ggufVersion if it was not found with common keys
    if (spec.ggufVersion === undefined && wllamaMeta.meta) {
        const versionKey = Object.keys(wllamaMeta.meta).find(k => k.toLowerCase().includes('gguf') && k.toLowerCase().includes('version'));
        if (versionKey && wllamaMeta.meta[versionKey]) {
            const parsedVersion = parseInt(wllamaMeta.meta[versionKey], 10);
            if (!isNaN(parsedVersion)) {
                spec.ggufVersion = parsedVersion;
            }
        }
    }
    
    // Attempt to get a more specific quantization string if general.file_type is too generic (e.g., "all F32")
    // This is still a simplification.
    if (wllamaMeta.meta && (spec.quantization === undefined || spec.quantization.toLowerCase().includes('f32') || spec.quantization.toLowerCase().includes('f16') && !spec.quantization.toLowerCase().includes('q'))) {
        const quantKey = Object.keys(wllamaMeta.meta).find(k => k.endsWith('.quantization_type') && !k.startsWith('general.'));
        if (quantKey && wllamaMeta.meta[quantKey]) {
            spec.quantization = wllamaMeta.meta[quantKey];
        }
    }

    return spec;
  }
}

// Helper function (can be moved outside or to a utils file)
function keyToCamelCase(key: string): string {
    return key.replace(/[^a-zA-Z0-9]+(.)/g, (m, chr) => chr.toUpperCase());
} 