import { ModelCache } from './model-cache.js';
import { ModelSpecification } from './model-spec.js'; // Import ModelSpecification

// Define the types for callbacks
export type ProgressCallback = (progress: number, total: number) => void;
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
};

// GGUF version constraints for validation
const MIN_SUPPORTED_GGUF_VERSION = 2;
const MAX_SUPPORTED_GGUF_VERSION = 3;

export class LlamaRunner {
  private worker: Worker | null = null;
  private modelCache: ModelCache;
  private isInitialized = false;
  private isLoadingModel = false;
  private onModelLoadedCallback: (() => void) | null = null;
  private onModelLoadErrorCallback: ((error: Error) => void) | null = null;
  private currentTokenCallback: TokenCallback | null = null;
  private currentCompletionCallback: CompletionCallback | null = null;
  private currentModelId: string | null = null; // Keep track of the current model ID for metadata
  private currentModelMetadata: ModelSpecification | null = null; // Store current model metadata

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
      const { event: action, text, error, metadata } = event.data;
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
        // Handle potential errors from worker
        case 'ERROR': // Assuming worker posts { event: 'ERROR', message: '...'}
            console.error('Error from worker:', error);
            if (this.isLoadingModel && this.onModelLoadErrorCallback) {
                this.onModelLoadErrorCallback(new Error(error || 'Unknown worker error during model load'));
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

      if (this.isLoadingModel && this.onModelLoadErrorCallback) {
        this.onModelLoadErrorCallback(new Error(detailedErrorMessage));
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
   * Validates and handles the model metadata received from the worker
   * @param metadata The parsed model metadata
   */
  private async handleModelMetadata(metadata: ModelSpecification): Promise<void> {
    // Store the metadata locally
    this.currentModelMetadata = metadata;

    // Validate the metadata
    const validationError = this.validateModelMetadata(metadata);
    if (validationError) {
      console.error('Model metadata validation failed:', validationError);
      if (this.isLoadingModel && this.onModelLoadErrorCallback) {
        this.onModelLoadErrorCallback(new Error(`Invalid model metadata: ${validationError}`));
        this.isLoadingModel = false;
        this.onModelLoadedCallback = null;
        this.onModelLoadErrorCallback = null;
      }
      return;
    }

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
   * @returns A string with the error message if validation fails, or null if successful
   */
  private validateModelMetadata(metadata: ModelSpecification): string | null {
    // Check GGUF version compatibility
    if (metadata.ggufVersion === undefined) {
      return 'Missing GGUF version information';
    }
    
    if (metadata.ggufVersion < MIN_SUPPORTED_GGUF_VERSION || 
        metadata.ggufVersion > MAX_SUPPORTED_GGUF_VERSION) {
      return `Unsupported GGUF version: ${metadata.ggufVersion}. Supported versions: ${MIN_SUPPORTED_GGUF_VERSION}-${MAX_SUPPORTED_GGUF_VERSION}`;
    }
    
    // Other potential validations:
    // 1. Check if critical fields are present (architecture, context length, etc.)
    if (!metadata.architecture) {
      console.warn('Model metadata is missing architecture information');
      // Not fatal, but worth warning about
    }
    
    // 2. Validate context length if present (e.g., must be a reasonable value)
    if (metadata.contextLength !== undefined) {
      if (metadata.contextLength < 1 || metadata.contextLength > 32768) {
        return `Invalid context length: ${metadata.contextLength}. Expected a value between 1 and 32768`;
      }
    }
    
    // All validations passed
    return null;
  }

  /**
   * Retrieves the current model's metadata
   * @returns The current model's metadata or null if no model is loaded
   */
  public getModelMetadata(): ModelSpecification | null {
    return this.currentModelMetadata;
  }

  /**
   * Load a GGUF model from a URL or File object.
   * @param source URL string or File object for the GGUF model.
   * @param modelId A unique ID for caching. If not provided, URL or filename+size will be used.
   * @param progressCallback Optional callback for download/file reading progress.
   * @returns Promise<void> Resolves when the model is loaded and ready for inference.
   */
  public async loadModel(
    source: string | File,
    modelId?: string,
    progressCallback?: ProgressCallback
  ): Promise<void> {
    if (!this.worker) {
      throw new Error('Worker not initialized.');
    }
    if (this.isLoadingModel) {
        throw new Error('Another model is already being loaded.');
    }
    this.isLoadingModel = true;

    return new Promise(async (resolve, reject) => {
      this.onModelLoadedCallback = resolve;
      this.onModelLoadErrorCallback = reject;

      const actualModelId = modelId || (typeof source === 'string' ? source : `${source.name}-${source.size}`);
      this.currentModelId = actualModelId; // Store current model ID for metadata handling
      this.currentModelMetadata = null; // Reset metadata for new model
      
      let modelData: ArrayBuffer | null = null;
      let cachedModelInfo = null;

      // 1. Try fetching from cache
      try {
        cachedModelInfo = await this.modelCache.getModelWithSpecificationFromCache(actualModelId);
        modelData = cachedModelInfo?.modelData || null;
        
        // If cache has metadata, store it immediately
        if (cachedModelInfo?.specification) {
          this.currentModelMetadata = cachedModelInfo.specification;
        }
      } catch (err) {
        console.warn('Error retrieving from cache, will load from source:', err);
        modelData = null;
      }

      if (modelData) {
        if (progressCallback) progressCallback(modelData.byteLength, modelData.byteLength); // Cached, so 100%
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
          const response = await fetch(source);
          if (!response.ok) throw new Error(`Failed to download model: ${response.statusText}`);
          if (!response.body) throw new Error('Response body is null');

          const contentLength = Number(response.headers.get('Content-Length') || '0');
          const reader = response.body.getReader();
          const chunks: Uint8Array[] = [];
          let receivedLength = 0;

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            receivedLength += value.length;
            if (progressCallback) progressCallback(receivedLength, contentLength);
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
          modelData = await new Promise<ArrayBuffer>((resolveFile, rejectFile) => {
            const reader = new FileReader();
            reader.onload = (e) => resolveFile(e.target?.result as ArrayBuffer);
            reader.onerror = (e) => rejectFile(reader.error || new Error('File reading error'));
            reader.onprogress = (e) => {
              if (e.lengthComputable && progressCallback) {
                progressCallback(e.loaded, e.total);
              }
            };
            reader.readAsArrayBuffer(source);
          });
        }

        if (modelData) {
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
          
          // Store with initial specification - will be updated later with parsed data
          await this.modelCache.cacheModel(actualModelId, modelData, initialSpec, modelFileName, modelContentType);

          this.worker?.postMessage({
            event: workerActions.LOAD_MODEL_DATA,
            modelData: modelData,
          });
          // Again, promise resolves on INITIALIZED from worker
        } else {
            throw new Error('Model data could not be retrieved.');
        }
      } catch (err) {
        console.error('Error loading model:', err);
        this.isLoadingModel = false;
        this.onModelLoadedCallback = null;
        this.onModelLoadErrorCallback = null;
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  /**
   * Generate text based on a prompt with token-by-token streaming.
   * @param prompt The input prompt string.
   * @param params Optional parameters for text generation.
   * @param tokenCallback Callback for each generated token string.
   * @param completionCallback Callback for when generation is fully complete.
   */
  public generateText(
    prompt: string,
    params: GenerateTextParams = {},
    tokenCallback: TokenCallback,
    completionCallback: CompletionCallback
  ): void {
    if (!this.worker || !this.isInitialized) {
      throw new Error('LlamaRunner is not initialized or model not loaded.');
    }
    if (this.currentTokenCallback || this.currentCompletionCallback) {
        console.warn('Text generation already in progress. New request will be ignored or queued (not implemented yet).');
        // For POC, we might just throw an error or ignore
        throw new Error("Text generation already in progress.");
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
      this.currentTokenCallback = null;
      this.currentCompletionCallback = null;
    }
  }
} 