import { ModelCache } from './model-cache.js';

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
};

// const MAX_CACHEABLE_SIZE = 1 * 1024 * 1024 * 1024; // 1 GiB -- This is no longer needed as ModelCache handles chunking.

export class LlamaRunner {
  private worker: Worker | null = null;
  private modelCache: ModelCache;
  private isInitialized = false;
  private isLoadingModel = false;
  private onModelLoadedCallback: (() => void) | null = null;
  private onModelLoadErrorCallback: ((error: Error) => void) | null = null;
  private currentTokenCallback: TokenCallback | null = null;
  private currentCompletionCallback: CompletionCallback | null = null;

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
      const { event: action, text, error } = event.data;
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
      let modelData: ArrayBuffer | null = null;

      // 1. Try fetching from cache
      modelData = await this.modelCache.getModelFromCache(actualModelId);
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
          // The new ModelCache handles chunking, so the explicit size check is removed.
          // if (modelData.byteLength < MAX_CACHEABLE_SIZE) { // Old check
          //   await this.modelCache.cacheModel(actualModelId, modelData);
          // } else {
          //   console.warn(
          //     `Model ${actualModelId} is too large to cache (${(modelData.byteLength / (1024*1024)).toFixed(2)} MB). Max cacheable size: ${(MAX_CACHEABLE_SIZE / (1024*1024)).toFixed(2)} MB. Skipping cache.`
          //   );
          // }

          // Pass modelFileName and modelContentType if available (from File object)
          const modelFileName = typeof source !== 'string' ? source.name : undefined;
          const modelContentType = typeof source !== 'string' ? source.type : undefined;
          // Pass undefined for the specification for now, it will be populated in a later phase.
          // The existing originalFileName and modelContentType can still be passed for cacheEntryMeta.
          await this.modelCache.cacheModel(actualModelId, modelData, undefined, modelFileName, modelContentType);

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