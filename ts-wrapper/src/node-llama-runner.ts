import { Worker } from 'worker_threads';
import path from 'path'; // Use path module for robust path handling
import { ModelSpecification } from './model-spec.js';
import { ProgressCallback, ProgressInfo, LoadingStage, getStageDescription } from './loading-progress.js';
import { 
  NetworkError, 
  FileError, 
  GGUFParsingError, 
  ModelCompatibilityError, 
  CacheError, // Keep for potential future use? Or remove if strictly no caching
  VFSError,
  WasmError,
  OperationCancelledError,
  LocalWebAIError,
  ModelInitializationError,
  classifyError
} from './errors.js';

// Define the types for callbacks (same as browser version)
export type TokenCallback = (token: string) => void;
export type CompletionCallback = () => void;

// Define the structure for generation parameters (same as browser version)
export interface GenerateTextParams {
  n_predict?: number;
  ctx_size?: number;
  batch_size?: number;
  temp?: number;
  n_gpu_layers?: number; // Might not be applicable in typical Node Wasm, but keep for consistency
  top_k?: number;
  top_p?: number;
  no_display_prompt?: boolean;
  chatml?: boolean;
  // Add other llama.cpp params as needed
}

// Worker actions need to match node-worker.ts
const workerActions = {
  LOAD: 'LOAD', // May not be used directly by Node runner if init happens via LOAD_NODE
  INITIALIZED: 'INITIALIZED',
  RUN_MAIN: 'RUN_MAIN',
  WRITE_RESULT: 'WRITE_RESULT',
  RUN_COMPLETED: 'RUN_COMPLETED',
  LOAD_MODEL_DATA: 'LOAD_MODEL_DATA',
  MODEL_METADATA: 'MODEL_METADATA',
  PROGRESS_UPDATE: 'PROGRESS_UPDATE',
  CANCEL_LOAD: 'CANCEL_LOAD',
  LOAD_NODE: 'LOAD_NODE' // Node-specific init action
};

export class NodeLlamaRunner {
  private worker: Worker | null = null;
  private isInitialized = false; // Tracks if Wasm module in worker is ready
  private isModelLoaded = false; // Tracks if a model is loaded in the worker VFS
  private isLoadingModel = false;
  private onModelLoadedCallback: (() => void) | null = null;
  private onModelLoadErrorCallback: ((error: Error) => void) | null = null;
  private currentProgressCallback: ProgressCallback | null = null;
  private currentTokenCallback: TokenCallback | null = null;
  private currentCompletionCallback: CompletionCallback | null = null;
  private currentModelMetadata: ModelSpecification | null = null; // Store current model metadata
  private lastProgressInfo: ProgressInfo | null = null; // Track the last progress update
  private abortController: AbortController | null = null; // For tracking active load cancellation

  // Promise to track Wasm module readiness in the worker
  private wasmModuleReadyPromise: Promise<void>;
  private resolveWasmModuleReady: (() => void) | null = null;
  private rejectWasmModuleReady: ((reason?: any) => void) | null = null;

  /**
   * @param workerPath Absolute path to the compiled node-worker.ts (e.g., node-worker.js)
   * @param wasmNodeModulePath Absolute path to the Node.js-specific Emscripten JS glue file (e.g., from llama-cpp-wasm/dist/node/.../main.js)
   * @param wasmNodePath Absolute path to the Node.js-specific .wasm file (e.g., from llama-cpp-wasm/dist/node/.../main.wasm)
   */
  constructor(
    private workerPath: string,
    private wasmNodeModulePath: string,
    private wasmNodePath: string
  ) {
    this.wasmModuleReadyPromise = new Promise<void>((resolve, reject) => {
      this.resolveWasmModuleReady = resolve;
      this.rejectWasmModuleReady = reject; // Store rejector for error cases during init
    });
    this.initWorker();
  }

  private initWorker(): void {
    try {
      console.log(`[NodeLlamaRunner] Initializing worker from path: ${this.workerPath}`);
      // Ensure paths are absolute before passing to worker
      // Worker needs paths relative to its own location or absolute paths
      // For simplicity, let's assume the paths passed are resolvable by the worker context
      // or we make them absolute here.
      const absoluteWorkerPath = path.resolve(this.workerPath);
      
      // Worker data is less common for initial setup; sending a message is often cleaner.
      this.worker = new Worker(absoluteWorkerPath);
      
      this.worker.on('message', (data) => {
        this.handleWorkerMessage(data);
      });

      this.worker.on('error', (error) => {
        console.error('[NodeLlamaRunner] Worker errored:', error);
        const workerError = new WasmError(`Worker thread encountered an error: ${error.message}`);
        // If Wasm module initialization fails, reject the promise
        if (this.rejectWasmModuleReady) {
            this.rejectWasmModuleReady(workerError);
        }
        this.handleWorkerError(workerError);
      });

      this.worker.on('exit', (code) => {
        console.log(`[NodeLlamaRunner] Worker exited with code: ${code}`);
        this.isInitialized = false;
        this.isModelLoaded = false;
        this.isLoadingModel = false;
        this.worker = null; // Worker is gone
        // Potentially notify the user or attempt restart if appropriate
        if (code !== 0) {
             const exitError = new WasmError(`Worker thread exited unexpectedly with code: ${code}`);
             // If Wasm module initialization fails due to exit, reject the promise
            if (this.rejectWasmModuleReady) {
                this.rejectWasmModuleReady(exitError);
            }
             this.handleWorkerError(exitError);
        }
      });

      // Send initialization message with paths needed by the worker
      console.log('[NodeLlamaRunner] Sending LOAD_NODE message to worker with paths:', { 
        wasmNodeModulePath: this.wasmNodeModulePath, 
        wasmNodePath: this.wasmNodePath 
      });
      this.worker.postMessage({
        event: workerActions.LOAD_NODE,
        wasmNodeModulePath: this.wasmNodeModulePath, // Send paths needed by worker
        wasmNodePath: this.wasmNodePath,
      });

    } catch (error: any) {
        console.error('[NodeLlamaRunner] Failed to initialize worker:', error);
        // Cannot proceed if worker fails to start
        throw new WasmError(`Failed to initialize Node.js worker: ${error.message}`);
    }
  }

  private handleWorkerMessage(data: any): void {
    const { event: action, text, error, errorDetails, metadata, stage, ...progressDetails } = data;
    // console.log(`[NodeLlamaRunner] Received message from worker:`, data); // Optional: for debugging

    switch (action) {
      case workerActions.INITIALIZED:
        // This message can come twice:
        // 1. After Wasm module is initialized in the worker.
        // 2. After model data is loaded into VFS in the worker.
        if (!this.isInitialized) { // First INITIALIZED: Wasm module is ready
            this.isInitialized = true;
            console.log('[NodeLlamaRunner] Worker initialized Wasm module.');
            if (this.resolveWasmModuleReady) {
                this.resolveWasmModuleReady(); // Resolve the promise
                this.resolveWasmModuleReady = null;
                this.rejectWasmModuleReady = null;
            }
        } else { // Second INITIALIZED (or subsequent if multiple models are loaded sequentially): Model VFS load complete
            this.isModelLoaded = true;
            this.isLoadingModel = false;
            console.log('[NodeLlamaRunner] Worker confirmed model loaded into VFS.');
            if (this.onModelLoadedCallback) {
                this.onModelLoadedCallback();
            }
            this.onModelLoadedCallback = null;
            this.onModelLoadErrorCallback = null;
        }
        break;
        
      case workerActions.WRITE_RESULT:
        if (this.currentTokenCallback && typeof text === 'string') {
          this.currentTokenCallback(text);
        }
        break;
        
      case workerActions.RUN_COMPLETED:
        console.log('[NodeLlamaRunner] Worker signaled run completion.');
        if (data.stderr && data.stderr.trim().length > 0) {
          console.warn(`[NodeLlamaRunner] Worker stderr output:\n${data.stderr.trim()}`);
        }
        this.currentCompletionCallback?.(); // Call without arguments
        this.resetCallbacks(); // Call the reset method
        break;
        
      case workerActions.MODEL_METADATA:
        this.currentModelMetadata = metadata as ModelSpecification;
        // Potentially validate metadata here if needed
        console.log('[NodeLlamaRunner] Received model metadata from worker.');
        // Progress callback might have already reported it, but update internal state
        this.reportProgress({ metadata: this.currentModelMetadata });
        break;
        
      case workerActions.PROGRESS_UPDATE:
        if (stage && this.currentProgressCallback) {
          const progressInfo: ProgressInfo = {
            stage: stage as LoadingStage,
            ...progressDetails,
            metadata: progressDetails.metadata || this.currentModelMetadata || undefined // Ensure metadata is attached if available
          };
          this.lastProgressInfo = progressInfo;
          this.currentProgressCallback(progressInfo);
        }
        break;
        
      case 'ERROR': // Handle structured errors from worker
        console.error('[NodeLlamaRunner] Received ERROR event from worker:', { error, errorDetails });
        let errorInstance: Error;
        if (errorDetails) {
            // Re-construct specific error types
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
                case 'FileError':
                  errorInstance = new FileError(errorDetails.message, errorDetails.fileName);
                  break;
                default:
                  errorInstance = new LocalWebAIError(errorDetails.message || error);
              }
              // Attempt to restore stack trace if provided (might not work perfectly)
              if (errorDetails.stack) {
                  errorInstance.stack = errorDetails.stack;
              }
        } else {
          errorInstance = classifyError(error || 'Unknown worker error');
        }
        this.handleWorkerError(errorInstance);
        break;
        
      default:
        console.warn(`[NodeLlamaRunner] Received unknown action from worker: ${action}`);
    }
  }
  
  // Centralized error handling for worker errors or promise rejections
  private handleWorkerError(error: Error): void {
      this.reportProgress({
        stage: LoadingStage.ERROR,
        message: error.message,
        error: error.message // Pass message for consistency
      });
      
      if (this.isLoadingModel && this.onModelLoadErrorCallback) {
          this.onModelLoadErrorCallback(error);
      }
      // Reset state if a critical error occurred during loading
      this.isLoadingModel = false;
      this.onModelLoadedCallback = null;
      this.onModelLoadErrorCallback = null;
      // Reset model loaded state only if error is *not* cancellation
      if (!(error instanceof OperationCancelledError)) {
          this.isModelLoaded = false; 
      }
      // Potentially terminate worker or handle based on error type
  }

  /**
   * Reports progress to the callback, updating internal state.
   */
  private reportProgress(info: Partial<ProgressInfo>): void {
    const progressInfo: ProgressInfo = {
      stage: info.stage || (this.lastProgressInfo?.stage || LoadingStage.PREPARING_MODEL_DATA),
      message: info.message || getStageDescription(info.stage || (this.lastProgressInfo?.stage || LoadingStage.PREPARING_MODEL_DATA)),
      loaded: info.loaded,
      total: info.total,
      metadata: info.metadata || this.currentModelMetadata || undefined,
      error: info.error
    };
    
    this.lastProgressInfo = progressInfo;
    if (this.currentProgressCallback) {
        this.currentProgressCallback(progressInfo);
    }
  }

  /**
   * Load a GGUF model from a local file path.
   * @param modelPath Path to the GGUF model file.
   * @param progressCallback Optional callback for detailed progress reporting.
   * @param signal Optional AbortSignal for cancellation support.
   * @returns Promise<void> Resolves when the model is loaded and ready in the worker.
   * @throws Various error types (FileError, VFSError, WasmError, etc.) on failure.
   */
  public async loadModel(
    modelPath: string, 
    progressCallback?: ProgressCallback,
    signal?: AbortSignal
  ): Promise<void> {
    if (!this.worker) {
      // If worker construction failed or worker exited early, wasmModuleReadyPromise might be rejected
      // Propagate that rejection.
      await this.wasmModuleReadyPromise; // This will throw if worker init failed.
      // If it didn't throw, but worker is null, it's an unexpected state.
      throw new WasmError('Node worker is not available, but initialization promise did not reject.');
    }
    if (this.isLoadingModel) {
        throw new ModelInitializationError('Another model is already being loaded.');
    }
    
    // Basic path validation (check if it's a non-empty string)
    if (typeof modelPath !== 'string' || modelPath.trim() === '') {
        throw new FileError('Invalid model path provided.');
    }
    
    this.isLoadingModel = true;
    this.isModelLoaded = false; // Reset model loaded status
    this.currentProgressCallback = progressCallback || null;
    this.lastProgressInfo = null;
    this.currentModelMetadata = null; // Reset metadata

    this.abortController = signal ? null : new AbortController();
    const abortSignal = signal || this.abortController!.signal;

    // Wait for the Wasm module to be ready if it hasn't initialized yet.
    // this.isInitialized is set when the first INITIALIZED message is received.
    // this.wasmModuleReadyPromise is resolved at that point.
    try {
      if (!this.isInitialized) {
        console.log('[NodeLlamaRunner] Waiting for Wasm module to initialize in worker...');
        // Check abortSignal before awaiting indefinitely
        if (abortSignal.aborted) {
          throw new OperationCancelledError('Model loading aborted before Wasm initialization');
        }
        
        const raceAbort = new Promise((_, rejectRace) => {
          abortSignal.addEventListener('abort', () => {
            rejectRace(new OperationCancelledError('Wasm initialization aborted'));
          }, { once: true });
        });

        await Promise.race([this.wasmModuleReadyPromise, raceAbort]);
        console.log('[NodeLlamaRunner] Wasm module is initialized. Proceeding to load model data.');
      }
    } catch (initError: any) {
        this.isLoadingModel = false; // Ensure flag is reset
        // Propagate the error (could be cancellation or actual Wasm init error)
        throw initError; 
    }

    const handleAbort = () => {
      if (this.isLoadingModel) {
        console.log('[NodeLlamaRunner] Aborting model loading...');
        if (this.worker) {
          this.worker.postMessage({ event: workerActions.CANCEL_LOAD });
        }
        const abortError = new OperationCancelledError('Model loading aborted by user');
        // Reject the promise via the error callback
        if (this.onModelLoadErrorCallback) {
            this.onModelLoadErrorCallback(abortError);
        }
        this.handleWorkerError(abortError); // Also report progress/update state
      }
    };

    abortSignal.addEventListener('abort', handleAbort);

    return new Promise<void>((resolve: () => void, reject) => {
      // Assign the correctly typed resolve/reject callbacks
      this.onModelLoadedCallback = resolve; 
      this.onModelLoadErrorCallback = reject;

      if (abortSignal.aborted) {
        this.isLoadingModel = false;
        reject(new OperationCancelledError('Model loading aborted before starting'));
        return;
      }

      // Report initial stage
      this.reportProgress({
        stage: LoadingStage.PREPARING_MODEL_DATA,
        message: 'Sending model path to worker'
      });

      // Send the model path to the worker to start the loading process
      this.worker!.postMessage({
        event: workerActions.LOAD_MODEL_DATA,
        modelPath: path.resolve(modelPath) // Send absolute path
      });
    }).finally(() => {
         // Clean up abort listener regardless of outcome
        abortSignal.removeEventListener('abort', handleAbort);
        if (this.abortController && !signal) {
          this.abortController = null; // Clean up controller we created
        }
        // Ensure loading flag is reset if promise settles (might already be reset by handlers)
        this.isLoadingModel = false; 
    });
  }

  /**
   * Generate text based on a prompt using the loaded model.
   * @param prompt The input prompt string.
   * @param params Optional parameters for text generation.
   * @param tokenCallback Callback for each generated token string.
   * @param completionCallback Callback for when generation is fully complete.
   * @throws ModelInitializationError if the model is not initialized or loaded.
   */
  public generateText(
    prompt: string,
    params: GenerateTextParams = {},
    tokenCallback: TokenCallback,
    completionCallback: CompletionCallback
  ): void {
    if (!this.worker) {
      throw new WasmError('Node worker is not running.');
    }
    if (!this.isInitialized || !this.isModelLoaded) {
      throw new ModelInitializationError('Worker is not initialized or model not loaded.');
    }
    if (this.currentTokenCallback || this.currentCompletionCallback) {
        // In a production scenario, might queue requests or handle concurrency
        console.warn('[NodeLlamaRunner] Text generation already in progress. Ignoring new request.');
        throw new ModelInitializationError("Text generation already in progress.");
    }

    this.currentTokenCallback = tokenCallback;
    this.currentCompletionCallback = completionCallback;

    console.log('[NodeLlamaRunner] Sending RUN_MAIN message to worker.');
    this.worker.postMessage({
      event: workerActions.RUN_MAIN,
      prompt: prompt,
      params: params, // Pass generation parameters
    });
  }

  /**
   * Cancels an ongoing model loading operation.
   * @returns True if cancellation was attempted, false otherwise.
   */
  public cancelLoading(): boolean {
    if (!this.isLoadingModel) {
      return false;
    }
    if (this.abortController) {
        this.abortController.abort(); // Use controller if we created it
        return true;
    } else {
        // If using an external signal, we can't abort it here,
        // but we can still tell the worker to cancel.
        if (this.worker) {
            console.log('[NodeLlamaRunner] Requesting worker cancel load (external signal used).');
            this.worker.postMessage({ event: workerActions.CANCEL_LOAD });
            return true;
        }
    }
    return false;
  }

  /**
   * Retrieves the current model's metadata.
   * @returns The current model's metadata or null if no model is loaded.
   */
  public getModelMetadata(): ModelSpecification | null {
    return this.currentModelMetadata;
  }

  /**
   * Retrieves the last reported progress information.
   * @returns The last progress info or null.
   */
  public getLastProgressInfo(): ProgressInfo | null {
    return this.lastProgressInfo;
  }

  /**
   * Terminates the worker thread immediately.
   * The NodeLlamaRunner instance should not be used after calling this.
   */
  public terminate(): void {
    if (this.worker) {
      console.log('[NodeLlamaRunner] Terminating worker thread.');
      this.worker.terminate();
      this.worker = null;
      this.isInitialized = false;
      this.isModelLoaded = false;
      this.isLoadingModel = false;
      // Clear callbacks and state
      this.onModelLoadedCallback = null;
      this.onModelLoadErrorCallback = null;
      this.currentProgressCallback = null;
      this.currentTokenCallback = null;
      this.currentCompletionCallback = null;
      this.currentModelMetadata = null;
      this.lastProgressInfo = null;
      this.abortController = null;
    }
  }

  private resetCallbacks(): void {
    this.currentTokenCallback = null;
    this.currentCompletionCallback = null;
  }
} 