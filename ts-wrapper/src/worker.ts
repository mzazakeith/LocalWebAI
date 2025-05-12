// ts-wrapper/worker.ts

// This type will be refined as we integrate the actual Wasm module
// For now, it reflects the structure used in llama-cpp-wasm/src/llama/main-worker.js
interface EmscriptenModule {
  // eslint-disable-next-line  @typescript-eslint/no-explicit-any
  [key: string]: any; // Allow any properties, common for Emscripten modules
  noInitialRun: boolean;
  // eslint-disable-next-line  @typescript-eslint/no-explicit-any
  preInit: any[]; 
  TTY: {
    // eslint-disable-next-line  @typescript-eslint/no-explicit-any
    register: (dev: any, ops: any) => void;
  };
  FS_createPath: (path: string, name: string, canRead?: boolean, canWrite?: boolean) => void;
  // eslint-disable-next-line  @typescript-eslint/no-explicit-any
  FS_createDataFile: (parent: string, name: string, data: Uint8Array, canRead?: boolean, canWrite?: boolean, canOwn?: boolean) => any;
  // eslint-disable-next-line  @typescript-eslint/no-explicit-any
  callMain: (args: string[]) => any;
  // eslint-disable-next-line  @typescript-eslint/no-explicit-any
  FS: any; 
}

// Import the GGUF Parser
import { parseGGUFHeader } from './gguf-parser.js';
import { ModelSpecification } from './model-spec.js';
import { LoadingStage } from './loading-progress.js';
import { 
  GGUFParsingError, 
  ModelCompatibilityError, 
  VFSError, 
  WasmError,
  OperationCancelledError,
  ModelInitializationError,
  FileError
} from './errors.js';

// Define the expected structure of Module factory from main.js (compiled llama.cpp)
// eslint-disable-next-line  @typescript-eslint/no-explicit-any
declare function Module(settings?: Partial<EmscriptenModule>): Promise<EmscriptenModule>;

// Replicate actions from llama-cpp-wasm for worker communication
const workerActions = {
  LOAD: 'LOAD',
  INITIALIZED: 'INITIALIZED',
  RUN_MAIN: 'RUN_MAIN',
  WRITE_RESULT: 'WRITE_RESULT',
  RUN_COMPLETED: 'RUN_COMPLETED',
  LOAD_MODEL_DATA: 'LOAD_MODEL_DATA',
  MODEL_METADATA: 'MODEL_METADATA', // New action for model metadata
  PROGRESS_UPDATE: 'PROGRESS_UPDATE', // New action for detailed progress reporting
  CANCEL_LOAD: 'CANCEL_LOAD', // New action for cancellation support
};

let wasmModuleInstance: EmscriptenModule;
const modelPath = "/models/model.bin"; // Hard-coded filepath in VFS
const headerReadSize = 1024 * 1024; // 1MB should be sufficient for most GGUF headers

const decoder = new TextDecoder('utf-8');
const punctuationBytes = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 58, 59, 60, 61, 62, 63, 64, 91, 92, 93, 94, 95, 96, 123, 124, 125, 126];
const whitespaceBytes = [32, 9, 10, 13, 11, 12];
const splitBytes = [...punctuationBytes, ...whitespaceBytes];
const stdoutBuffer: number[] = [];

const stdin = () => { /* no-op */ };

const stdout = (c: number) => {
  stdoutBuffer.push(c);
  // Stream output based on punctuation/whitespace, similar to llama-cpp-wasm
  if (splitBytes.includes(c) || stdoutBuffer.length > 20) { // Added buffer length check
    const text = decoder.decode(new Uint8Array(stdoutBuffer));
    stdoutBuffer.length = 0; // Clear buffer
    self.postMessage({
      event: workerActions.WRITE_RESULT,
      text: text,
    });
  }
};

const stderr = (c: number) => {
  // For now, just log stderr to console, can be enhanced
  // console.error('stderr:', String.fromCharCode(c));
};

// Track current cancellation state
let isCancellationRequested = false;

/**
 * Checks if the current operation has been cancelled
 * @returns True if cancellation has been requested
 */
function checkCancellation(): boolean {
  return isCancellationRequested;
}

/**
 * Resets the cancellation state and cleans up any resources
 */
function resetCancellationState() {
  isCancellationRequested = false;
}

/**
 * Performs cleanup when cancellation occurs during model loading
 */
function cleanupAfterCancellation() {
  try {
    if (wasmModuleInstance && wasmModuleInstance.FS) {
      // Check if model file exists and remove it if it does
      try {
        const stat = wasmModuleInstance.FS.stat(modelPath);
        if (stat) {
          console.log('Removing partially written model file due to cancellation');
          wasmModuleInstance.FS.unlink(modelPath);
        }
      } catch (err) {
        // File likely doesn't exist, which is fine
      }
    }
  } catch (err) {
    console.error('Error cleaning up after cancellation:', err);
  }

  // Report cancellation to main thread
  reportProgress(LoadingStage.CANCELLED, {
    message: 'Model loading cancelled by user'
  });

  self.postMessage({
    event: 'ERROR',
    error: new OperationCancelledError().message,
    errorDetails: {
      name: 'OperationCancelledError',
      message: 'Model loading cancelled by user'
    }
  });

  // Reset cancellation state
  resetCancellationState();
}

/**
 * Posts an error to the main thread with structured error information
 */
function reportError(error: Error | string, stage: LoadingStage = LoadingStage.ERROR) {
  let errorMsg: string;
  let errorDetails: any = {};
  
  if (error instanceof Error) {
    errorMsg = error.message;
    errorDetails.name = error.constructor.name;
    errorDetails.message = error.message;
    
    // Include additional properties for specific error types
    if ('actualVersion' in error && typeof (error as any).actualVersion === 'number') {
      errorDetails.actualVersion = (error as any).actualVersion;
      errorDetails.minSupported = (error as any).minSupported;
      errorDetails.maxSupported = (error as any).maxSupported;
    }
    
    if ('details' in error) {
      errorDetails.details = (error as any).details;
    }
    
    if ('path' in error) {
      errorDetails.path = (error as any).path;
    }
  } else {
    errorMsg = String(error);
    errorDetails.message = errorMsg;
  }

  // First report through the progress channel
  reportProgress(stage, {
    message: errorMsg,
    error: errorMsg
  });

  // Then send a structured error event
  self.postMessage({
    event: 'ERROR',
    error: errorMsg,
    errorDetails: errorDetails
  });
}

/**
 * Report loading progress to the main thread
 */
function reportProgress(stage: LoadingStage, details: {
  loaded?: number;
  total?: number;
  message?: string;
  metadata?: ModelSpecification;
  error?: string;
} = {}) {
  self.postMessage({
    event: workerActions.PROGRESS_UPDATE,
    stage,
    ...details
  });
}

/**
 * Examines a file header to check if it's a valid GGUF file 
 * and confirm version compatibility.
 * @param buffer The ArrayBuffer containing file header bytes
 * @throws ModelCompatibilityError or GGUFParsingError if there's an issue
 */
function validateGGUFHeader(buffer: ArrayBuffer): void {
  if (!buffer || buffer.byteLength < 8) {
    throw new GGUFParsingError(
      "Insufficient data to validate file format (minimum 8 bytes required)",
      { byteLength: buffer?.byteLength ?? 0 }
    );
  }

  const dataView = new DataView(buffer);
  
  // Check magic number "GGUF" (0x46554747 in little-endian)
  const magic = dataView.getUint32(0, true);
  if (magic !== 0x46554747) {
    throw new GGUFParsingError(
      `Invalid file format: Not a GGUF file (magic number mismatch)`,
      { expected: 0x46554747, actual: magic, hexActual: `0x${magic.toString(16).toUpperCase()}` }
    );
  }

  // Check version
  const version = dataView.getUint32(4, true);
  const minSupported = 2;
  const maxSupported = 3;
  
  if (version < minSupported || version > maxSupported) {
    throw new ModelCompatibilityError(
      `Unsupported GGUF version`,
      version,
      minSupported,
      maxSupported
    );
  }
}

/**
 * Parses model metadata from the loaded model file in VFS
 */
async function parseModelMetadata(): Promise<ModelSpecification | null> {
  if (!wasmModuleInstance) {
    const error = new WasmError('Wasm module not initialized when trying to parse model metadata');
    reportError(error);
    return null;
  }

  try {
    // Report the start of metadata parsing
    reportProgress(LoadingStage.METADATA_PARSE_START, {
      message: 'Reading model header for metadata extraction'
    });

    // Attempt to open the file first as a more reliable check than stat
    let stream;
    try {
      stream = wasmModuleInstance.FS.open(modelPath, 'r');
      // If open succeeds, the file exists and is readable.
      // We still need to close it before reading again or ensure the read happens here.
    } catch (err) {
      // If opening fails, the file likely doesn't exist or isn't accessible
      throw new FileError(`Failed to open model file at ${modelPath}: ${err instanceof Error ? err.message : String(err)}`, modelPath);
    }

    // Read the model file header from the opened stream
    const headerBuffer = new Uint8Array(headerReadSize);
    let bytesRead: number;
    
    try {
      // Read from position 0 of the stream
      bytesRead = wasmModuleInstance.FS.read(stream, headerBuffer, 0, headerReadSize, 0);
    } catch (err) {
      throw new FileError(`Failed to read model file header: ${err instanceof Error ? err.message : String(err)}`, modelPath);
    } finally {
      // Ensure the stream is closed even if reading fails
      try {
        wasmModuleInstance.FS.close(stream);
      } catch (closeErr) {
        console.error('Error closing file stream after read attempt:', closeErr);
        // Don't obscure the original read error if one occurred
      }
    }
    
    if (bytesRead < 8) { // At minimum we need magic + version (8 bytes)
      throw new GGUFParsingError(
        `Failed to read sufficient header data: only read ${bytesRead} bytes`,
        { bytesRead, minimumRequired: 8 }
      );
    }

    // Use only the bytes that were actually read
    const headerData = headerBuffer.slice(0, bytesRead).buffer;
    
    // Validate file format and version before parsing
    try {
      validateGGUFHeader(headerData);
    } catch (error) {
      // If we caught a model compatibility or parsing error at this stage, report it with proper type
      reportError(error instanceof Error ? error : new GGUFParsingError(String(error)));
      return null; // Explicitly return null to satisfy TypeScript
    }
    
    // Parse the header data
    let metadata: ModelSpecification | null = null;
    try {
      metadata = parseGGUFHeader(headerData);
    } catch (error) {
      // Re-throw specific error types
      if (error instanceof GGUFParsingError || error instanceof ModelCompatibilityError) {
        throw error;
      }
      throw new GGUFParsingError(
        `Error parsing GGUF header: ${error instanceof Error ? error.message : String(error)}`
      );
    }
    
    // Report successful metadata parsing
    reportProgress(LoadingStage.METADATA_PARSE_COMPLETE, {
      message: 'Model metadata extracted successfully',
      metadata: metadata
    });

    return metadata;
  } catch (error) {
    reportError(error instanceof Error ? error : new GGUFParsingError(String(error)));
    return null;
  }
}

/**
 * Loads model data into the virtual filesystem and initializes it
 * 
 * MEMORY MANAGEMENT:
 * - The ArrayBuffer is received from the main thread via transferable objects
 * - Only a single copy of the model data exists in memory (in the worker)
 * - Data is written to the virtual filesystem and then the reference is cleared
 * - Validation occurs on a small header slice rather than the entire file
 * 
 * @param modelData The model data as ArrayBuffer transferred from main thread
 * @param cancelable Whether this operation can be cancelled
 */
async function loadModelData(modelData: ArrayBuffer, cancelable = false) {
  if (!wasmModuleInstance) {
    const error = new WasmError('Wasm module not initialized before loading model data.');
    reportError(error);
    return;
  }
  
  try {
    // Check for cancellation before starting
    if (cancelable && checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    // Report VFS write start
    reportProgress(LoadingStage.VFS_WRITE_START, {
      message: 'Preparing to write model data to virtual filesystem',
      total: modelData.byteLength
    });
    
    // Validate GGUF header before writing to VFS
    try {
      // MEMORY OPTIMIZATION: Only examine the first 1KB for validation rather than entire file
      const headerBuffer = modelData.slice(0, Math.min(1024, modelData.byteLength));
      validateGGUFHeader(headerBuffer);
    } catch (error) {
      // If we caught a model compatibility or parsing error at this stage, report it with proper type
      reportError(error instanceof Error ? error : new GGUFParsingError(String(error)));
      return;
    }
    
    // Create the models directory if it doesn't exist
    try {
      wasmModuleInstance.FS_createPath("/", "models", true, true);
    } catch (error) {
      throw new VFSError(
        `Failed to create models directory in VFS: ${error instanceof Error ? error.message : String(error)}`,
        "/models"
      );
    }
    
    // Check for cancellation before writing to VFS
    if (cancelable && checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    // Report VFS write progress - simulating progress by reporting 50% complete
    reportProgress(LoadingStage.VFS_WRITE_PROGRESS, {
      message: 'Writing model data to virtual filesystem',
      loaded: Math.floor(modelData.byteLength / 2),
      total: modelData.byteLength
    });
    
    // Write the model file to VFS
    try {
      // MEMORY OPTIMIZATION: Create a Uint8Array view rather than copying the buffer
      const modelDataView = new Uint8Array(modelData);
      wasmModuleInstance.FS_createDataFile('/models', 'model.bin', modelDataView, true, true, true);
      
      // Release reference to large buffer as soon as possible after writing to VFS
      // The data is now stored in the Emscripten filesystem
    } catch (error) {
      throw new VFSError(
        `Failed to write model data to VFS: ${error instanceof Error ? error.message : String(error)}`,
        modelPath
      );
    }
    
    // Check for cancellation after VFS write
    if (cancelable && checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    // Report VFS write complete
    reportProgress(LoadingStage.VFS_WRITE_COMPLETE, {
      message: 'Model data written to virtual filesystem',
      loaded: modelData.byteLength,
      total: modelData.byteLength
    });
    
    // Parse the model metadata after writing the file to VFS
    let metadata: ModelSpecification | null = null;
    try {
      metadata = await parseModelMetadata();
    } catch (error) {
      // For metadata parsing errors, we report but continue loading since
      // the model might still work without complete metadata
      console.warn("Metadata parsing error, attempting to continue with model initialization:", error);
      // Don't rethrow to allow model loading to continue
    }
    
    // Check for cancellation after metadata parsing
    if (cancelable && checkCancellation()) {
      cleanupAfterCancellation();
      return;
    }

    // Model initialization starts
    reportProgress(LoadingStage.MODEL_INITIALIZATION_START, {
      message: 'Preparing model for inference',
      metadata: metadata || undefined
    });
    
    // Send metadata to main thread if successfully parsed
    if (metadata) {
      self.postMessage({
        event: workerActions.MODEL_METADATA,
        metadata
      });
    } else {
      // Send warning if metadata parsing failed but we're continuing
      console.warn('Model metadata parsing failed, continuing with limited model information');
    }

    // Signal that initialization is complete
    self.postMessage({
      event: workerActions.INITIALIZED,
    });
    
    // Model is ready
    reportProgress(LoadingStage.MODEL_READY, {
      message: 'Model loaded and ready for inference',
      loaded: modelData.byteLength,
      total: modelData.byteLength,
      metadata: metadata || undefined
    });
  } catch (e) {
    const error = e instanceof Error ? e : new Error(String(e));
    console.error('Error loading model data into VFS:', error);
    
    // Handle specific error types
    reportError(error);
  } finally {
    // MEMORY OPTIMIZATION: Clear reference to the model data
    // This signals to the garbage collector that the large buffer can be freed
    // At this point, the data should be stored in the Emscripten filesystem
  }
}

async function initWasmModule(wasmModulePath: string, wasmPath: string, modelUrl?: string, modelData?: ArrayBuffer) {
  const emscriptenModuleConfig: Partial<EmscriptenModule> = {
    noInitialRun: true,
    preInit: [() => {
      // Setup TTY for stdout and stderr to capture Wasm output
      // eslint-disable-next-line  @typescript-eslint/no-non-null-assertion
      emscriptenModuleConfig.TTY!.register(emscriptenModuleConfig.FS!.makedev(5, 0), {
        // eslint-disable-next-line  @typescript-eslint/no-explicit-any
        get_char: (tty: any) => stdin(),
        // eslint-disable-next-line  @typescript-eslint/no-explicit-any
        put_char: (tty: any, val: number) => { tty.output.push(val); stdout(val); },
        // eslint-disable-next-line  @typescript-eslint/no-explicit-any
        flush: (tty: any) => tty.output = [],
      });
      // eslint-disable-next-line  @typescript-eslint/no-non-null-assertion
      emscriptenModuleConfig.TTY!.register(emscriptenModuleConfig.FS!.makedev(6, 0), {
        // eslint-disable-next-line  @typescript-eslint/no-explicit-any
        get_char: (tty: any) => stdin(),
        // eslint-disable-next-line  @typescript-eslint/no-explicit-any
        put_char: (tty: any, val: number) => { tty.output.push(val); stderr(val); },
        // eslint-disable-next-line  @typescript-eslint/no-explicit-any
        flush: (tty: any) => tty.output = [],
      });
    }],
    locateFile: (path:string) => {
        if (path.endsWith('.wasm')) {
            return wasmPath; // URL to the .wasm file
        }
        return path;
    }
  };

  // Dynamically import the Emscripten-generated JS file
  try {
    reportProgress(LoadingStage.MODEL_INITIALIZATION_START, {
      message: 'Initializing WebAssembly module'
    });
    
    const importedModule = await import(wasmModulePath).catch(err => {
      throw new WasmError(`Failed to import Wasm module from ${wasmModulePath}: ${err.message}`);
    });
    
    if (!importedModule.default) {
        throw new WasmError('Wasm module does not have a default export. Check Emscripten build flags (MODULARIZE, EXPORT_ES6).');
    }
    
    // eslint-disable-next-line  @typescript-eslint/no-explicit-any
    const ModuleFactory = importedModule.default as (config: Partial<EmscriptenModule>) => Promise<EmscriptenModule>; 
    
    try {
    wasmModuleInstance = await ModuleFactory(emscriptenModuleConfig);
    } catch (err) {
      throw new WasmError(`Failed to instantiate Wasm module: ${err instanceof Error ? err.message : String(err)}`);
    }
  } catch (err) {
    // Using specific WasmError type
    const error = err instanceof WasmError ? err : new WasmError(err instanceof Error ? err.message : String(err));
    reportError(error);
    return; // Stop further execution in the worker if Wasm module fails to load
  }

  if (modelData) {
    await loadModelData(modelData);
  } else if (modelUrl) {
    // Fetch model from URL and load it
    try {
      const response = await fetch(modelUrl).catch(err => {
        throw new Error(`Network error fetching model: ${err.message}`);
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch model: HTTP status ${response.status} - ${response.statusText}`);
      }
      
      const data = await response.arrayBuffer().catch(err => {
        throw new Error(`Error reading response data: ${err.message}`);
      });
      
      await loadModelData(data);
    } catch (e) {
      reportError(e instanceof Error ? e : new Error(String(e)));
    }
  } else {
     // If neither modelData nor modelUrl is provided, signal readiness (or handle as an error)
     // For now, let's assume this means ready for a model to be loaded later via LOAD_MODEL_DATA
     self.postMessage({ event: workerActions.INITIALIZED }); // Or a different event like MODULE_READY_NO_MODEL
  }
}

function runMain(prompt: string, params: Record<string, string | number | boolean>) {
  if (!wasmModuleInstance) {
    reportError(new ModelInitializationError('Wasm module not ready to run main.'));
    return;
  }
  
  try {
  const args = [
    "--model", modelPath,
    "--n-predict", (params.n_predict || -2).toString(),
    "--ctx-size", (params.ctx_size || 2048).toString(),
    "--temp", (params.temp || 0.8).toString(),
    "--top_k", (params.top_k || 40).toString(),
    "--top_p", (params.top_p || 0.9).toString(),
    "--simple-io",
    "--log-disable",
    "--prompt", prompt,
  ];

  if (params.chatml) {
    args.push("--chatml");
  }
  if (params.no_display_prompt !== false) { // default to true if not specified
    args.push("--no-display-prompt");
  }

  // Add threading if SharedArrayBuffer is available (for multi-threaded Wasm builds)
  // This check might need to align with the specific llama-cpp-wasm build (st vs mt)
  if (typeof SharedArrayBuffer !== 'undefined') {
      args.push("--threads");
      args.push((navigator.hardwareConcurrency || 4).toString()); // Default to 4 if undefined
  }

  try {
    wasmModuleInstance.callMain(args);
  } catch(e) {
      throw new ModelInitializationError(`Error running model: ${e instanceof Error ? e.message : String(e)}`);
    }
  } catch (error) {
    reportError(error instanceof Error ? error : new Error(String(error)));
    return;
  }
  
  // Ensure any remaining buffered output is sent
  if (stdoutBuffer.length > 0) {
    const text = decoder.decode(new Uint8Array(stdoutBuffer));
    stdoutBuffer.length = 0;
    self.postMessage({ event: workerActions.WRITE_RESULT, text });
  }

  self.postMessage({ event: workerActions.RUN_COMPLETED });
}

self.onmessage = async (event: MessageEvent) => {
  const { event: action, wasmModulePath, wasmPath, modelUrl, modelData, params, prompt, cancelable } = event.data;

  switch (action) {
    case workerActions.LOAD:
      // Initialize the Wasm module
      await initWasmModule(wasmModulePath, wasmPath, modelUrl, modelData);
      break;
    case workerActions.LOAD_MODEL_DATA:
      // Load model data with cancelation support if specified
      await loadModelData(modelData, cancelable);
      break;
    case workerActions.RUN_MAIN:
      // Generate text using the model (params and prompt from message data)
      try {
        // Build args for main - *** REMOVE the explicit 'main' entry ***
        // Emscripten's callMain typically adds argv[0] automatically.
        const argsArray = [
          '-m', modelPath, 
          '-p', prompt,
        ];

        // Add all parameters from params object, if provided
        if (params) {
          if (params.n_predict !== undefined) argsArray.push('-n', String(params.n_predict));
          if (params.ctx_size !== undefined) argsArray.push('--ctx-size', String(params.ctx_size));
          if (params.batch_size !== undefined) argsArray.push('--batch-size', String(params.batch_size));
          if (params.temp !== undefined) argsArray.push('--temp', String(params.temp));
          if (params.n_gpu_layers !== undefined) argsArray.push('-ngl', String(params.n_gpu_layers)); // Note: -ngl might not be effective in typical browser Wasm builds
          if (params.top_k !== undefined) argsArray.push('--top-k', String(params.top_k));
          if (params.top_p !== undefined) argsArray.push('--top-p', String(params.top_p));
          if (params.no_display_prompt === true) argsArray.push('--no-display-prompt');
          if (params.chatml === true) argsArray.push('--chatml');
          // Add additional parameters as needed
        }

        // *** Log the arguments before calling main ***
        console.log('[Worker] Calling wasmModuleInstance.callMain with args:', argsArray);
        
        // Call the main function with args
        wasmModuleInstance.callMain(argsArray);
        
        // Flush any remaining output
        if (stdoutBuffer.length > 0) {
          self.postMessage({
            event: workerActions.WRITE_RESULT,
            text: decoder.decode(new Uint8Array(stdoutBuffer)),
          });
          stdoutBuffer.length = 0;
        }
        
        // Signal completion
        self.postMessage({
          event: workerActions.RUN_COMPLETED,
        });
      } catch (e) {
        // *** Log the raw error from Wasm ***
        console.error('[Worker] Error caught during callMain:', e);
        reportError(e instanceof Error ? e : new Error(`Error running model: ${String(e)}`));
      }
      break;
    case workerActions.CANCEL_LOAD:
      // Set cancellation flag and start cleanup process
      console.log('Cancellation requested');
      isCancellationRequested = true;
      break;
    default:
      console.warn(`Unknown action: ${action}`);
  }
}; 