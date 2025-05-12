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
    error: 'Model loading cancelled by user'
  });

  // Reset cancellation state
  resetCancellationState();
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
 * Parses model metadata from the loaded model file in VFS
 */
async function parseModelMetadata(): Promise<ModelSpecification | null> {
  if (!wasmModuleInstance) {
    console.error('Wasm module not initialized when trying to parse model metadata');
    reportProgress(LoadingStage.ERROR, {
      message: 'Wasm module not initialized when trying to parse model metadata'
    });
    return null;
  }

  try {
    // Report the start of metadata parsing
    reportProgress(LoadingStage.METADATA_PARSE_START, {
      message: 'Reading model header for metadata extraction'
    });

    // Read the model file header from VFS
    const stream = wasmModuleInstance.FS.open(modelPath, 'r');
    const headerBuffer = new Uint8Array(headerReadSize);
    const bytesRead = wasmModuleInstance.FS.read(stream, headerBuffer, 0, headerReadSize, 0);
    wasmModuleInstance.FS.close(stream);
    
    if (bytesRead < 8) { // At minimum we need magic + version (8 bytes)
      const errorMsg = `Failed to read sufficient header data: only read ${bytesRead} bytes`;
      reportProgress(LoadingStage.ERROR, { message: errorMsg });
      throw new Error(errorMsg);
    }

    // Use only the bytes that were actually read
    const headerData = headerBuffer.slice(0, bytesRead).buffer;
    
    // Parse the header data
    const metadata = parseGGUFHeader(headerData);
    
    // Report successful metadata parsing
    reportProgress(LoadingStage.METADATA_PARSE_COMPLETE, {
      message: 'Model metadata extracted successfully',
      metadata: metadata
    });

    return metadata;
  } catch (error) {
    console.error('Error parsing model metadata:', error);
    reportProgress(LoadingStage.ERROR, {
      message: `Error parsing model metadata: ${error instanceof Error ? error.message : String(error)}`
    });
    return null;
  }
}

async function loadModelData(modelData: ArrayBuffer, cancelable = false) {
  if (!wasmModuleInstance) {
    const errorMsg = 'Wasm module not initialized before loading model data.';
    console.error(errorMsg);
    reportProgress(LoadingStage.ERROR, { message: errorMsg });
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
    
    wasmModuleInstance.FS_createPath("/", "models", true, true);
    
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
    
    wasmModuleInstance.FS_createDataFile('/models', 'model.bin', new Uint8Array(modelData), true, true, true);
    
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
    const metadata = await parseModelMetadata();
    
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
      // Send error if metadata parsing failed
      self.postMessage({
        event: 'ERROR',
        error: 'Failed to parse model metadata'
      });
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
    const errorMsg = e instanceof Error ? e.message : String(e);
    console.error('Error loading model data into VFS:', e);
    // Post error message back to the main thread
    reportProgress(LoadingStage.ERROR, {
      message: `Error loading model data: ${errorMsg}`
    });
    
    self.postMessage({ 
      event: 'ERROR', 
      error: errorMsg
    });
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
    
    const importedModule = await import(wasmModulePath);
    if (!importedModule.default) {
        throw new Error('Wasm module does not have a default export. Check Emscripten build flags (MODULARIZE, EXPORT_ES6).');
    }
    // eslint-disable-next-line  @typescript-eslint/no-explicit-any
    const ModuleFactory = importedModule.default as (config: Partial<EmscriptenModule>) => Promise<EmscriptenModule>; 
    wasmModuleInstance = await ModuleFactory(emscriptenModuleConfig);
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`Error importing or instantiating wasm module from ${wasmModulePath}:`, err);
    // Post an error back to the main thread if initialization fails
    reportProgress(LoadingStage.ERROR, {
      message: `Failed to load Wasm module: ${errorMsg}`
    });
    
    self.postMessage({ event: 'ERROR', error: `Failed to load Wasm module: ${errorMsg}` });
    return; // Stop further execution in the worker if Wasm module fails to load
  }

  if (modelData) {
    await loadModelData(modelData);
  } else if (modelUrl) {
    // Fetch model from URL and load it
    try {
      const response = await fetch(modelUrl);
      if (!response.ok) throw new Error(`Failed to fetch model: ${response.status}`);
      const data = await response.arrayBuffer();
      await loadModelData(data);
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      console.error('Error fetching or loading model from URL:', e);
      // Post error message back to the main thread
      reportProgress(LoadingStage.ERROR, {
        message: `Error fetching model from URL: ${errorMsg}`
      });
      
      self.postMessage({ 
        event: 'ERROR', 
        error: errorMsg
      });
    }
  } else {
     // If neither modelData nor modelUrl is provided, signal readiness (or handle as an error)
     // For now, let's assume this means ready for a model to be loaded later via LOAD_MODEL_DATA
     self.postMessage({ event: workerActions.INITIALIZED }); // Or a different event like MODULE_READY_NO_MODEL
  }
}

function runMain(prompt: string, params: Record<string, string | number | boolean>) {
  if (!wasmModuleInstance) {
    console.error('Wasm module not ready to run main.');
    return;
  }
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
    console.error("Error during callMain:", e);
    // If llama.cpp exits non-zero, Emscripten might throw an exception here.
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
        // Build args for main
        const argsArray = [
          'main', 
          '-m', modelPath, 
          '-p', prompt,
        ];

        // Add all parameters from params object, if provided
        if (params) {
          if (params.n_predict !== undefined) argsArray.push('-n', String(params.n_predict));
          if (params.ctx_size !== undefined) argsArray.push('--ctx-size', String(params.ctx_size));
          if (params.batch_size !== undefined) argsArray.push('--batch-size', String(params.batch_size));
          if (params.temp !== undefined) argsArray.push('--temp', String(params.temp));
          if (params.n_gpu_layers !== undefined) argsArray.push('-ngl', String(params.n_gpu_layers));
          if (params.top_k !== undefined) argsArray.push('--top-k', String(params.top_k));
          if (params.top_p !== undefined) argsArray.push('--top-p', String(params.top_p));
          if (params.no_display_prompt === true) argsArray.push('--no-display-prompt');
          if (params.chatml === true) argsArray.push('--chatml');
          // Add additional parameters as needed
        }

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
        console.error('Error running model:', e);
        self.postMessage({ 
          event: 'ERROR', 
          error: e instanceof Error ? e.message : String(e)
        });
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