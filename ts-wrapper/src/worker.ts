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
};

let wasmModuleInstance: EmscriptenModule;
const modelPath = "/models/model.bin"; // Hard-coded filepath in VFS

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

async function loadModelData(modelData: ArrayBuffer) {
  if (!wasmModuleInstance) {
    console.error('Wasm module not initialized before loading model data.');
    return;
  }
  try {
    wasmModuleInstance.FS_createPath("/", "models", true, true);
    wasmModuleInstance.FS_createDataFile('/models', 'model.bin', new Uint8Array(modelData), true, true, true);
    self.postMessage({
      event: workerActions.INITIALIZED,
    });
  } catch (e) {
    console.error('Error loading model data into VFS:', e);
    // Optionally, post an error message back to the main thread
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
    const importedModule = await import(wasmModulePath);
    if (!importedModule.default) {
        throw new Error('Wasm module does not have a default export. Check Emscripten build flags (MODULARIZE, EXPORT_ES6).');
    }
    // eslint-disable-next-line  @typescript-eslint/no-explicit-any
    const ModuleFactory = importedModule.default as (config: Partial<EmscriptenModule>) => Promise<EmscriptenModule>; 
    wasmModuleInstance = await ModuleFactory(emscriptenModuleConfig);
  } catch (err) {
    console.error(`Error importing or instantiating wasm module from ${wasmModulePath}:`, err);
    // Post an error back to the main thread if initialization fails
    const errorMessage = err instanceof Error ? err.message : String(err);
    self.postMessage({ event: 'ERROR', error: `Failed to load Wasm module: ${errorMessage}` });
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
      console.error('Error fetching or loading model from URL:', e);
      // Optionally, post an error message back to the main thread
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

self.onmessage = async (e) => {
  switch (e.data.event) {
    case workerActions.LOAD:
      // e.data.wasmModulePath: path/URL to the Emscripten JS glue (e.g., './main.js')
      // e.data.wasmPath: path/URL to the actual .wasm file (e.g., './main.wasm')
      // e.data.modelUrl: optional URL to a model file to auto-load
      await initWasmModule(e.data.wasmModulePath, e.data.wasmPath, e.data.modelUrl);
      break;
    case workerActions.LOAD_MODEL_DATA:
      await loadModelData(e.data.modelData as ArrayBuffer);
      break;
    case workerActions.RUN_MAIN:
      runMain(e.data.prompt as string, e.data.params as Record<string, string | number | boolean>);
      break;
  }
}; 