/**
 * Loading progress tracking for model loading process.
 * Defines stages, types and utilities for granular progress reporting.
 */

import { ModelSpecification } from './model-spec.js';

/**
 * Enum of distinct model loading stages
 */
export enum LoadingStage {
  // Main thread stages
  DOWNLOADING_FROM_SOURCE = 'downloading_from_source',
  READING_FROM_FILE = 'reading_from_file',
  PREPARING_MODEL_DATA = 'preparing_model_data',
  
  // Worker stages
  VFS_WRITE_START = 'vfs_write_start',
  VFS_WRITE_PROGRESS = 'vfs_write_progress',
  VFS_WRITE_COMPLETE = 'vfs_write_complete',
  METADATA_PARSE_START = 'metadata_parse_start',
  METADATA_PARSE_COMPLETE = 'metadata_parse_complete',
  MODEL_INITIALIZATION_START = 'model_initialization_start',
  MODEL_READY = 'model_ready',
  
  // Error stages
  ERROR = 'error',
  
  // Cancellation stage
  CANCELLED = 'cancelled',

  // New stages
  MODEL_CACHE_CHECK_COMPLETE = "model_cache_check_complete",
  MODEL_FETCH_START = "model_fetch_start",
  MODEL_FETCH_PROGRESS = "model_fetch_progress",
  MODEL_FETCH_COMPLETE = "model_fetch_complete",
  MODEL_DECOMPRESSION_START = "model_decompression_start",
  MODEL_DECOMPRESSION_PROGRESS = "model_decompression_progress",
  MODEL_DECOMPRESSION_COMPLETE = "model_decompression_complete",
  METADATA_PARSE_PROGRESS = "metadata_parse_progress",
  EMBIND_MODULE_LOAD_START = "embind_module_load_start",
  EMBIND_MODULE_LOAD_PROGRESS = "embind_module_load_progress",
  EMBIND_MODULE_LOAD_COMPLETE = "embind_module_load_complete",
  MODEL_INITIALIZATION_PROGRESS = "model_initialization_progress",
  MODEL_INITIALIZATION_COMPLETE = "model_initialization_complete",
  IDLE = "idle"
}

/**
 * Enhanced progress info with loading stage and additional context
 */
export interface ProgressInfo {
  stage: LoadingStage;
  loaded?: number;    // Bytes loaded/processed (if applicable)
  total?: number;     // Total bytes (if applicable)
  message?: string;   // Human-readable message about current status
  metadata?: ModelSpecification; // Available after metadata parsing
  error?: string;     // Error message if stage is ERROR
}

/**
 * Progress callback type for reporting detailed loading progress
 */
export type ProgressCallback = (progress: ProgressInfo) => void;

/**
 * Error type mapping for user-friendly descriptions
 */
const ERROR_TYPE_DESCRIPTIONS: Record<string, string> = {
  'NetworkError': 'Network Error',
  'FileError': 'File Error',
  'GGUFParsingError': 'Model Format Error',
  'ModelCompatibilityError': 'Model Compatibility Error',
  'VFSError': 'Virtual Filesystem Error',
  'WasmError': 'WebAssembly Error',
  'CacheError': 'Cache Storage Error',
  'OperationCancelledError': 'Operation Cancelled',
  'ModelInitializationError': 'Model Initialization Error',
};

/**
 * Get a human-readable description of a loading stage
 * @param stage The loading stage to describe
 * @param errorType Optional error type for more specific error descriptions
 * @returns A user-friendly description of the stage
 */
export function getStageDescription(stage: LoadingStage, errorType?: string): string {
  switch (stage) {
    case LoadingStage.DOWNLOADING_FROM_SOURCE:
      return 'Downloading model from source';
    case LoadingStage.READING_FROM_FILE:
      return 'Reading model from file';
    case LoadingStage.PREPARING_MODEL_DATA:
      return 'Preparing model data';
    case LoadingStage.VFS_WRITE_START:
      return 'Starting to write model to virtual filesystem';
    case LoadingStage.VFS_WRITE_PROGRESS:
      return 'Writing model to virtual filesystem';
    case LoadingStage.VFS_WRITE_COMPLETE:
      return 'Model written to virtual filesystem';
    case LoadingStage.METADATA_PARSE_START:
      return 'Starting to parse model metadata';
    case LoadingStage.METADATA_PARSE_COMPLETE:
      return 'Model metadata parsed successfully';
    case LoadingStage.MODEL_INITIALIZATION_START:
      return 'Starting model initialization';
    case LoadingStage.MODEL_READY:
      return 'Model ready for use';
    case LoadingStage.ERROR:
      // If errorType is provided, use it for a more specific description
      if (errorType && errorType in ERROR_TYPE_DESCRIPTIONS) {
        return `Error: ${ERROR_TYPE_DESCRIPTIONS[errorType]}`;
      }
      return 'Error loading model';
    case LoadingStage.CANCELLED:
      return 'Model loading cancelled by user';
    case LoadingStage.MODEL_CACHE_CHECK_COMPLETE:
      return 'Model cache check complete';
    case LoadingStage.MODEL_FETCH_START:
      return 'Starting to fetch model';
    case LoadingStage.MODEL_FETCH_PROGRESS:
      return 'Fetching model progress';
    case LoadingStage.MODEL_FETCH_COMPLETE:
      return 'Model fetch complete';
    case LoadingStage.MODEL_DECOMPRESSION_START:
      return 'Starting model decompression';
    case LoadingStage.MODEL_DECOMPRESSION_PROGRESS:
      return 'Model decompression progress';
    case LoadingStage.MODEL_DECOMPRESSION_COMPLETE:
      return 'Model decompression complete';
    case LoadingStage.METADATA_PARSE_PROGRESS:
      return 'Parsing model metadata progress';
    case LoadingStage.EMBIND_MODULE_LOAD_START:
      return 'Starting to load embind module';
    case LoadingStage.EMBIND_MODULE_LOAD_PROGRESS:
      return 'Loading embind module progress';
    case LoadingStage.EMBIND_MODULE_LOAD_COMPLETE:
      return 'Loaded embind module';
    case LoadingStage.MODEL_INITIALIZATION_PROGRESS:
      return 'Model initialization progress';
    case LoadingStage.MODEL_INITIALIZATION_COMPLETE:
      return 'Model initialization complete';
    case LoadingStage.IDLE:
      return 'Runner is idle';
    default:
      return `Unknown stage: ${stage}`;
  }
}

/**
 * Get a more detailed explanation of a specific error type
 * @param errorType The type of error (name of error class)
 * @returns A user-friendly explanation of the error
 */
export function getErrorExplanation(errorType: string): string {
  switch (errorType) {
    case 'NetworkError':
      return 'There was a problem downloading the model. Check your internet connection and verify the URL is correct.';
      
    case 'FileError':
      return 'There was a problem reading the model file. Make sure the file is not corrupted and is a valid GGUF model file.';
      
    case 'GGUFParsingError':
      return 'The model file appears to be an invalid or corrupted GGUF file. Ensure you are using a properly formatted GGUF model.';
      
    case 'ModelCompatibilityError':
      return 'The model is not compatible with this version of the application. It may use an unsupported GGUF format version.';
      
    case 'CacheError':
      return 'There was a problem storing or retrieving the model from browser storage. Your browser may have limited storage space or restrictions.';
      
    case 'VFSError':
      return 'There was a problem with the virtual filesystem used to store the model in memory. This could be due to memory limitations.';
      
    case 'WasmError':
      return 'There was a problem with the WebAssembly module. This could be due to browser compatibility issues or memory constraints.';
      
    case 'OperationCancelledError':
      return 'The operation was cancelled by the user.';
      
    case 'ModelInitializationError':
      return 'There was a problem initializing the model. This could be due to insufficient memory or incompatible model format.';
      
    default:
      return 'An unexpected error occurred while loading or running the model.';
  }
} 