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
  CANCELLED = 'cancelled'
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
 * Get a human-readable description of a loading stage
 */
export function getStageDescription(stage: LoadingStage): string {
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
      return 'Error loading model';
    case LoadingStage.CANCELLED:
      return 'Model loading cancelled by user';
    default:
      return `Unknown stage: ${stage}`;
  }
} 