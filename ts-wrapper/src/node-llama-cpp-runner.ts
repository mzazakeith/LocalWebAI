import * as path from 'path';
import {
    getLlama,
    Llama,
    LlamaModel,
    LlamaContext,
    LlamaChatSession,
    LlamaLogLevel,
    LlamaGrammar,
    Token,
    type LlamaOptions,
    type LlamaModelOptions,
    type LlamaContextOptions,
    type LlamaChatSessionOptions,
    type GgufFileInfo,
    type LLamaChatPromptOptions,
    type LlamaChatSessionRepeatPenalty
} from "@node-llama-cpp/index";
import type { ModelSpecification } from './model-spec.js';
import type { ProgressCallback, ProgressInfo } from './loading-progress.js';
import { LoadingStage, getStageDescription } from './loading-progress.js';
import {
    LocalWebAIError,
    ModelCompatibilityError,
    ModelInitializationError,
    OperationCancelledError,
    classifyError
} from './errors.js';
import type { TokenCallback, CompletionCallback, GenerateTextParams } from './llama-runner.js'; // Assuming these are exported or defined here

// Define NodeModelLoadParams based on LlamaModelOptions and ts-wrapper needs
export interface NodeModelLoadParams {
  modelPath: string; // Main argument for loadModel, but also here for consistency if passed in options object
  gpuLayers?: number;
  // Add other LlamaModelOptions as needed for ts-wrapper abstraction
  // Example: vocabOnly, useMmap, useMlock, etc.
  // logLevel is handled by LlamaOptions passed to the constructor for getLlama
}

export class NodeJsLlamaCppRunner {
    private llama: Llama | null = null;
    private model: LlamaModel | null = null;
    private context: LlamaContext | null = null;
    private chatSession: LlamaChatSession | null = null;

    private isLoadingModel: boolean = false;
    private currentModelSpec: ModelSpecification | null = null;
    private currentProgressCallback: ProgressCallback | null = null;
    private lastProgressInfo: ProgressInfo | null = null;
    private activeAbortController: AbortController | null = null;

    private llamaOptions: LlamaOptions | undefined;

    constructor(llamaOptions?: LlamaOptions) {
        this.llamaOptions = llamaOptions;
        this.logInfo('NodeJsLlamaCppRunner initialized.');
    }

    private logInfo(message: string, details?: any) {
        console.log(`[NodeJsLlamaCppRunner INFO] ${message}`, details || '');
    }

    private logError(message: string, error?: any) {
        console.error(`[NodeJsLlamaCppRunner ERROR] ${message}`, error || '');
    }

    private reportProgress(info: Partial<ProgressInfo>): void {
        if (!this.currentProgressCallback) return;

        const fullInfo: ProgressInfo = {
            stage: info.stage || (this.lastProgressInfo?.stage || LoadingStage.PREPARING_MODEL_DATA),
            message: info.message || getStageDescription(info.stage || (this.lastProgressInfo?.stage || LoadingStage.PREPARING_MODEL_DATA)),
            loaded: info.loaded,
            total: info.total,
            metadata: info.metadata || this.currentModelSpec || undefined,
            error: info.error,
            ...info // Allow overriding any field
        };

        this.lastProgressInfo = fullInfo;
        this.currentProgressCallback(fullInfo);
    }

    public async loadModel(
        modelPath: string,
        loadParams?: Partial<NodeModelLoadParams & { progressCallback?: ProgressCallback; signal?: AbortSignal }>
    ): Promise<void> {
        if (this.isLoadingModel) {
            throw new ModelInitializationError('Another model is already being loaded.');
        }
        if (this.model) {
            this.logInfo('A model is already loaded. Terminating existing model before loading a new one.');
            await this.terminate(); // Ensure cleanup before loading a new model
        }

        this.isLoadingModel = true;
        this.currentProgressCallback = loadParams?.progressCallback || null;
        // Corrected AbortController handling:
        // If a signal is provided, we don't create a new controller for it.
        // We only create a new controller if no signal is provided.
        const isExternalSignal = !!loadParams?.signal;
        this.activeAbortController = isExternalSignal ? null : new AbortController();
        const signal = loadParams?.signal || this.activeAbortController!.signal;

        const handleAbort = () => {
            if (this.isLoadingModel) {
                this.isLoadingModel = false;
                const error = new OperationCancelledError('Model loading aborted by user.');
                this.logInfo('Model loading aborted.');
                this.reportProgress({ stage: LoadingStage.CANCELLED, message: error.message, error: error.message });
                // Rejection should be handled by the caller of loadModel
                // Consider how to propagate this if loadModel is internally chained.
            }
        };

        if (signal?.aborted) {
            handleAbort();
            throw new OperationCancelledError('Model loading aborted before starting.');
        }
        signal?.addEventListener('abort', handleAbort, { once: true });

        try {
            this.reportProgress({ stage: LoadingStage.PREPARING_MODEL_DATA, message: 'Initializing Llama instance...' });
            this.logInfo(`Attempting to load model from path: ${modelPath}`);

            this.llama = await getLlama(this.llamaOptions);
            this.reportProgress({ stage: LoadingStage.MODEL_INITIALIZATION_START, message: 'Llama instance initialized. Loading model...' });

            const absoluteModelPath = path.resolve(modelPath);
            const modelOptions: LlamaModelOptions = {
                modelPath: absoluteModelPath,
                gpuLayers: loadParams?.gpuLayers
            };
            
            this.logInfo('LlamaModelOptions prepared:', modelOptions);
            this.model = await this.llama.loadModel(modelOptions);

            if (signal?.aborted) throw new OperationCancelledError('Model loading aborted after loadModel call.');
            this.logInfo('Model loaded successfully into LlamaModel.');

            this.reportProgress({ stage: LoadingStage.METADATA_PARSE_START, message: 'Fetching model metadata...' });
            // Corrected to use fileInfo property
            const fileInfo = this.model.fileInfo; 
            if (fileInfo && fileInfo.metadata) {
                // Map GgufMetadata/GgufFileInfo to ModelSpecification
                this.currentModelSpec = this.mapGgufFileInfoToModelSpecification(fileInfo, absoluteModelPath);
                this.logInfo('Model metadata fetched and mapped:', this.currentModelSpec);
                this.reportProgress({ stage: LoadingStage.METADATA_PARSE_COMPLETE, metadata: this.currentModelSpec });
            } else {
                this.logError('Failed to retrieve GGUF metadata from model.');
                this.reportProgress({ stage: LoadingStage.METADATA_PARSE_COMPLETE, message: 'Failed to retrieve GGUF metadata.' });
            }
            
            if (signal?.aborted) throw new OperationCancelledError('Model loading aborted after metadata parsing.');

            // For chat, we'd create context and session
            const contextOptions: LlamaContextOptions = {
                 // Example: set context size from GenerateTextParams or a default
                contextSize: 2048 // Default, should be configurable
            };
            this.context = await this.model.createContext(contextOptions);
            this.logInfo('LlamaContext created.');

            const chatSessionOptions: LlamaChatSessionOptions = {
                contextSequence: this.context.getSequence()
                // systemPrompt: "You are a helpful assistant." // Optional: make configurable
            };
            this.chatSession = new LlamaChatSession(chatSessionOptions);
            this.logInfo('LlamaChatSession created.');

            this.reportProgress({ stage: LoadingStage.MODEL_READY, message: 'Model is ready for use.' });
            this.isLoadingModel = false;

        } catch (error: any) {
            this.isLoadingModel = false;
            const classifiedError = error instanceof LocalWebAIError ? error : classifyError(error);
            this.logError('Error during model loading:', classifiedError);
            this.reportProgress({
                stage: LoadingStage.ERROR,
                message: classifiedError.message,
                error: classifiedError.message
            });
            await this.terminate(); // Clean up on error
            throw classifiedError;
        } finally {
            signal?.removeEventListener('abort', handleAbort);
            // Only nullify activeAbortController if it was created internally and is the one associated with the signal
            if (!isExternalSignal && this.activeAbortController && signal === this.activeAbortController.signal) {
                this.activeAbortController = null;
            }
        }
    }
    
    private mapGgufFileInfoToModelSpecification(fileInfo: GgufFileInfo, modelPath: string): ModelSpecification {
        const metadata = fileInfo.metadata; // GgufMetadata object
        const archMetadata = fileInfo.architectureMetadata; // Architecture-specific merged metadata

        const spec: ModelSpecification = {
            ggufVersion: fileInfo.version,
            sourceURL: path.isAbsolute(modelPath) ? undefined : modelPath,
            fileName: path.basename(modelPath),
        };

        if (metadata.general) { // Check if general metadata exists
            spec.architecture = metadata.general.architecture;
            spec.modelName = metadata.general.name;
        }
        
        // Use archMetadata for architecture-specific fields
        if (archMetadata) {
            if (typeof archMetadata.context_length === 'number') spec.contextLength = archMetadata.context_length;
            if (typeof archMetadata.embedding_length === 'number') spec.embeddingLength = archMetadata.embedding_length;
            if (typeof archMetadata.block_count === 'number') spec.layerCount = archMetadata.block_count;

            if (archMetadata.attention) {
                if (typeof archMetadata.attention.head_count === 'number') spec.headCount = archMetadata.attention.head_count;
                if (typeof archMetadata.attention.head_count_kv === 'number') spec.headCountKv = archMetadata.attention.head_count_kv;
            }
        }
        
        // We might need to query file system for fileSize if GgufFileInfo doesn't provide it.
        // For now, this is a basic mapping.
        return spec;
    }

    public getModelMetadata(): ModelSpecification | null {
        if (!this.model || !this.currentModelSpec) {
            this.logInfo("getModelMetadata called, but no model is loaded or metadata is not available.");
            return null;
        }
        return this.currentModelSpec;
    }

    public async generateText(
        promptText: string,
        generateParams: GenerateTextParams, // Assuming GenerateTextParams is defined in ./llama-runner.ts
        tokenCallback: TokenCallback,      // Assuming TokenCallback is (tokens: Token[]) => void from ./llama-runner.ts
        completionCallback: CompletionCallback // Assuming CompletionCallback is (error: Error | null, result?: string) => void from ./llama-runner.ts
    ): Promise<void> {
        if (!this.chatSession || !this.model) {
            const error = new ModelInitializationError("Model is not loaded. Call loadModel() first.");
            this.logError(error.message);
            throw error;
        }
        if (this.isLoadingModel) {
            const error = new ModelInitializationError("Model is currently being loaded. Please wait.");
            this.logError(error.message);
            throw error;
        }

        const currentAbortController = new AbortController();
        const signal = currentAbortController.signal;

        try {
            this.logInfo("Starting text generation with prompt:", promptText);
            this.logInfo("Generation parameters (from llama-runner.ts GenerateTextParams):", generateParams);

            const promptOptions: LLamaChatPromptOptions = {};

            if (generateParams.n_predict !== undefined) promptOptions.maxTokens = generateParams.n_predict;
            if (generateParams.temp !== undefined) promptOptions.temperature = generateParams.temp;
            if (generateParams.top_k !== undefined) promptOptions.topK = generateParams.top_k;
            if (generateParams.top_p !== undefined) promptOptions.topP = generateParams.top_p;
            promptOptions.signal = signal;
            promptOptions.stopOnAbortSignal = true; 
            promptOptions.onTextChunk = (textChunk: string) => {
                try {
                    tokenCallback(textChunk);
                } catch (e: any) {
                    this.logError("Error in tokenCallback:", e);
                }
            };
            
            this.logInfo("Mapped promptOptions for node-llama-cpp:", promptOptions);

            const response = await this.chatSession.promptWithMeta(promptText, promptOptions);
            const fullResponseText = response.response;

            this.logInfo("Text generation completed. Full response:", fullResponseText);
            completionCallback();

        } catch (error: any) {
            const classifiedError = error instanceof LocalWebAIError ? error : classifyError(error);
            this.logError("Error during text generation:", classifiedError);
            throw classifiedError;
        } finally {
            // currentAbortController is function-scoped, no specific cleanup needed here for it unless it was stored on `this`
        }
    }

    // Stub for terminate
    public async terminate(): Promise<void> {
        this.logInfo('terminate() called. Cleaning up resources.');
        if (this.activeAbortController) {
            this.activeAbortController.abort();
            this.activeAbortController = null;
        }
        // Actual llama.cpp resource cleanup will be more involved
        if (this.chatSession) {
            // Assuming LlamaChatSession might have a dispose or cleanup method
            // this.chatSession.dispose(); 
            this.chatSession = null;
        }
        if (this.context) {
            await this.context.dispose();
            this.context = null;
        }
        if (this.model) {
            await this.model.dispose();
            this.model = null;
        }
        if (this.llama) {
            // Llama instance itself doesn't have a dispose, it's a singleton manager
            this.llama = null; 
        }
        this.isLoadingModel = false;
        this.currentModelSpec = null;
        this.currentProgressCallback = null;
        this.lastProgressInfo = null;
        this.reportProgress({ stage: LoadingStage.IDLE, message: "Runner terminated and idle."});
        this.logInfo('NodeJsLlamaCppRunner resources cleaned up.');
    }

    // getModelMetadata, generateText, cancelLoading methods to be implemented next
} 