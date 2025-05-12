import { ModelSpecification } from './model-spec.js'; // Import the new interface

// Database name and version
const DB_NAME = 'llama-wasm-models';
const DB_VERSION = 3; // Incremented version due to schema changes
const METADATA_STORE_NAME = 'modelMetadata';
const CHUNK_STORE_NAME = 'modelChunks';

const DEFAULT_CHUNK_SIZE_BYTES = 16 * 1024 * 1024; // 16MB chunks
const DEFAULT_MAX_CACHE_SIZE_BYTES = 512 * 1024 * 1024; // 512MB max cache size

/**
 * Describes the metadata specifically for managing a model's entry in the cache,
 * such as its chunking information and access timestamps.
 */
interface CacheEntryMetadata { // Renamed from ModelMetadata
  modelId: string;
  totalSize: number;
  chunkCount: number;
  chunkSize: number;
  createdAt: number;
  lastAccessed: number;
  // Optional: User-provided filename or other info related to the cache entry itself
  // These were part of the old ModelMetadata, let's clarify their purpose if they
  // are distinct from ModelSpecification.fileName and ModelSpecification.contentType
  // For now, assuming these might be redundant if ModelSpecification covers them, or specific to cache context.
  // Let's keep them for now to see how they fit with ModelSpecification's provenance.
  fileName?: string; 
  contentType?: string;
}

/**
 * Represents the complete metadata stored for a cached model, combining
 * cache management details with the model's intrinsic specification.
 */
interface CachedModelFullMetadata {
  cacheEntry: CacheEntryMetadata;
  specification?: ModelSpecification; // Model spec might not be available for very old cache entries or if parsing failed
}

interface ModelChunk {
  modelId: string;
  chunkIndex: number;
  data: ArrayBuffer;
}

/**
 * ModelCache class - handles caching models in IndexedDB with chunking and LRU eviction.
 */
export class ModelCache {
  private db: IDBDatabase | null = null;
  private dbReadyPromise: Promise<void>;
  private readonly chunkSize: number;
  private readonly maxCacheSize: number;

  constructor(chunkSizeInBytes?: number, maxCacheSizeInBytes?: number) {
    this.chunkSize = chunkSizeInBytes || DEFAULT_CHUNK_SIZE_BYTES;
    this.maxCacheSize = maxCacheSizeInBytes || DEFAULT_MAX_CACHE_SIZE_BYTES;
    this.dbReadyPromise = this.initDatabase();
  }

  /**
   * Initialize the IndexedDB database for model caching
   */
  private async initDatabase(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (typeof indexedDB === 'undefined') {
        console.warn('IndexedDB is not available. Caching will be disabled.');
        resolve(); 
        return;
      }
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        console.error("Failed to open model cache database:", request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        this.db = request.result;
        const currentTransaction = (event.target as IDBOpenDBRequest).transaction;
        if (!currentTransaction) {
            console.error("Upgrade transaction is null. Cannot proceed with DB schema upgrade.");
            if (event.oldVersion > 0) { 
                reject(new Error("Upgrade transaction was null during onupgradeneeded"));
            }
            return;
        }

        if (event.oldVersion < 2 && this.db.objectStoreNames.contains('models')) {
          console.log("Upgrading database: Deleting old 'models' store.");
          this.db.deleteObjectStore('models');
        }

        if (!this.db.objectStoreNames.contains(METADATA_STORE_NAME)) {
          console.log("Upgrading database: Creating metadata store.");
          // The keyPath for metadata store remains 'modelId' which will be part of CacheEntryMetadata.
          // The stored object will be CachedModelFullMetadata.
          const metadataOS = this.db.createObjectStore(METADATA_STORE_NAME, { keyPath: 'cacheEntry.modelId' });
          metadataOS.createIndex('idx_lastAccessed', 'cacheEntry.lastAccessed', { unique: false });
        }
        if (!this.db.objectStoreNames.contains(CHUNK_STORE_NAME)) {
          console.log("Upgrading database: Creating chunks store.");
          this.db.createObjectStore(CHUNK_STORE_NAME, { keyPath: ['modelId', 'chunkIndex'] });
        }
      };
    });
  }

  private async getDb(): Promise<IDBDatabase> {
    await this.dbReadyPromise;
    if (!this.db) {
      throw new Error('IndexedDB is not available or failed to initialize. Caching is disabled.');
    }
    return this.db;
  }

  /**
   * Get a model from the cache if available. Returns the reassembled model data and its specification.
   * @param modelId Unique identifier for the model
   * @returns A promise that resolves to an object { modelData: ArrayBuffer, specification?: ModelSpecification } or null if not found.
   */
  public async getModelWithSpecificationFromCache(modelId: string): Promise<{ modelData: ArrayBuffer; specification?: ModelSpecification } | null> {
    const db = await this.getDb();

    return new Promise(async (resolve, reject) => {
      try {
        const metadataTransaction = db.transaction(METADATA_STORE_NAME, 'readwrite');
        const metadataStore = metadataTransaction.objectStore(METADATA_STORE_NAME);
        // Query using the modelId which is now nested under cacheEntry.modelId
        const metadataRequest = metadataStore.get(modelId);

        metadataRequest.onsuccess = async () => {
          const fullMetadata: CachedModelFullMetadata | undefined = metadataRequest.result;
          if (!fullMetadata || !fullMetadata.cacheEntry) {
            resolve(null);
            return;
          }

          const cacheEntry = fullMetadata.cacheEntry;
          cacheEntry.lastAccessed = Date.now();
          const updateMetadataRequest = metadataStore.put(fullMetadata); // Put back the whole CachedModelFullMetadata object
          
          updateMetadataRequest.onerror = () => {
            console.warn("Failed to update lastAccessed timestamp for model:", modelId, updateMetadataRequest.error);
          };

          const chunkTransaction = db.transaction(CHUNK_STORE_NAME, 'readonly');
          const chunkStore = chunkTransaction.objectStore(CHUNK_STORE_NAME);
          const chunkRange = IDBKeyRange.bound([modelId, 0], [modelId, Number.MAX_SAFE_INTEGER]);
          const getAllChunksRequest = chunkStore.getAll(chunkRange);

          getAllChunksRequest.onsuccess = () => {
            const chunks: ModelChunk[] = getAllChunksRequest.result;
            if (chunks.length !== cacheEntry.chunkCount) {
              console.error(`Inconsistent chunk count for model ${modelId}. Expected ${cacheEntry.chunkCount}, found ${chunks.length}. Invalidating this cache entry.`);
              this.deleteModel(modelId).catch(err => console.error("Error cleaning up inconsistent model:", err));
              resolve(null);
              return;
            }
            
            chunks.sort((a, b) => a.chunkIndex - b.chunkIndex);

            const reassembledBuffer = new ArrayBuffer(cacheEntry.totalSize);
            const reassembledView = new Uint8Array(reassembledBuffer);
            let offset = 0;
            for (const chunk of chunks) {
              reassembledView.set(new Uint8Array(chunk.data), offset);
              offset += chunk.data.byteLength;
            }
            resolve({ modelData: reassembledBuffer, specification: fullMetadata.specification });
          };

          getAllChunksRequest.onerror = () => {
            console.error("Error fetching model chunks from cache for model:", modelId, getAllChunksRequest.error);
            reject(getAllChunksRequest.error);
          };
        };

        metadataRequest.onerror = () => {
          console.error("Error fetching model metadata from cache for model:", modelId, metadataRequest.error);
          reject(metadataRequest.error);
        };
      } catch (error) {
        console.error("Error accessing cache for getModelFromCache:", error);
        reject(error);
      }
    });
  }

  /**
   * For LlamaRunner, which might only need the ArrayBuffer initially.
   * Internally calls getModelWithSpecificationFromCache and returns only the data part.
   */
  public async getModelFromCache(modelId: string): Promise<ArrayBuffer | null> {
    const result = await this.getModelWithSpecificationFromCache(modelId);
    return result ? result.modelData : null;
  }

  /**
   * Cache a model in IndexedDB along with its specification.
   * @param modelId Unique identifier for the model
   * @param modelData The model data as ArrayBuffer
   * @param specification The model's parsed specification (optional for now, but will be required later)
   * @param originalFileName Optional: original filename (can be part of specification.fileName)
   * @param originalContentType Optional: original content type (can be part of specification.contentType)
   */
  public async cacheModel(
    modelId: string, 
    modelData: ArrayBuffer,
    specification?: ModelSpecification, // Added parameter
    originalFileName?: string, // Kept for now, assess redundancy with specification.fileName
    originalContentType?: string // Kept for now, assess redundancy
  ): Promise<void> {
    const db = await this.getDb();

    return new Promise(async (resolve, reject) => {
      let transaction: IDBTransaction | null = null;
      try {
        await this.deleteModel(modelId, db); 
        
        const totalSize = modelData.byteLength;
        const chunkCount = Math.ceil(totalSize / this.chunkSize);
        const now = Date.now();

        const cacheEntryMeta: CacheEntryMetadata = {
          modelId,
          totalSize,
          chunkCount,
          chunkSize: this.chunkSize,
          createdAt: now,
          lastAccessed: now,
          fileName: originalFileName, // Use if provided, or it might come from specification
          contentType: originalContentType, // Use if provided
        };

        const fullMetadata: CachedModelFullMetadata = {
          cacheEntry: cacheEntryMeta,
          specification: specification, // Store the provided specification
        };

        transaction = db.transaction([METADATA_STORE_NAME, CHUNK_STORE_NAME], 'readwrite');
        const metadataStore = transaction.objectStore(METADATA_STORE_NAME);
        const chunkStore = transaction.objectStore(CHUNK_STORE_NAME);

        // The key for the metadata store is modelId, which is part of cacheEntryMeta.
        // The object stored is fullMetadata.
        console.log("Attempting to cache metadata:", JSON.stringify(fullMetadata, null, 2)); // Log the object
        const putMetadataRequest = metadataStore.put(fullMetadata);
        putMetadataRequest.onerror = () => {
            console.error("Error caching model metadata:", modelId, putMetadataRequest.error);
            if (transaction) transaction.abort(); 
            reject(putMetadataRequest.error);
            return;
        };
        
        let chunkPromises: Promise<void>[] = [];
        for (let i = 0; i < chunkCount; i++) {
          const start = i * this.chunkSize;
          const end = Math.min(start + this.chunkSize, totalSize);
          const chunkData = modelData.slice(start, end);

          const chunk: ModelChunk = {
            modelId, // This modelId must match cacheEntry.modelId for consistency
            chunkIndex: i,
            data: chunkData,
          };
          
          chunkPromises.push(new Promise((resChunk, rejChunk) => {
            const putChunkRequest = chunkStore.put(chunk);
            putChunkRequest.onsuccess = () => resChunk();
            putChunkRequest.onerror = () => {
                console.error(`Error caching chunk ${i} for model ${modelId}:`, putChunkRequest.error);
                rejChunk(putChunkRequest.error);
            };
          }));
        }

        await Promise.all(chunkPromises).catch(err => {
            console.error("Error during chunk writing, aborting transaction for model:", modelId, err);
            if (transaction) transaction.abort();
            reject(err);
            throw err; 
        });

        transaction.oncomplete = async () => {
          await this.evictModelsIfNeeded(db);
          resolve();
        };
        transaction.onerror = () => {
          console.error("Error in cacheModel transaction:", modelId, transaction?.error);
          reject(transaction?.error);
        };

      } catch (error) {
        console.error("Error writing to cache for cacheModel:", error);
        if (transaction && transaction.error === null) { 
            try {
                transaction.abort();
            } catch (abortError) {
                console.error("Error aborting transaction in outer catch:", abortError);
            }
        }
        reject(error);
      }
    });
  }
  
  /**
   * Deletes a specific model from the cache
   * @param modelId The ID of the model to delete
   * @param dbInstance Optional database instance, useful when already in a transaction
   */
  public async deleteModel(modelId: string, dbInstance?: IDBDatabase): Promise<void> {
    const db = dbInstance || await this.getDb();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction([METADATA_STORE_NAME, CHUNK_STORE_NAME], 'readwrite');
        const metadataStore = transaction.objectStore(METADATA_STORE_NAME);
        const chunkStore = transaction.objectStore(CHUNK_STORE_NAME);

        // When deleting, modelId is the key for the metadata store (which points to cacheEntry.modelId)
        const deleteMetadataRequest = metadataStore.delete(modelId);
        deleteMetadataRequest.onerror = () => {
            console.warn(`Failed to delete metadata for ${modelId}:`, deleteMetadataRequest.error);
        };

        const chunkRange = IDBKeyRange.bound([modelId, 0], [modelId, Number.MAX_SAFE_INTEGER]);
        const cursorRequest = chunkStore.openCursor(chunkRange);
        let deleteChunkPromises: Promise<void>[] = [];

        cursorRequest.onsuccess = (event) => {
            const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
            if (cursor) {
                deleteChunkPromises.push(new Promise((res, rej) => {
                    const deleteRequest = cursor.delete();
                    deleteRequest.onsuccess = () => res();
                    deleteRequest.onerror = () => {
                        console.warn(`Failed to delete chunk for ${modelId}:`, deleteRequest.error);
                        rej(deleteRequest.error); 
                    };
                }));
                cursor.continue();
            } else {
                Promise.all(deleteChunkPromises).then(() => {
                }).catch(err => {
                    console.warn(`Some chunks for ${modelId} might not have been deleted:`, err);
                });
            }
        };
        cursorRequest.onerror = () => {
            console.error(`Error opening cursor to delete chunks for ${modelId}:`, cursorRequest.error);
        };
        
        transaction.oncomplete = () => {
            resolve();
        };
        transaction.onerror = () => {
            console.error(`Error in deleteModel transaction for ${modelId}:`, transaction.error);
            reject(transaction.error);
        };
    });
  }

  private async evictModelsIfNeeded(dbInstance?: IDBDatabase): Promise<void> {
    const db = dbInstance || await this.getDb();
    
    return new Promise(async (resolve, reject) => {
        const transaction = db.transaction(METADATA_STORE_NAME, 'readonly'); 
        const metadataStore = transaction.objectStore(METADATA_STORE_NAME);
        const getAllMetadataRequest = metadataStore.getAll(); // This will get all CachedModelFullMetadata objects

        getAllMetadataRequest.onsuccess = async () => {
            const allFullMetadata: CachedModelFullMetadata[] = getAllMetadataRequest.result;
            // Sum totalSize from the cacheEntry part of each full metadata object
            let currentCacheSize = allFullMetadata.reduce((sum, meta) => sum + (meta.cacheEntry ? meta.cacheEntry.totalSize : 0), 0);

            if (currentCacheSize <= this.maxCacheSize) {
                resolve();
                return;
            }

            if (allFullMetadata.length === 1 && currentCacheSize > this.maxCacheSize) {
                console.log(`Cache contains a single model of size ${currentCacheSize / (1024*1024)}MB which exceeds max cache size of ${this.maxCacheSize / (1024*1024)}MB. Allowing it to stay.`);
                resolve();
                return;
            }
            
            console.log(`Cache size ${currentCacheSize / (1024*1024)}MB exceeds max ${this.maxCacheSize / (1024*1024)}MB. Evicting models.`);

            // Sort by lastAccessed (oldest first) from cacheEntry for LRU eviction
            allFullMetadata.sort((a, b) => (a.cacheEntry?.lastAccessed || 0) - (b.cacheEntry?.lastAccessed || 0));

            for (const meta of allFullMetadata) {
                if (currentCacheSize <= this.maxCacheSize || !meta.cacheEntry) {
                    break;
                }
                console.log(`Evicting model: ${meta.cacheEntry.modelId}, size: ${meta.cacheEntry.totalSize / (1024*1024)}MB, last accessed: ${new Date(meta.cacheEntry.lastAccessed).toISOString()}`);
                try {
                    await this.deleteModel(meta.cacheEntry.modelId, db); 
                    currentCacheSize -= meta.cacheEntry.totalSize;
                } catch (error) {
                    console.error(`Error evicting model ${meta.cacheEntry.modelId}:`, error);
                }
            }
            resolve();
        };
        getAllMetadataRequest.onerror = () => {
            console.error("Error fetching all metadata for eviction check:", getAllMetadataRequest.error);
            reject(getAllMetadataRequest.error);
        };
    });
  }

  /**
   * Checks if a model is already cached.
   * @param modelId Unique identifier for the model.
   * @returns Promise<boolean> True if the model is cached, false otherwise.
   */
  public async isModelCached(modelId: string): Promise<boolean> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      try {
        const transaction = db.transaction(METADATA_STORE_NAME, 'readonly');
        const store = transaction.objectStore(METADATA_STORE_NAME);
        // Querying by modelId, which is the keyPath cacheEntry.modelId
        const request = store.get(modelId); 

        request.onsuccess = () => {
          resolve(!!request.result);
        };
        request.onerror = () => {
          console.error("Error checking if model is cached:", modelId, request.error);
          reject(request.error);
        };
      } catch (error) {
        console.error("Error accessing cache for isModelCached:", error);
        reject(error);
      }
    });
  }

  /**
   * Clears the entire model cache.
   * @returns Promise<void>
   */
  public async clearCache(): Promise<void> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      try {
        const transaction = db.transaction([METADATA_STORE_NAME, CHUNK_STORE_NAME], 'readwrite');
        const metadataStore = transaction.objectStore(METADATA_STORE_NAME);
        const chunkStore = transaction.objectStore(CHUNK_STORE_NAME);

        const clearMetadataRequest = metadataStore.clear();
        const clearChunksRequest = chunkStore.clear();
        
        let metadataCleared = false;
        let chunksCleared = false;

        clearMetadataRequest.onsuccess = () => {
            metadataCleared = true;
            if (chunksCleared) resolve();
        };
        clearMetadataRequest.onerror = (event) => {
          console.error("Error clearing metadata store:", (event.target as IDBRequest)?.error);
        };

        clearChunksRequest.onsuccess = () => {
            chunksCleared = true;
            if (metadataCleared) resolve();
        };
        clearChunksRequest.onerror = (event) => {
          console.error("Error clearing chunks store:", (event.target as IDBRequest)?.error);
        };
        
        transaction.oncomplete = () => {
            if (metadataCleared && chunksCleared) {
                resolve();
            } else {
                reject(new Error("Cache clearing failed for one or more stores. Check console for details."));
            }
        };
        transaction.onerror = (event) => {
          console.error("Error in clearCache transaction:", (event.target as IDBTransaction)?.error);
          reject((event.target as IDBTransaction)?.error || new Error("Unknown error clearing cache"));
        };

      } catch (error) {
        console.error("Error initiating cache clearing:", error);
        reject(error);
      }
    });
  }
} 