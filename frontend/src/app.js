// Main Application Logic

console.log("Here");

class VectorMVPApp {
    constructor() {
        this.currentTab = 'upload';
        this.isLoading = false;
        
        // Initialize ethers provider
        this.provider = null;
        this.signer = null;
        this.contract = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadIndexStats();
        this.checkBackendHealth();
    }
    
    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });
        
        // URL input validation
        document.getElementById('url-input').addEventListener('input', () => {
            this.validateUrlInput();
        });
        
        // Wallet connection
        document.getElementById('connect-wallet').addEventListener('click', () => {
            this.connectWallet();
        });
        
        document.getElementById('disconnect-wallet').addEventListener('click', () => {
            this.disconnectWallet();
        });
        
        // Upload form
        document.getElementById('upload-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleUpload();
        });
        
        // Search form
        document.getElementById('search-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleSearch();
        });
        
        // RAG form
        document.getElementById('rag-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleRAG();
        });
        
        // Management actions
        document.getElementById('refresh-stats').addEventListener('click', () => {
            this.loadIndexStats();
        });
        
        document.getElementById('clear-index').addEventListener('click', () => {
            this.clearIndex();
        });
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        this.currentTab = tabName;
        
        // Load data for specific tabs
        if (tabName === 'manage') {
            this.loadIndexStats();
        } else if (tabName === 'rag') {
            // Clear form when switching to RAG tab
            document.getElementById('rag-form').reset();
            document.getElementById('rag-output').classList.add('hidden');
        }
    }
    
    async connectWallet() {
        try {
            this.showLoading('Connecting wallet...');
            await walletManager.connect();
            this.showNotification('Wallet connected successfully!', 'success');
        } catch (error) {
            this.showNotification(error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    disconnectWallet() {
        walletManager.disconnect();
        this.showNotification('Wallet disconnected', 'info');
    }
    
    validateUrlInput() {
        const urlInput = document.getElementById('url-input');
        const url = urlInput.value.trim();
        
        if (url && !this.isValidUrl(url)) {
            urlInput.setCustomValidity('Please enter a valid URL');
        } else {
            urlInput.setCustomValidity('');
        }
    }
    
    isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }

    async fetchContentFromUrl(url) {
        try {
            // Use backend /fetch_url endpoint to properly extract content
            const response = await fetch('http://localhost:8000/fetch_url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Failed to fetch content: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            return {
                title: data.title || 'Extracted Content',
                content: data.content,
                description: data.description || '',
                author: data.author || '',
                keywords: data.keywords || []
            };
        } catch (error) {
            throw new Error(`Content extraction failed: ${error.message}`);
        }
    }

    async handleUpload() {
        try {
            const urlInput = document.getElementById('url-input');
            const resultDiv = document.getElementById('upload-result');
            
            const url = urlInput.value.trim();
            if (!url) {
                throw new Error('Please enter a URL');
            }
            
            this.showLoading('Fetching content from URL...');
            const urlData = await this.fetchContentFromUrl(url);
            const text = urlData.content;
            
            if (!text) {
                throw new Error('No content found at URL');
            }
            
            // Create metadata from URL data
            const metadata = {
                source_url: url,
                title: urlData.title || 'Untitled',
                description: urlData.description || '',
                author: urlData.author || '',
                keywords: urlData.keywords || [],
                extracted_at: new Date().toISOString()
            };
            
            this.showLoading('Processing text...');
            
            // Upload and get manifest
            const uploadResponse = await fetch('http://localhost:8000/upload_and_manifest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: metadata.title,
                    source_url: metadata.source_url,
                    raw_text: text,
                    tags: metadata.keywords || []
                })
            });
            
            if (!uploadResponse.ok) {
                throw new Error(`Upload failed: ${uploadResponse.statusText}`);
            }
            
            const uploadResult = await uploadResponse.json();
            
            console.log(uploadResult);

            this.showLoading('Minting NFT...');
            
            // Check if wallet is connected
            if (!walletManager.isWalletConnected()) {
                throw new Error('Please connect your wallet first');
            }
            
            // Initialize ethers provider and contract if not already done
            if (!this.provider) {
                this.provider = new ethers.providers.Web3Provider(window.ethereum);
                this.signer = this.provider.getSigner();
                this.contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, this.signer);
            }


            
            // Preflight: simulate mint to catch reverts (avoids UNPREDICTABLE_GAS_LIMIT)
            try {
                await this.contract.callStatic.mint(
                    uploadResult.contentHash,
                    uploadResult.vectorHash,
                    uploadResult.manifestCID,
                    uploadResult.sourceUrl
                );
            } catch (e) {
                const reason = e?.error?.message || e?.data || e?.message || 'execution reverted';
                throw new Error(`On-chain validation failed: ${reason}`);
            }

            // Send transaction using dynamic hashes (correct param order)
            const tx = await this.contract.mint(
                uploadResult.contentHash,
                uploadResult.vectorHash,
                uploadResult.manifestCID,
                uploadResult.sourceUrl
            );
            
            console.log(tx);

            const receipt = await tx.wait();

            const txHash = tx.hash;
            // Try to parse tokenId from Minted event
            let tokenId = null;
            try {
                const mintedEvent = receipt.events?.find(e => e.event === 'Minted');
                if (mintedEvent && mintedEvent.args && mintedEvent.args.tokenId) {
                    tokenId = mintedEvent.args.tokenId.toString();
                }
            } catch (_) {}

            // Generate render image URL if vector_url is available
            const renderImageUrl = uploadResult.vector_url ? 
                `http://localhost:8000/render?url=${encodeURIComponent(uploadResult.vector_url)}&label=Token ${tokenId || 'New'}` : null;
            
            // Show success result
            resultDiv.innerHTML = `
                <div class="result success">
                    <h4>Content Uploaded & NFT Minted Successfully!</h4>
                    ${renderImageUrl ? `<div class="render-image" style="text-align: center; margin: 15px 0;"><img src="${renderImageUrl}" alt="Vector Visualization" style="max-width: 300px; height: auto; border-radius: 0px; border: 2px solid #4CAF50;"></div>` : ''}
                    <p><strong>Transaction Hash:</strong> <code>${txHash}</code></p>
                    ${tokenId ? `<p><strong>Token ID:</strong> ${tokenId}</p>` : ''}
                    <p><strong>Content Hash:</strong> <code>${uploadResult.contentHash}</code></p>
                    <p><strong>Vector Hash:</strong> <code>${uploadResult.vectorHash}</code></p>
                    <p><strong>Manifest CID:</strong> <code>${uploadResult.manifestCID}</code></p>
                    <p><strong>Source URL:</strong> ${metadata.source_url}</p>
                    <p><strong>Title:</strong> ${metadata.title}</p>
                </div>
            `;
            resultDiv.classList.remove('hidden');
            
            // Clear form
            urlInput.value = '';
            
            this.showNotification('Content uploaded and indexed successfully!', 'success');
            
        } catch (error) {
            const resultDiv = document.getElementById('upload-result');
            resultDiv.innerHTML = `
                <div class="result error">
                    <h4>Upload Failed</h4>
                    <p>${APIUtils.formatError(error)}</p>
                </div>
            `;
            resultDiv.classList.remove('hidden');
            this.showNotification(APIUtils.formatError(error), 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async handleSearch() {
        try {
            const searchInput = document.getElementById('search-input');
            const topKInput = document.getElementById('top-k');
            const resultsDiv = document.getElementById('search-results');
            
            const query = searchInput.value.trim();
            const topK = parseInt(topKInput.value) || 5;
            
            if (!query) {
                throw new Error('Please enter a search query');
            }
            
            this.showLoading('Searching vectors...');
            
            const response = await fetch('http://localhost:8000/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    topk: topK,
                    min_score: 0.0
                })
            });
            
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Display results
            if (data.hits && data.hits.length > 0) {
                let html = `<h3>Found ${data.hits.length} results:</h3>`;
                data.hits.forEach((hit, index) => {
                    const similarity = (hit.score * 100).toFixed(1);
                    // Generate render image URL for this NFT using embedding_uri from metadata
                    const renderImageUrl = hit.metadata && hit.metadata.embedding_uri ? 
                        `http://localhost:8000/render?url=${encodeURIComponent(hit.metadata.embedding_uri)}&label=Token ${hit.tokenId}` : null;
                    
                    html += `
                        <div class="search-result">
                            <div class="result-header">
                                <span class="result-rank">#${hit.rank}</span>
                                <span class="result-score">${similarity}% match</span>
                            </div>
                            <div class="result-content">
                                ${renderImageUrl ? `<div class="render-image"><img src="${renderImageUrl}" alt="Vector Visualization" style=""></div>` : ''}
                                <strong>Token ID:</strong> ${hit.tokenId}<br>
                                <strong>Chunk ID:</strong> ${hit.chunkId}<br>
                                ${hit.metadata ? `<strong>Title:</strong> ${hit.metadata.title || 'N/A'}<br>` : ''}
                                ${hit.metadata && hit.metadata.source_url ? `<strong>Source:</strong> <a href="${hit.metadata.source_url}" target="_blank">${hit.metadata.source_url}</a>` : ''}
                            </div>
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html;
            } else {
                resultsDiv.innerHTML = '<p>No results found. Try a different query.</p>';
            }
            
            resultsDiv.classList.remove('hidden');
            
        } catch (error) {
            this.showNotification(error.message, 'error');
            console.error('Search error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    async syncAllNFTs() {
        try {
            if (!walletManager.isWalletConnected()) {
                throw new Error('Please connect your wallet first');
            }
            
            const resultDiv = document.getElementById('add-index-result');
            this.showLoading('Scanning contract for all NFTs...');
            
            // Get contract instance
            const contract = walletManager.getContract();
            if (!contract) {
                throw new Error('Contract not available');
            }
            
            // Get total supply of NFTs
            let totalSupply;
            try {
                totalSupply = await contract.totalSupply();
                totalSupply = totalSupply.toNumber();
            } catch (e) {
                // If totalSupply doesn't exist, try to get the next token ID
                try {
                    const nextTokenId = await contract.nextTokenId();
                    totalSupply = nextTokenId.toNumber();
                } catch (e2) {
                    throw new Error('Cannot determine total NFT count from contract');
                }
            }
            
            if (totalSupply === 0) {
                resultDiv.innerHTML = `
                    <div class="info-message">
                        <h3>ℹ️ No NFTs Found</h3>
                        <p>No NFTs have been minted on this contract yet.</p>
                    </div>
                `;
                resultDiv.classList.remove('hidden');
                return;
            }
            
            let successCount = 0;
            let errorCount = 0;
            const errors = [];
            
            // Process each NFT
            for (let tokenId = 1; tokenId < totalSupply; tokenId++) {
                try {
                    this.showLoading(`Processing NFT ${tokenId}/${totalSupply - 1}...`);
                    
                    // Get token URI and metadata
                    const tokenURI = await contract.tokenURI(tokenId);
                    
                    // Fetch metadata from IPFS
                    const metadataResponse = await fetch(tokenURI.replace('ipfs://', 'https://plum-peculiar-pheasant-309.mypinata.cloud/ipfs/'));
                    const metadata = await metadataResponse.json();
                    
                    // Extract embedding URI from metadata
                    const embeddingURI = metadata.embedding?.file?.uri;
                    if (!embeddingURI) {
                        errors.push(`Token ${tokenId}: No embedding URI found in metadata`);
                        errorCount++;
                        continue;
                    }
                    
                    // Add to index via backend
                    const response = await fetch('http://localhost:8000/add_index', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            tokenId: tokenId,
                            embedding_uri: embeddingURI,
                            metadata: {
                                title: metadata.content?.title || `NFT #${tokenId}`,
                                source_url: metadata.content?.source_refs?.[0]?.uri,
                                created_at: metadata.content?.created_at
                            }
                        })
                    });
                    
                    if (response.ok) {
                        successCount++;
                    } else {
                        const errorData = await response.json().catch(() => ({}));
                        errors.push(`Token ${tokenId}: ${errorData.detail || 'Add failed'}`);
                        errorCount++;
                    }
                    
                } catch (error) {
                    errors.push(`Token ${tokenId}: ${error.message}`);
                    errorCount++;
                }
            }
            
            // Display results
            let resultHTML = `
                <div class="sync-results">
                    <h3>🔄 Sync Complete</h3>
                    <div class="stats">
                        <p><strong>Total NFTs Found:</strong> ${totalSupply - 1}</p>
                        <p><strong>Successfully Added:</strong> ${successCount}</p>
                        <p><strong>Errors:</strong> ${errorCount}</p>
                    </div>
            `;
            
            if (errors.length > 0 && errors.length <= 5) {
                resultHTML += `
                    <div class="error-details">
                        <h4>Errors:</h4>
                        <ul>
                            ${errors.map(error => `<li>${error}</li>`).join('')}
                        </ul>
                    </div>
                `;
            } else if (errors.length > 5) {
                resultHTML += `
                    <div class="error-details">
                        <h4>Errors (showing first 5):</h4>
                        <ul>
                            ${errors.slice(0, 5).map(error => `<li>${error}</li>`).join('')}
                        </ul>
                        <p>... and ${errors.length - 5} more errors</p>
                    </div>
                `;
            }
            
            resultHTML += '</div>';
            resultDiv.innerHTML = resultHTML;
            resultDiv.classList.remove('hidden');
            
            // Refresh index stats
            this.loadIndexStats();
            
            this.showNotification(`Sync complete: ${successCount} NFTs added, ${errorCount} errors`, successCount > 0 ? 'success' : 'warning');
            
        } catch (error) {
            const resultDiv = document.getElementById('add-index-result');
            resultDiv.innerHTML = `
                <div class="error-message">
                    <h3>❌ Sync Failed</h3>
                    <p>${error.message}</p>
                </div>
            `;
            resultDiv.classList.remove('hidden');
            
            this.showNotification(error.message, 'error');
            console.error('Sync all NFTs error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    async handleRAG() {
        try {
            const ragInput = document.getElementById('rag-input');
            const ragOutput = document.getElementById('rag-output');
            
            const query = ragInput.value.trim();
            if (!query) {
                throw new Error('Please enter a question');
            }
            
            this.showLoading('Processing your question...');
            
            const response = await fetch('http://localhost:8000/rag', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    topk: 5
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${errorText}`);
            }
            
            const result = await response.json();
            
            let html = `
                <div class="rag-result">
                    <h3>AI Response:</h3>
                    <div class="answer">${result.answer}</div>
                    <div class="context-info">
                        <small>Based on ${result.context_sources || 'multiple'} relevant sources</small>
                    </div>
            `;
            
            // Display context sources with vector visualizations
            if (result.hits && result.hits.length > 0) {
                html += '<div class="context-sources"><h5>Context Sources:</h5>';
                result.hits.forEach((hit, index) => {
                    const metadata = hit.metadata || {};
                    const score = (hit.score * 100).toFixed(1);
                    
                    // Generate render image URL if embedding_uri is available
                    const renderImageUrl = metadata.embedding_uri ? 
                        `http://localhost:8000/render?url=${encodeURIComponent(metadata.embedding_uri)}&label=Source ${index + 1}` : null;
                    
                    html += `
                        <div class="context-item">
                            <strong>Source ${index + 1}</strong> (${score}% relevance)<br>
                            ${renderImageUrl ? `<div class="render-image" style="text-align: center; margin: 8px 0;"><img src="${renderImageUrl}" alt="Vector Visualization" style="max-width: 150px; height: auto; border-radius: 0px; border: 1px solid #FFD700;"></div>` : ''}
                            <em>${metadata.title || 'Untitled'}</em><br>
                            ${hit.text ? hit.text.substring(0, 150) + '...' : 'No preview available'}
                            ${metadata.source_url ? `<br><a href="${metadata.source_url}" target="_blank">View Source</a>` : ''}
                        </div>
                    `;
                });
                html += '</div>';
            }
            
            html += '</div>';
            ragOutput.innerHTML = html;
            ragOutput.classList.remove('hidden');
            
        } catch (error) {
            console.error('RAG error:', error);
            const ragOutput = document.getElementById('rag-output');
            ragOutput.innerHTML = `
                <div class="error-message">
                    <h3>❌ Error</h3>
                    <p>${error.message}</p>
                </div>
            `;
            ragOutput.classList.remove('hidden');
        } finally {
            this.hideLoading();
        }
    }
    
    async handleAddIndex() {
        try {
            const tokenIdInput = document.getElementById('token-id-input');
            const resultDiv = document.getElementById('add-index-result');
            
            const tokenId = parseInt(tokenIdInput.value);
            
            if (!tokenId || tokenId < 1) {
                throw new Error('Please enter a valid Token ID');
            }
            
            if (!walletManager.isWalletConnected()) {
                throw new Error('Please connect your wallet first');
            }
            
            this.showLoading('Adding NFT to index...');
            
            // Get contract instance and fetch metadata
            const contract = walletManager.getContract();
            const tokenURI = await contract.tokenURI(tokenId);
            
            // Fetch metadata from IPFS
            const metadataResponse = await fetch(tokenURI.replace('ipfs://', 'https://plum-peculiar-pheasant-309.mypinata.cloud/ipfs/'));
            const metadata = await metadataResponse.json();
            
            // Extract embedding URI
            const embeddingURI = metadata.embedding?.file?.uri;
            if (!embeddingURI) {
                throw new Error('No embedding URI found in NFT metadata');
            }
            
            const response = await fetch('http://localhost:8000/add_index', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    tokenId: tokenId,
                    embedding_uri: embeddingURI,
                    metadata: {
                        title: metadata.content?.title || `NFT #${tokenId}`,
                        source_url: metadata.content?.source_refs?.[0]?.uri,
                        created_at: metadata.content?.created_at
                    }
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Add to index failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Display success result
            resultDiv.innerHTML = `
                <div class="success-message">
                    <h3>✅ Successfully Added to Index</h3>
                    <p><strong>Token ID:</strong> ${tokenId}</p>
                    <p><strong>Title:</strong> ${metadata.content?.title || 'N/A'}</p>
                    <p><strong>Status:</strong> ${data.message || 'Added successfully'}</p>
                </div>
            `;
            resultDiv.classList.remove('hidden');
            
            // Clear form
            document.getElementById('add-index-form').reset();
            
            // Refresh index stats
            this.loadIndexStats();
            
            this.showNotification('NFT successfully added to search index!', 'success');
            
        } catch (error) {
            const resultDiv = document.getElementById('add-index-result');
            resultDiv.innerHTML = `
                <div class="error-message">
                    <h3>❌ Error</h3>
                    <p>${error.message}</p>
                </div>
            `;
            resultDiv.classList.remove('hidden');
            
            this.showNotification(error.message, 'error');
            console.error('Add to index error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    async loadIndexStats() {
        try {
            const response = await fetch('http://localhost:8000/stats');
            if (!response.ok) {
                throw new Error(`Failed to fetch stats: ${response.statusText}`);
            }
            
            const stats = await response.json();
            document.getElementById('total-vectors').textContent = stats.total_vectors || '0';
            
            // Calculate approximate index size (vectors * dimension * 4 bytes per float)
            const sizeBytes = (stats.total_vectors || 0) * (stats.dimension || 1536) * 4;
            const sizeMB = (sizeBytes / (1024 * 1024)).toFixed(2);
            document.getElementById('index-size').textContent = `${sizeMB} MB`;
            
            console.log('Index stats loaded:', stats);
        } catch (error) {
            console.error('Failed to load index stats:', error);
            // Set default values on error
            document.getElementById('total-vectors').textContent = 'Error';
            document.getElementById('index-size').textContent = 'Error';
        }
    }
    
    async clearIndex() {
        if (!confirm('Are you sure you want to clear the entire vector index? This action cannot be undone.')) {
            return;
        }
        
        try {
            this.showLoading('Clearing index...');
            await apiClient.clearIndex();
            await this.loadIndexStats();
            this.showNotification('Index cleared successfully', 'success');
        } catch (error) {
            this.showNotification(APIUtils.formatError(error), 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async checkBackendHealth() {
        try {
            await apiClient.healthCheck();
            console.log('Backend is healthy');
        } catch (error) {
            this.showNotification('Backend connection failed. Please ensure the backend server is running.', 'error');
            console.error('Backend health check failed:', error);
        }
    }
    
    showLoading(message = 'Loading...') {
        this.isLoading = true;
        const loadingDiv = document.getElementById('loading');
        const loadingText = loadingDiv.querySelector('p');
        loadingText.textContent = message;
        loadingDiv.classList.remove('hidden');
    }
    
    hideLoading() {
        this.isLoading = false;
        document.getElementById('loading').classList.add('hidden');
    }
    
    showNotification(message, type = 'info') {
        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.className = `notification ${type}`;
        notification.classList.remove('hidden');
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            notification.classList.add('hidden');
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new VectorMVPApp();
});