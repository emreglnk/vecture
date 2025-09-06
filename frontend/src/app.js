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
            const response = await fetch(`http://localhost:8000/fetch_url`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch content: ${response.statusText}`);
            }

            const data = await response.json();
            return {
                content: data.content,
                title: data.title,
                description: data.description || '',
                author: data.author || '',
                keywords: data.keywords || []
            };
        } catch (error) {
            throw new Error(`URL fetch failed: ${error.message}`);
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

            // Show success result
            resultDiv.innerHTML = `
                <div class="result success">
                    <h4>Content Uploaded & NFT Minted Successfully!</h4>
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
            
            // Validate inputs
            const query = APIUtils.validateSearchQuery(searchInput.value);
            const topK = parseInt(topKInput.value) || 5;
            
            this.showLoading('Searching vectors...');
            
            // Perform search
            const searchResults = await apiClient.searchVectors(query, topK);
            const formattedResults = APIUtils.formatSearchResults(searchResults);
            
            // Display results
            if (formattedResults.length === 0) {
                resultsDiv.innerHTML = `
                    <div class="result">
                        <p>No similar vectors found for your query.</p>
                    </div>
                `;
            } else {
                const resultsHTML = formattedResults.map(result => `
                    <div class="search-result">
                        <h4>Result ${result.id + 1}</h4>
                        <div class="score">Similarity Score: ${(result.score * 100).toFixed(2)}%</div>
                        <div class="content">${result.text}</div>
                        ${Object.keys(result.metadata).length > 0 ? 
                            `<div class="metadata">Metadata: ${APIUtils.formatMetadata(result.metadata)}</div>` : 
                            ''
                        }
                    </div>
                `).join('');
                
                resultsDiv.innerHTML = `
                    <h3>Search Results (${formattedResults.length} found)</h3>
                    ${resultsHTML}
                `;
            }
            
            this.showNotification(`Found ${formattedResults.length} similar vectors`, 'success');
            
        } catch (error) {
            const resultsDiv = document.getElementById('search-results');
            resultsDiv.innerHTML = `
                <div class="result error">
                    <h4>Search Failed</h4>
                    <p>${APIUtils.formatError(error)}</p>
                </div>
            `;
            this.showNotification(APIUtils.formatError(error), 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async loadIndexStats() {
        try {
            // Set default values for now since stats endpoint may not be implemented
            document.getElementById('total-vectors').textContent = '0';
            document.getElementById('index-size').textContent = '0 MB';
        } catch (error) {
            console.error('Failed to load index stats:', error);
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