/**
 * Training Status Component
 * 
 * Shared component for displaying training job status icon and dialog.
 * Can be used on any page that needs to show training status.
 */

class TrainingStatus {
    constructor(options = {}) {
        this.iconContainerId = options.iconContainerId || 'trainingStatusIcon';
        this.modalId = options.modalId || 'trainingStatusModal';
        this.dialogRefreshInterval = options.dialogRefreshInterval || 20000; // 20 seconds
        this.startTime = null;
        this.dialogRefreshIntervalId = null;
        this.durationUpdateIntervalId = null; // For continuous duration updates
        this.baseDurationSeconds = null; // Server-provided duration (base)
        this.baseDurationTimestamp = null; // When we received the base duration
        this.currentEpoch = 0;
        this.totalEpochs = 0;
        this.estimatedRemainingSeconds = null;
        
        this.init();
    }
    
    init() {
        // Check status once on initialization only
        this.checkStatus();
        
        // Close modal when clicking outside
        document.addEventListener('click', (e) => {
            const modal = document.getElementById(this.modalId);
            if (e.target === modal) {
                this.hideDialog();
            }
        });
    }
    
    setStartTime(time) {
        this.startTime = time || new Date();
    }
    
    formatDuration(totalSeconds) {
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${seconds}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds}s`;
        } else {
            return `${seconds}s`;
        }
    }
    
    async checkStatus() {
        try {
            const response = await fetch('/api/ai-training/training-status');
            const status = await response.json();
            
            const iconContainer = document.getElementById(this.iconContainerId);
            if (!iconContainer) return; // Icon container not found on this page
            
            const iconBadge = document.getElementById('statusIconBadge');
            const tooltip = document.getElementById('statusIconTooltip');
            
            if (!iconBadge || !tooltip) return;
            
            if (status.status === 'training') {
                // Show icon with hourglass animation
                iconContainer.style.display = 'block';
                iconBadge.textContent = '⏳';
                iconBadge.style.animation = 'hourglass-flip 1.5s ease-in-out infinite';
                tooltip.textContent = 'Training job in progress - Click to view details';
                
                // Server now provides duration, so we don't need to track start time
                // But keep it as fallback for edge cases
                if (!this.startTime && status.progress && status.progress.start_time) {
                    // Use server-provided start_time if available
                    this.startTime = new Date(status.progress.start_time * 1000);
                }
            } else if (status.status === 'completed') {
                // Show completed icon briefly, then hide
                iconContainer.style.display = 'block';
                iconBadge.textContent = '✅';
                iconBadge.style.animation = 'none';
                tooltip.textContent = 'Training completed successfully!';
                
                // Hide after 5 seconds
                setTimeout(() => {
                    iconContainer.style.display = 'none';
                }, 5000);
            } else if (status.status === 'cancelled') {
                // Show cancelled icon
                iconContainer.style.display = 'block';
                iconBadge.textContent = '⏹️';
                iconBadge.style.animation = 'none';
                tooltip.textContent = 'Training was cancelled';
                
                // Hide after 5 seconds
                setTimeout(() => {
                    iconContainer.style.display = 'none';
                }, 5000);
                this.startTime = null;
            } else if (status.status === 'error') {
                // Show error icon
                iconContainer.style.display = 'block';
                iconBadge.textContent = '❌';
                iconBadge.style.animation = 'pulse 2s infinite';
                tooltip.textContent = 'Training error occurred - Click to view details';
            } else {
                // Hide icon when idle
                iconContainer.style.display = 'none';
                this.startTime = null;
                
                // Stop any dialog refresh if training is not active
                if (this.dialogRefreshIntervalId) {
                    clearInterval(this.dialogRefreshIntervalId);
                    this.dialogRefreshIntervalId = null;
                }
            }
        } catch (error) {
            console.error('Error checking training status:', error);
        }
    }
    
    async showDialog() {
        try {
            const response = await fetch('/api/ai-training/training-status');
            const status = await response.json();
            
            const modal = document.getElementById(this.modalId);
            const content = document.getElementById('trainingStatusContent');
            
            if (!modal || !content) return;
            
            let html = '';
            
            if (status.status === 'training') {
                const progress = status.progress || {};
                const epoch = progress.epoch || 0;
                const totalEpochs = progress.total_epochs || 0;
                const trainLoss = progress.train_loss || 0;
                const valLoss = progress.val_loss || 0;
                const message = progress.message || 'Training...';
                
                // Format duration from server (in seconds) - prefer server value
                let durationText = 'Calculating...';
                if (progress.duration_seconds !== undefined && progress.duration_seconds >= 0) {
                    const totalSeconds = progress.duration_seconds;
                    durationText = this.formatDuration(totalSeconds);
                } else if (this.startTime) {
                    // Fallback to client-side calculation if server doesn't provide it
                    const elapsed = Math.floor((Date.now() - this.startTime.getTime()) / 1000);
                    durationText = this.formatDuration(elapsed);
                }
                
                // Format estimated remaining time from server (in seconds) - prefer server value
                let estimatedTimeText = 'Calculating...';
                if (progress.estimated_remaining_seconds !== undefined && progress.estimated_remaining_seconds > 0) {
                    estimatedTimeText = this.formatDuration(progress.estimated_remaining_seconds);
                } else if (epoch > 0 && totalEpochs > 0 && this.startTime) {
                    // Fallback to client-side calculation
                    const elapsed = (Date.now() - this.startTime.getTime()) / 1000;
                    const avgTimePerEpoch = elapsed / epoch;
                    const remainingEpochs = totalEpochs - epoch;
                    const estimatedSeconds = Math.floor(avgTimePerEpoch * remainingEpochs);
                    estimatedTimeText = this.formatDuration(estimatedSeconds);
                }
                
                // Calculate progress percentage
                const progressPercent = totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0;
                
                html = `
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h2 style="margin: 0 0 10px 0; font-size: 1.5em;">🔄 Training In Progress</h2>
                        <p style="margin: 0; opacity: 0.9;">${message}</p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                            <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Status</div>
                            <div style="font-size: 1.3em; font-weight: 600; color: #667eea;">Training</div>
                        </div>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                            <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Progress</div>
                            <div style="font-size: 1.3em; font-weight: 600; color: #28a745;">${progressPercent}%</div>
                        </div>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                            <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Epoch</div>
                            <div style="font-size: 1.3em; font-weight: 600; color: #333;">${epoch} / ${totalEpochs}</div>
                        </div>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                            <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Duration</div>
                            <div id="trainingDurationDisplay" style="font-size: 1.3em; font-weight: 600; color: #333;">${durationText}</div>
                        </div>
                    </div>
                    
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #ffc107;">
                        <div style="font-weight: 600; margin-bottom: 10px;">⏱️ Estimated Time Remaining</div>
                        <div id="trainingEstimatedTimeDisplay" style="font-size: 1.2em; color: #856404;">${estimatedTimeText}</div>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <div style="font-weight: 600; margin-bottom: 10px;">📊 Current Metrics</div>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                            <div>
                                <div style="font-size: 0.9em; color: #666;">Train Loss</div>
                                <div style="font-size: 1.1em; font-weight: 600; color: #333;">${trainLoss.toFixed(4)}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9em; color: #666;">Validation Loss</div>
                                <div style="font-size: 1.1em; font-weight: 600; color: #333;">${valLoss.toFixed(4)}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 2px solid #e0e0e0;">
                        <button id="stopTrainingBtn" 
                                style="width: 100%; padding: 12px; background: #dc3545; color: white; border: none; border-radius: 8px; font-size: 1em; font-weight: 600; cursor: pointer; transition: background 0.2s;"
                                onmouseover="this.style.background='#c82333'"
                                onmouseout="this.style.background='#dc3545'">
                            ⏹️ Stop Training
                        </button>
                    </div>
                    
                    ${progress.training_mode ? `
                        <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <div style="font-weight: 600; margin-bottom: 5px;">🎯 Training Mode</div>
                            <div style="color: #004085;">${progress.training_mode === 'classification' ? 'Classification' : 'Regression'}</div>
                            ${progress.num_classes ? `<div style="color: #004085; font-size: 0.9em;">Classes: ${progress.num_classes}</div>` : ''}
                        </div>
                    ` : ''}
                    
                    ${progress.dataset_stats ? `
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <div style="font-weight: 600; margin-bottom: 10px;">📦 Dataset Information</div>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 0.9em;">
                                <div>
                                    <div style="color: #666;">Train Samples</div>
                                    <div style="font-weight: 600;">${progress.dataset_stats.train_samples || 'N/A'}</div>
                                </div>
                                <div>
                                    <div style="color: #666;">Validation Samples</div>
                                    <div style="font-weight: 600;">${progress.dataset_stats.val_samples || 'N/A'}</div>
                                </div>
                            </div>
                        </div>
                    ` : ''}
                    
                    ${progress.training_config ? `
                        <div style="background: #e7f3ff; padding: 15px; border-radius: 8px;">
                            <div style="font-weight: 600; margin-bottom: 10px;">⚙️ Training Configuration</div>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 0.9em;">
                                <div>
                                    <div style="color: #666;">Total Epochs</div>
                                    <div style="font-weight: 600;">${progress.total_epochs || 'N/A'}</div>
                                </div>
                                <div>
                                    <div style="color: #666;">Learning Rate</div>
                                    <div style="font-weight: 600;">${progress.training_config.learning_rate || 'N/A'}</div>
                                </div>
                                <div>
                                    <div style="color: #666;">Batch Size</div>
                                    <div style="font-weight: 600;">${progress.training_config.batch_size || progress.dataset_stats?.batch_size || 'N/A'}</div>
                                </div>
                                <div>
                                    <div style="color: #666;">Train/Val Split</div>
                                    <div style="font-weight: 600;">${progress.training_config.train_split ? `${(progress.training_config.train_split * 100).toFixed(0)}/${((1 - progress.training_config.train_split) * 100).toFixed(0)}` : 'N/A'}</div>
                                </div>
                                <div>
                                    <div style="color: #666;">Augmentation</div>
                                    <div style="font-weight: 600;">${progress.training_config.use_augmentation ? 'Enabled (6x data)' : 'Disabled'}</div>
                                </div>
                                <div>
                                    <div style="color: #666;">Normalization</div>
                                    <div style="font-weight: 600;">${progress.training_config.use_normalization ? 'Enabled ([0,1])' : progress.training_mode === 'classification' ? 'Disabled (classification)' : 'Disabled (raw)'}</div>
                                </div>
                            </div>
                        </div>
                    ` : ''}
                `;
            } else if (status.status === 'completed') {
                const progress = status.progress || {};
                const modelPath = progress.model_path || 'N/A';
                const trainMetrics = progress.train_metrics || {};
                const valMetrics = progress.val_metrics || {};
                
                html = `
                    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h2 style="margin: 0 0 10px 0; font-size: 1.5em;">✅ Training Completed</h2>
                        <p style="margin: 0; opacity: 0.9;">Model saved successfully</p>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <div style="font-weight: 600; margin-bottom: 5px;">💾 Model Path</div>
                        <div style="font-family: monospace; font-size: 0.9em; color: #667eea;">${modelPath}</div>
                    </div>
                    
                    ${Object.keys(trainMetrics).length > 0 || Object.keys(valMetrics).length > 0 ? `
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <div style="font-weight: 600; margin-bottom: 10px;">📊 Final Metrics</div>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                                <div>
                                    <div style="font-weight: 600; color: #667eea; margin-bottom: 5px;">Training Set</div>
                                    ${Object.entries(trainMetrics).map(([key, value]) => `
                                        <div style="font-size: 0.9em; margin: 3px 0;">
                                            <span style="color: #666;">${key}:</span>
                                            <span style="font-weight: 600;">${typeof value === 'number' ? value.toFixed(4) : value}</span>
                                        </div>
                                    `).join('')}
                                </div>
                                <div>
                                    <div style="font-weight: 600; color: #28a745; margin-bottom: 5px;">Validation Set</div>
                                    ${Object.entries(valMetrics).map(([key, value]) => `
                                        <div style="font-size: 0.9em; margin: 3px 0;">
                                            <span style="color: #666;">${key}:</span>
                                            <span style="font-weight: 600;">${typeof value === 'number' ? value.toFixed(4) : value}</span>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    ` : ''}
                `;
            } else if (status.status === 'cancelled') {
                html = `
                    <div style="background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h2 style="margin: 0 0 10px 0; font-size: 1.5em;">⏹️ Training Cancelled</h2>
                        <p style="margin: 0; opacity: 0.9;">${status.error || 'Training was cancelled by user'}</p>
                    </div>
                    ${status.progress && status.progress.epoch ? `
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <div style="font-weight: 600; margin-bottom: 5px;">Progress at Cancellation</div>
                            <div style="color: #666;">Completed ${status.progress.epoch} of ${status.progress.total_epochs || 'N/A'} epochs</div>
                        </div>
                    ` : ''}
                `;
            } else if (status.status === 'error') {
                html = `
                    <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h2 style="margin: 0 0 10px 0; font-size: 1.5em;">❌ Training Error</h2>
                        <p style="margin: 0; opacity: 0.9;">${status.error || 'Unknown error occurred'}</p>
                    </div>
                `;
            } else {
                html = `
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 10px;">😴</div>
                        <div style="font-size: 1.2em; font-weight: 600; color: #666;">No Training Job Running</div>
                        <div style="color: #999; margin-top: 5px;">Start a new training job to see status here</div>
                    </div>
                `;
            }
            
            content.innerHTML = html;
            modal.style.display = 'flex';
            
            // Store base duration and timestamp for continuous updates
            if (status.status === 'training') {
                const progress = status.progress || {};
                if (progress.duration_seconds !== undefined && progress.duration_seconds >= 0) {
                    this.baseDurationSeconds = progress.duration_seconds;
                    this.baseDurationTimestamp = Date.now();
                }
                this.currentEpoch = progress.epoch || 0;
                this.totalEpochs = progress.total_epochs || 0;
                this.estimatedRemainingSeconds = progress.estimated_remaining_seconds || null;
                
                // Start continuous duration updates (every second)
                this.startDurationUpdates();
            } else {
                // Stop duration updates if not training
                this.stopDurationUpdates();
            }
            
            // Set up stop training button handler
            const stopBtn = document.getElementById('stopTrainingBtn');
            if (stopBtn && status.status === 'training') {
                stopBtn.onclick = async () => {
                    if (!confirm('Are you sure you want to stop the training? The current epoch will complete, then training will be cancelled.')) {
                        return;
                    }
                    
                    try {
                        const response = await fetch('/api/ai-training/stop-training', {
                            method: 'POST'
                        });
                        const result = await response.json();
                        
                        if (response.ok) {
                            alert(result.message || 'Training stop requested. The job will stop after the current epoch completes.');
                            // Refresh dialog to show updated status
                            setTimeout(() => this.showDialog(), 1000);
                        } else {
                            alert('Error: ' + (result.detail || 'Failed to stop training'));
                        }
                    } catch (error) {
                        console.error('Error stopping training:', error);
                        alert('Error stopping training: ' + error.message);
                    }
                };
            }
            
            // Auto-refresh dialog only if training is in progress (every 20 seconds)
            // Only refresh if dialog is actually open
            if (status.status === 'training') {
                // Clear any existing interval
                if (this.dialogRefreshIntervalId) {
                    clearInterval(this.dialogRefreshIntervalId);
                }
                // Set up refresh interval for open dialog
                this.dialogRefreshIntervalId = setInterval(() => {
                    // Check if dialog is still open before refreshing
                    const modalCheck = document.getElementById(this.modalId);
                    if (modalCheck && modalCheck.style.display === 'flex') {
                        this.showDialog();
                    } else {
                        // Dialog closed, stop refreshing
                        if (this.dialogRefreshIntervalId) {
                            clearInterval(this.dialogRefreshIntervalId);
                            this.dialogRefreshIntervalId = null;
                        }
                    }
                }, this.dialogRefreshInterval);
            } else {
                // Training not active, stop refreshing and duration updates
                this.stopDurationUpdates();
                if (this.dialogRefreshIntervalId) {
                    clearInterval(this.dialogRefreshIntervalId);
                    this.dialogRefreshIntervalId = null;
                }
            }
        } catch (error) {
            console.error('Error fetching training status:', error);
            const content = document.getElementById('trainingStatusContent');
            if (content) {
                content.innerHTML = `
                    <div style="background: #f8d7da; padding: 20px; border-radius: 8px; color: #721c24;">
                        <strong>Error:</strong> Failed to fetch training status: ${error.message}
                    </div>
                `;
            }
        }
    }
    
    startDurationUpdates() {
        // Clear any existing interval
        this.stopDurationUpdates();
        
        // Update duration every second
        this.durationUpdateIntervalId = setInterval(() => {
            this.updateDurationDisplay();
        }, 1000);
        
        // Initial update
        this.updateDurationDisplay();
    }
    
    stopDurationUpdates() {
        if (this.durationUpdateIntervalId) {
            clearInterval(this.durationUpdateIntervalId);
            this.durationUpdateIntervalId = null;
        }
    }
    
    updateDurationDisplay() {
        const durationElement = document.getElementById('trainingDurationDisplay');
        const estimatedElement = document.getElementById('trainingEstimatedTimeDisplay');
        
        if (!durationElement) return; // Dialog might be closed
        
        // Calculate current duration: base + elapsed since base timestamp
        let currentDurationSeconds = 0;
        if (this.baseDurationSeconds !== null && this.baseDurationTimestamp !== null) {
            const elapsedSinceBase = Math.floor((Date.now() - this.baseDurationTimestamp) / 1000);
            currentDurationSeconds = this.baseDurationSeconds + elapsedSinceBase;
        } else {
            // Fallback: use startTime if available
            if (this.startTime) {
                currentDurationSeconds = Math.floor((Date.now() - this.startTime.getTime()) / 1000);
            } else {
                durationElement.textContent = 'Calculating...';
                return;
            }
        }
        
        // Update duration display
        durationElement.textContent = this.formatDuration(currentDurationSeconds);
        
        // Update estimated remaining time if we have epoch info
        if (estimatedElement && this.currentEpoch > 0 && this.totalEpochs > 0) {
            let estimatedText = 'Calculating...';
            
            if (this.estimatedRemainingSeconds !== null && this.estimatedRemainingSeconds > 0) {
                // Use server-provided estimate, but adjust for time elapsed since last update
                // We'll recalculate based on current progress
                const elapsedSinceBase = Math.floor((Date.now() - this.baseDurationTimestamp) / 1000);
                const adjustedEstimate = Math.max(0, this.estimatedRemainingSeconds - elapsedSinceBase);
                estimatedText = this.formatDuration(adjustedEstimate);
            } else if (this.baseDurationSeconds !== null && this.baseDurationTimestamp !== null) {
                // Calculate estimate based on average epoch time
                const elapsedSinceBase = Math.floor((Date.now() - this.baseDurationTimestamp) / 1000);
                const totalElapsed = this.baseDurationSeconds + elapsedSinceBase;
                
                if (this.currentEpoch > 0) {
                    const avgTimePerEpoch = totalElapsed / this.currentEpoch;
                    const remainingEpochs = this.totalEpochs - this.currentEpoch;
                    const estimatedSeconds = Math.floor(avgTimePerEpoch * remainingEpochs);
                    estimatedText = this.formatDuration(estimatedSeconds);
                }
            }
            
            estimatedElement.textContent = estimatedText;
        }
    }
    
    hideDialog() {
        const modal = document.getElementById(this.modalId);
        if (modal) {
            modal.style.display = 'none';
        }
        
        // Stop all intervals when dialog is closed
        this.stopDurationUpdates();
        if (this.dialogRefreshIntervalId) {
            clearInterval(this.dialogRefreshIntervalId);
            this.dialogRefreshIntervalId = null;
        }
    }
    
    destroy() {
        if (this.dialogRefreshIntervalId) {
            clearInterval(this.dialogRefreshIntervalId);
        }
    }
}

// Global instance (can be accessed from HTML)
let trainingStatusInstance = null;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        trainingStatusInstance = new TrainingStatus();
    });
} else {
    trainingStatusInstance = new TrainingStatus();
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TrainingStatus;
}
