/**
 * Distribution Preview Module
 * 
 * Handles displaying feature distribution previews with histogram,
 * statistics, and auto-classification options.
 */

const DistributionPreview = {
    currentFeature: null,
    distributionData: null,
    selectedPreviewClasses: null, // Store selected preview (num_classes)
    
    /**
     * Show distribution preview modal
     */
    async showPreview(featureName, event) {
        event.stopPropagation(); // Prevent feature selection
        
        this.currentFeature = featureName;
        const modal = document.getElementById('distributionModal');
        const modalBody = document.getElementById('modalBody');
        const modalStats = document.getElementById('modalStats');
        const modalTitle = document.getElementById('modalTitle');
        
        // Prevent body scrolling
        document.body.classList.add('modal-open');
        
        // Show loading state
        modalTitle.textContent = `Distribution Preview: ${featureName}`;
        modalStats.innerHTML = ''; // Clear stats
        modalBody.innerHTML = `
            <div style="text-align: center; padding: 40px;">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #666;">Loading distribution data...</p>
            </div>
        `;
        modal.classList.add('show');
        
        try {
            // Fetch pre-calculated data from server
            const response = await fetch(`/api/ai-training/feature-distribution/${encodeURIComponent(featureName)}`);
            
            if (!response.ok) {
                throw new Error(`Failed to load distribution: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.distributionData = data;
            
            // Render modal content
            this.renderModal(data);
            
        } catch (error) {
            console.error('Error loading distribution:', error);
            modalStats.innerHTML = '';
            modalBody.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #dc3545;">
                    <div style="font-size: 3em; margin-bottom: 15px;">⚠️</div>
                    <h3>Error Loading Distribution</h3>
                    <p style="margin-top: 10px;">${error.message}</p>
                </div>
            `;
        }
    },
    
    /**
     * Render modal content (histogram, stats, classes)
     */
    renderModal(data) {
        const modalStats = document.getElementById('modalStats');
        const modalBody = document.getElementById('modalBody');
        
        // 1. Render statistics (fixed at top)
        modalStats.innerHTML = this.renderStats(data.statistics);
        
        // 2. Render histogram (scrollable)
        const histogramHtml = this.renderHistogram(data.histogram);
        
        // 3. Render auto-classification section (scrollable)
        const classesHtml = this.renderAutoClasses(data.auto_classifications);
        
        modalBody.innerHTML = `
            ${histogramHtml}
            ${classesHtml}
        `;
    },
    
    /**
     * Render statistics section
     */
    renderStats(stats) {
        return `
            <div class="stats-grid-modal">
                <div class="stat-item">
                    <div class="stat-item-label">Min</div>
                    <div class="stat-item-value">${stats.min.toFixed(1)}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-item-label">Max</div>
                    <div class="stat-item-value">${stats.max.toFixed(1)}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-item-label">Mean</div>
                    <div class="stat-item-value">${stats.mean.toFixed(1)}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-item-label">Median</div>
                    <div class="stat-item-value">${stats.median.toFixed(1)}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-item-label">Std Dev</div>
                    <div class="stat-item-value">${stats.std.toFixed(1)}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-item-label">Range</div>
                    <div class="stat-item-value">${stats.range.toFixed(1)}</div>
                </div>
            </div>
        `;
    },
    
    /**
     * Render histogram
     */
    renderHistogram(histogram) {
        const maxCount = Math.max(...histogram.data.map(bin => bin.count));
        
        const barsHtml = histogram.data.map(bin => {
            // Calculate width as percentage of max count (not 100% for each)
            const widthPercent = maxCount > 0 ? (bin.count / maxCount) * 100 : 0;
            return `
                <div class="histogram-bar">
                    <div class="histogram-label">${bin.min.toFixed(1)}</div>
                    <div class="histogram-bar-visual" style="width: ${widthPercent}%; max-width: 100%;"></div>
                    <div class="histogram-count">${bin.count} (${bin.percentage}%)</div>
                </div>
            `;
        }).join('');
        
        return `
            <div class="histogram-container">
                <h3 style="margin-bottom: 15px; color: #333;">Distribution Histogram</h3>
                <div style="margin-bottom: 10px; color: #666; font-size: 0.9em;">
                    Total Samples: <strong>${histogram.data.reduce((sum, bin) => sum + bin.count, 0)}</strong>
                </div>
                ${barsHtml}
            </div>
        `;
    },
    
    /**
     * Render auto-classification section
     */
    renderAutoClasses(autoClassifications) {
        const buttonsHtml = Object.keys(autoClassifications).map(key => {
            const numClasses = parseInt(key.replace('_classes', ''));
            const isSelected = this.selectedPreviewClasses === numClasses;
            return `
                <button class="class-btn ${isSelected ? 'class-btn-selected' : ''}" 
                        onclick="DistributionPreview.showClassPreview(${numClasses})">
                    ${numClasses} Classes
                </button>
            `;
        }).join('');
        
        // Show preview only if a class count is selected
        let previewHtml = '';
        if (this.selectedPreviewClasses) {
            const previewKey = `${this.selectedPreviewClasses}_classes`;
            const previewData = autoClassifications[previewKey];
            if (previewData) {
                previewHtml = this.renderClassPreview(previewData, this.selectedPreviewClasses);
            }
        }
        
        return `
            <div class="auto-classes-section">
                <h3 style="margin-bottom: 15px; color: #333;">Auto Classification</h3>
                <p style="color: #666; margin-bottom: 15px; font-size: 0.95em;">
                    Click on a number to preview class distribution. Then decide whether to save the classes to the database.
                </p>
                <div class="auto-classes-buttons">
                    ${buttonsHtml}
                </div>
                ${previewHtml}
            </div>
        `;
    },
    
    /**
     * Show class preview (without saving)
     */
    showClassPreview(numClasses) {
        this.selectedPreviewClasses = numClasses;
        // Re-render the modal with updated preview
        if (this.distributionData) {
            this.renderModal(this.distributionData);
        }
    },
    
    /**
     * Render class preview
     */
    renderClassPreview(classData, numClasses) {
        const totalSamples = classData.classes.reduce((sum, cls) => sum + cls.count, 0);
        
        // Distribution plot (bars) - use actual percentage for width
        const distributionPlot = classData.classes.map((cls, index) => {
            // Use actual percentage directly (100% = 100% width)
            const widthPercent = cls.percentage;
            // Use different colors for each class
            const colors = [
                'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                'linear-gradient(90deg, #f093fb 0%, #f5576c 100%)',
                'linear-gradient(90deg, #4facfe 0%, #00f2fe 100%)',
                'linear-gradient(90deg, #43e97b 0%, #38f9d7 100%)',
                'linear-gradient(90deg, #fa709a 0%, #fee140 100%)',
                'linear-gradient(90deg, #30cfd0 0%, #330867 100%)',
                'linear-gradient(90deg, #a8edea 0%, #fed6e3 100%)',
                'linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%)',
                'linear-gradient(90deg, #ffecd2 0%, #fcb69f 100%)',
                'linear-gradient(90deg, #ff8a80 0%, #ea4c89 100%)'
            ];
            const color = colors[index % colors.length];
            
            return `
                <div class="class-distribution-bar">
                    <div class="class-distribution-label">
                        <div style="font-weight: 600; color: #333;">${cls.label}</div>
                        <div style="font-size: 0.85em; color: #666; margin-top: 2px;">
                            ${cls.min.toFixed(1)} - ${cls.max.toFixed(1)}
                        </div>
                    </div>
                    <div class="class-distribution-bar-container">
                        <div class="class-distribution-bar-visual" style="width: ${widthPercent}%; background: ${color};"></div>
                    </div>
                    <div class="class-distribution-count">
                        <div style="font-weight: 600; color: #333;">${cls.count}</div>
                        <div style="font-size: 0.85em; color: #666;">${cls.percentage}%</div>
                    </div>
                </div>
            `;
        }).join('');
        
        return `
            <div class="class-info">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h4 style="margin: 0; color: #667eea;">Preview: ${numClasses} Classes</h4>
                    <button class="class-btn class-btn-save" onclick="DistributionPreview.handleGenerateClasses(${numClasses})">
                        💾 Save to Database
                    </button>
                </div>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 15px;">
                    This will add class labels as additional features. Original values remain unchanged.
                </p>
                
                <!-- Distribution Plot -->
                <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef;">
                    <h5 style="margin: 0 0 15px 0; color: #333; font-size: 1em;">Class Distribution</h5>
                    <div style="margin-bottom: 10px; color: #666; font-size: 0.9em;">
                        Total Samples: <strong>${totalSamples}</strong>
                    </div>
                    ${distributionPlot}
                </div>
            </div>
        `;
    },
    
    /**
     * Handle "Add N Classes" button click
     */
    async handleGenerateClasses(numClasses) {
        if (!this.currentFeature) {
            alert('No feature selected');
            return;
        }
        
        if (!confirm(`Generate ${numClasses} classes for "${this.currentFeature}"?\n\nThis will add class labels to all database entries.`)) {
            return;
        }
        
        const modalBody = document.getElementById('modalBody');
        const originalContent = modalBody.innerHTML;
        
        // Show loading state
        modalBody.innerHTML = `
            <div style="text-align: center; padding: 40px;">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #666;">Generating classes...</p>
            </div>
        `;
        
        try {
            // Call backend to generate and save classes
            const response = await fetch('/api/ai-training/generate-classes', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    feature_name: this.currentFeature,
                    num_classes: numClasses,
                    method: 'quantile'
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to generate classes');
            }
            
            const result = await response.json();
            
            // Show success message
            modalBody.innerHTML = `
                <div style="text-align: center; padding: 40px;">
                    <div style="font-size: 3em; margin-bottom: 15px;">✅</div>
                    <h3 style="color: #28a745; margin-bottom: 15px;">Classes Generated Successfully!</h3>
                    <p style="color: #666; margin-bottom: 20px;">
                        Updated <strong>${result.updated_count}</strong> entries in database.
                    </p>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: left;">
                        <h4 style="margin-bottom: 10px; color: #333;">Class Distribution:</h4>
                        ${result.class_info.map(cls => `
                            <div style="padding: 8px; border-bottom: 1px solid #e9ecef;">
                                <strong>Class ${cls.class_id}</strong> (${cls.range}): 
                                ${cls.count} samples (${cls.percentage}%)
                            </div>
                        `).join('')}
                    </div>
                    <button class="btn" onclick="DistributionPreview.hideModal()" style="margin-top: 20px;">
                        Close
                    </button>
                </div>
            `;
            
        } catch (error) {
            console.error('Error generating classes:', error);
            modalBody.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #dc3545;">
                    <div style="font-size: 3em; margin-bottom: 15px;">⚠️</div>
                    <h3>Error Generating Classes</h3>
                    <p style="margin-top: 10px;">${error.message}</p>
                    <button class="btn" onclick="DistributionPreview.showPreview('${this.currentFeature}', {stopPropagation: () => {}})" 
                            style="margin-top: 20px;">
                        Try Again
                    </button>
                </div>
            `;
        }
    },
    
    /**
     * Hide modal
     */
    hideModal() {
        const modal = document.getElementById('distributionModal');
        modal.classList.remove('show');
        // Re-enable body scrolling
        document.body.classList.remove('modal-open');
        this.currentFeature = null;
        this.distributionData = null;
        this.selectedPreviewClasses = null; // Reset preview selection
    }
};

// Close modal on overlay click and prevent background scrolling
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('distributionModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                DistributionPreview.hideModal();
            }
        });
        
        // Prevent scrolling on modal overlay (only allow scrolling in .modal-body-scrollable)
        modal.addEventListener('wheel', (e) => {
            const scrollable = modal.querySelector('.modal-body-scrollable');
            if (scrollable && e.target !== scrollable && !scrollable.contains(e.target)) {
                e.preventDefault();
                e.stopPropagation();
            }
        }, { passive: false });
        
        // Prevent touch scrolling on modal overlay
        modal.addEventListener('touchmove', (e) => {
            const scrollable = modal.querySelector('.modal-body-scrollable');
            if (scrollable && e.target !== scrollable && !scrollable.contains(e.target)) {
                e.preventDefault();
                e.stopPropagation();
            }
        }, { passive: false });
    }
});

