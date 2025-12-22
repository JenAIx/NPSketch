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
    customClassNames: {}, // Store custom class names {num_classes: {class_id: name}}
    
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
        
        // Check if this is a Custom_Class feature
        const isCustomClass = featureName.startsWith('Custom_Class_');
        
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
            if (isCustomClass) {
                // Load custom class distribution
                await this.showCustomClassPreview(featureName);
            } else {
                // Load regular feature distribution
                const response = await fetch(`/api/ai-training/feature-distribution/${encodeURIComponent(featureName)}`);
                
                if (!response.ok) {
                    throw new Error(`Failed to load distribution: ${response.statusText}`);
                }
                
                const data = await response.json();
                this.distributionData = data;
                
                // Render modal content
                this.renderModal(data);
            }
            
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
     * Show preview for Custom_Class feature
     */
    async showCustomClassPreview(featureName) {
        // Extract num_classes from feature name (e.g., "Custom_Class_5" -> "5")
        const numClasses = featureName.replace('Custom_Class_', '');
        
        const response = await fetch(`/api/ai-training/custom-class-distribution/${encodeURIComponent(featureName)}`);
        
        if (!response.ok) {
            throw new Error(`Failed to load custom class distribution: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Render custom class preview
        this.renderCustomClassPreview(data, numClasses);
    },
    
    /**
     * Render custom class preview (for existing classifications)
     */
    renderCustomClassPreview(data, numClasses) {
        const modalStats = document.getElementById('modalStats');
        const modalBody = document.getElementById('modalBody');
        
        // Stats: Show class counts
        const totalSamples = data.total_samples;
        const classStats = data.classes.map(cls => 
            `<div class="stat-item">
                <div class="stat-item-label">${cls.name_custom || cls.name_generic}</div>
                <div class="stat-item-value">${cls.count}</div>
            </div>`
        ).join('');
        
        modalStats.innerHTML = `<div class="stats-grid-modal">${classStats}</div>`;
        
        // Body: Show class distribution bars
        const maxCount = Math.max(...data.classes.map(cls => cls.count));
        const colors = [
            'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
            'linear-gradient(90deg, #f093fb 0%, #f5576c 100%)',
            'linear-gradient(90deg, #4facfe 0%, #00f2fe 100%)',
            'linear-gradient(90deg, #43e97b 0%, #38f9d7 100%)',
            'linear-gradient(90deg, #fa709a 0%, #fee140 100%)'
        ];
        
        const barsHtml = data.classes.map((cls, index) => {
            const widthPercent = cls.percentage;
            const color = colors[index % colors.length];
            const name = cls.name_custom || cls.name_generic;
            
            return `
                <div class="class-distribution-bar">
                    <div class="class-distribution-label">
                        <div style="font-weight: 600; color: #333;">${name}</div>
                        <div style="font-size: 0.85em; color: #666; margin-top: 2px;">
                            ${cls.range}
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
        
        modalBody.innerHTML = `
            <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef;">
                <h3 style="margin: 0 0 15px 0; color: #333;">Class Distribution</h3>
                <div style="margin-bottom: 10px; color: #666; font-size: 0.9em;">
                    Total Samples: <strong>${totalSamples}</strong> in <strong>${numClasses}</strong> classes
                </div>
                ${barsHtml}
            </div>
            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea;">
                <strong>ℹ️ Note:</strong> This classification is already saved in the database. 
                To modify, create a new classification with different number of classes.
            </div>
        `;
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
                    Click on a number to preview class distribution. <strong>Classes are balanced</strong> to have approximately equal number of samples per class. Classes represent score ranges from lowest to highest.
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
        
        // Get custom names if they exist
        const customNames = this.customClassNames[numClasses] || {};
        
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
            
            const customName = customNames[cls.id] || cls.label;
            
            const rangeDisplay = cls.min === cls.max 
                ? `= <span class="editable-boundary" contenteditable="true" 
                          data-class-id="${cls.id}" 
                          data-boundary="single"
                          data-num-classes="${numClasses}"
                          onblur="DistributionPreview.updateBoundary(${numClasses}, ${cls.id}, 'single', this.textContent)"
                          onkeydown="if(event.key==='Enter'){event.preventDefault();this.blur();}">${cls.min}</span>`
                : `<span class="editable-boundary" contenteditable="true"
                         data-class-id="${cls.id}"
                         data-boundary="min"
                         data-num-classes="${numClasses}"
                         onblur="DistributionPreview.updateBoundary(${numClasses}, ${cls.id}, 'min', this.textContent)"
                         onkeydown="if(event.key==='Enter'){event.preventDefault();this.blur();}">${cls.min}</span>-<span class="editable-boundary" contenteditable="true"
                         data-class-id="${cls.id}"
                         data-boundary="max"
                         data-num-classes="${numClasses}"
                         onblur="DistributionPreview.updateBoundary(${numClasses}, ${cls.id}, 'max', this.textContent)"
                         onkeydown="if(event.key==='Enter'){event.preventDefault();this.blur();}">${cls.max}</span>`;
            
            return `
                <div class="class-distribution-bar">
                    <div class="class-distribution-label">
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <span class="editable-label" 
                                  contenteditable="true" 
                                  style="font-weight: 600; color: #333; outline: none; cursor: text;"
                                  data-class-id="${cls.id}"
                                  data-num-classes="${numClasses}"
                                  onblur="DistributionPreview.saveClassName(${numClasses}, ${cls.id}, this.textContent)"
                                  onkeydown="if(event.key==='Enter'){event.preventDefault();this.blur();}">${customName}</span>
                            <span class="edit-icon" title="Click to edit">✏️</span>
                        </div>
                        <div style="font-size: 0.85em; color: #666; margin-top: 2px;">
                            ${rangeDisplay}
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
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <h4 style="margin: 0; color: #667eea;">Preview: ${numClasses} Classes</h4>
                        <button class="magic-wand-btn" 
                                onclick="DistributionPreview.autoRenameClasses(${numClasses})"
                                title="Auto-generate meaningful class names">
                            🪄
                        </button>
                    </div>
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
            // Get current class data (with any custom modifications)
            const classKey = `${numClasses}_classes`;
            const classesData = this.distributionData.auto_classifications[classKey];
            const classes = classesData.classes;
            
            // Get custom names if any
            const customNames = this.customClassNames[numClasses] || {};
            
            // Prepare custom classes data
            const customClasses = classes.map(cls => ({
                id: cls.id,
                min: cls.min,
                max: cls.max,
                count: cls.count,
                percentage: cls.percentage,
                custom_name: customNames[cls.id] || null,  // null if not customized
                generic_name: `Class_${cls.id} [${cls.min}-${cls.max}]`
            }));
            
            // Call backend to generate and save classes
            const response = await fetch('/api/ai-training/generate-classes', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    feature_name: this.currentFeature,
                    num_classes: numClasses,
                    method: 'custom',  // Use 'custom' when we have modifications
                    custom_classes: customClasses
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
     * Save custom class name
     */
    saveClassName(numClasses, classId, newName) {
        if (!this.customClassNames[numClasses]) {
            this.customClassNames[numClasses] = {};
        }
        this.customClassNames[numClasses][classId] = newName.trim();
    },
    
    /**
     * Update boundary value
     */
    async updateBoundary(numClasses, classId, boundaryType, newValue) {
        const value = parseInt(newValue.trim());
        
        if (isNaN(value)) {
            alert('Please enter a valid number');
            // Re-render to restore original value
            if (this.distributionData) {
                this.renderModal(this.distributionData);
            }
            return;
        }
        
        // Get current class data
        const classKey = `${numClasses}_classes`;
        const classesData = this.distributionData.auto_classifications[classKey];
        const classes = classesData.classes;
        
        // Validate range
        const stats = this.distributionData.statistics;
        if (value < stats.min || value > stats.max) {
            alert(`Value must be between ${stats.min} and ${stats.max}`);
            this.renderModal(this.distributionData);
            return;
        }
        
        // Update boundaries
        if (boundaryType === 'min') {
            classes[classId].min = value;
            if (classId > 0) {
                classes[classId - 1].max = value - 1;
            }
        } else if (boundaryType === 'max') {
            classes[classId].max = value;
            if (classId < classes.length - 1) {
                classes[classId + 1].min = value + 1;
            }
        } else if (boundaryType === 'single') {
            classes[classId].min = value;
            classes[classId].max = value;
            if (classId > 0) {
                classes[classId - 1].max = value - 1;
            }
            if (classId < classes.length - 1) {
                classes[classId + 1].min = value + 1;
            }
        }
        
        // Recalculate counts and percentages
        await this.recalculateClassCounts(numClasses);
        
        // Re-render to show updated values
        this.renderModal(this.distributionData);
    },
    
    /**
     * Recalculate class counts based on new boundaries
     */
    async recalculateClassCounts(numClasses) {
        const classKey = `${numClasses}_classes`;
        const classesData = this.distributionData.auto_classifications[classKey];
        const classes = classesData.classes;
        
        // We need the raw scores to recalculate
        // Fetch them from the histogram data (approximate) or make API call
        try {
            const response = await fetch(`/api/ai-training/recalculate-classes`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    feature_name: this.currentFeature,
                    num_classes: numClasses,
                    boundaries: classes.map(cls => [cls.min, cls.max])
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                // Update counts and percentages
                result.classes.forEach((updated, idx) => {
                    classes[idx].count = updated.count;
                    classes[idx].percentage = updated.percentage;
                });
            } else {
                console.warn('Could not recalculate counts - using estimates');
                // Fallback: estimate from histogram
                this.estimateCountsFromHistogram(classes);
            }
        } catch (error) {
            console.warn('Error recalculating counts:', error);
            this.estimateCountsFromHistogram(classes);
        }
    },
    
    /**
     * Estimate counts from histogram (fallback)
     */
    estimateCountsFromHistogram(classes) {
        const histogram = this.distributionData.histogram;
        const totalSamples = this.distributionData.total_samples;
        
        classes.forEach(cls => {
            let estimatedCount = 0;
            
            histogram.data.forEach(bin => {
                // Check if bin overlaps with class range
                const binMin = bin.min;
                const binMax = bin.max;
                
                if (binMax >= cls.min && binMin <= cls.max) {
                    // Overlap - add proportional count
                    const overlapMin = Math.max(binMin, cls.min);
                    const overlapMax = Math.min(binMax, cls.max);
                    const binWidth = binMax - binMin;
                    const overlapWidth = overlapMax - overlapMin;
                    
                    if (binWidth > 0) {
                        const proportion = overlapWidth / binWidth;
                        estimatedCount += bin.count * proportion;
                    } else {
                        estimatedCount += bin.count;
                    }
                }
            });
            
            cls.count = Math.round(estimatedCount);
            cls.percentage = ((cls.count / totalSamples) * 100).toFixed(2);
        });
    },
    
    /**
     * Auto-rename classes with meaningful names
     * Toggles between ascending (Poor→Excellent) and descending (Excellent→Poor)
     */
    autoRenameClasses(numClasses) {
        const presets = {
            2: ['Poor', 'Good'],
            3: ['Poor', 'Fair', 'Good'],
            4: ['Poor', 'Fair', 'Good', 'Excellent'],
            5: ['Poor', 'Fair', 'Moderate', 'Good', 'Excellent']
        };
        
        let names = presets[numClasses] || [];
        if (names.length === 0) {
            alert('No preset names available for this number of classes');
            return;
        }
        
        // Check if already renamed - if so, reverse the order
        const existingNames = this.customClassNames[numClasses] || {};
        const hasCustomNames = Object.keys(existingNames).length > 0;
        
        if (hasCustomNames) {
            // Check if current names match the preset (ascending)
            const isAscending = existingNames[0] === names[0];
            if (isAscending) {
                // Reverse to descending (Excellent → Poor)
                names = [...names].reverse();
            }
            // If descending or custom, apply ascending again
        }
        
        // Store names
        this.customClassNames[numClasses] = {};
        names.forEach((name, idx) => {
            this.customClassNames[numClasses][idx] = name;
        });
        
        // Re-render to show new names
        if (this.distributionData) {
            this.renderModal(this.distributionData);
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
        // Keep customClassNames for next time
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

