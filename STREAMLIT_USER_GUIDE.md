# CBM Expert Feedback System - User Guide

**Application URL:** https://xr5/nsai.com

---

## Overview

The CBM (Concept-Based Model) Expert Feedback System is an AI-powered platform for detecting infrastructure defects in video footage. The system analyzes videos for:
- **Normal** conditions
- **Cracks** in infrastructure
- **Corrosion** damage
- **Leakage** issues

The platform uses active learning, meaning the AI model improves over time based on expert feedback.

---

## Getting Started

### Accessing the Application

1. Open your web browser
2. Navigate to **https://xr5/nsai.com**
3. The application will load with the Dashboard view

### Main Sections

The application has **5 main sections** accessible from the left sidebar:

- **🏠 Dashboard** - System overview and statistics
- **🎥 Video Processing** - Upload and analyze videos
- **📝 Expert Feedback** - Review AI predictions and provide corrections
- **📊 Analytics** - View performance metrics and trends
- **🤖 Model Management** - Retrain the AI model

### System Status

The sidebar displays the **System Status** at the top:
- **✅ API Connected** (Green) - System is operational
- **⚠️ API Error** (Yellow) - Connection issues
- **❌ API Disconnected** (Red) - System unavailable

---

## Dashboard

The Dashboard provides a quick overview:

- **Total Samples** - Number of feedback samples collected
- **Corrections** - How many AI predictions were corrected by experts
- **Retraining Status** - Whether the system is ready for model updates
- **Defect Samples** - Number of non-normal classifications
- **Retraining Progress Bar** - Shows progress toward next model update (50 samples needed)
- **Sample Distribution Chart** - Visual breakdown by classification type

---

## Video Processing

### Upload & Process Tab

#### Step 1: Upload Video
1. Click **"Choose a video file"**
2. Select a video (Supported: MP4, AVI, MOV, MKV, WMV, FLV)
3. Review video preview and file details

#### Step 2: Configure Processing

**Frame Interval** (1-300):
- Lower = more frames (detailed but slower)
- Higher = fewer frames (faster but less detail)
- **Recommended:** 30 frames

**Auto-run predictions:**
- Check to analyze frames immediately after extraction

**Quality Mode:**
- Standard / High Quality / Fast

#### Step 3: Process
1. Click **"🚀 Upload & Process Video"**
2. Wait for processing (see progress bar)
3. Review results

**Processing Time:**
- Small video (< 50 MB): ~30 seconds
- Medium (50-200 MB): 1-3 minutes
- Large (> 200 MB): 3-10 minutes

### Manage Videos Tab

View all processed videos with:
- Video name
- Frame count
- Prediction status

**Actions:**
- **📊 View Predictions** - Review for feedback
- **🔄 Re-analyze** - Run again with different settings
- **🗑️ Delete** - Remove video data

---

## Expert Feedback

### Review Predictions Tab

#### Step 1: Select Video
Choose a processed video from the dropdown

#### Step 2: Filter Predictions

- **Include Normal Predictions** - Show all or only defects
- **Filter by Class** - All, Normal, Crack, Corrosion, or Leakage
- **Min Confidence** - Threshold slider (0.0 - 1.0)

#### Step 3: Review Options

**Sort by:**
- Frame Order
- Confidence (Low to High) - Start with uncertain predictions
- Confidence (High to Low)
- Class

**Items per page:** 5, 10, 20, or 50

**Review Mode:**
- **Standard** - Full feedback forms
- **Quick Review** - Fast confirmation/correction
- **Detailed Analysis** - Comprehensive feedback

#### Step 4: Submit Feedback

For each prediction:

1. **Correct Classification** - Select true class (Normal, Crack, Corrosion, Leakage)
2. **Your Confidence** - How confident are you? (0.0 - 1.0)
   - 0.9+: Very confident
   - 0.7-0.8: Moderately confident
   - Below 0.7: Uncertain
3. **Priority** - Low, Medium, High, or Critical
4. **Notes** - Describe defect, location, severity, recommended actions
5. Click **"Submit Feedback"**

**Feedback Types:**
- **🔄 Correction** - Expert disagrees with AI
- **✅ Confirmation** - Expert agrees with AI

Both types help improve the model!

### Manual Upload Tab

Upload individual images (not from video):
1. Choose image file
2. Enter Frame/Image ID
3. Select classifications
4. Add confidence and notes
5. Submit

---

## Model Management

### Retraining Tab

**Status Metrics:**
- Samples Collected
- Threshold (typically 50)
- Status (Ready or Collecting)

**When Ready (50+ samples):**

1. Configure options:
   - ✅ Backup current model (recommended)
   - ✅ Use all available data (recommended)
   - Validation split (default: 0.2)

2. Click **"🚀 Start Retraining"**
3. Wait 5-30 minutes
4. Review results

After retraining, the model uses your feedback to make better predictions!

---

## Tips & Best Practices

### For Best Results

1. **Start with Low Confidence Predictions**
   - Sort by "Confidence (Low to High)"
   - Focus on where AI needs most help

2. **Provide Detailed Notes**
   - Describe location, size, severity
   - Note unusual features
   - Recommend actions

3. **Be Consistent**
   - Use same criteria each time
   - Lower confidence when unsure

4. **Confirm Correct Predictions**
   - Both corrections AND confirmations improve the model

### Frame Interval Guidelines

- Slow-moving video: 30-50 frames
- Medium-speed video: 20-30 frames
- Fast-moving video: 10-20 frames
- Detailed inspection: 5-10 frames

### Good Feedback Examples

✅ "Clear horizontal crack in lower right quadrant, approximately 15cm long, moderate severity. Recommend inspection within 30 days."

✅ "Surface corrosion on metal pipe joint, rust-colored, covers approximately 30% of visible surface. Monitor for progression."

❌ "crack" (too brief)
❌ "looks bad" (not specific)

### When to Retrain

**Retrain when:**
- ✅ You have 50+ feedback samples
- ✅ Multiple videos reviewed
- ✅ Notice consistent errors

**Don't retrain when:**
- ❌ Only one video reviewed
- ❌ Fewer than 50 samples
- ❌ Just retrained recently

---

## Common Workflows

### First-Time Video Analysis

1. Go to **Video Processing** → Upload & Process
2. Upload video, set interval to 30, enable auto-predict
3. Click "Upload & Process Video"
4. Go to **Expert Feedback** → Review Predictions
5. Provide feedback on predictions
6. Repeat until 50+ samples
7. Go to **Model Management** → Retraining
8. Click "Start Retraining"

### Quick Review

1. **Expert Feedback** → Review Predictions
2. Select video
3. Uncheck "Include Normal"
4. Set Min Confidence to 0.5
5. Sort by "Confidence (Low to High)"
6. Use "Quick Review" mode
7. Confirm/correct predictions

---

## Troubleshooting

### API Disconnected
- Refresh page
- Check internet connection
- Contact system administrator

### Video Upload Failed
- Check file size (< 500 MB recommended)
- Use supported format (MP4, AVI, MOV)
- Try during off-peak hours

### No Predictions Showing
- Check if video has predictions (Manage Videos tab)
- Adjust filters (Include Normal, lower confidence threshold)
- Verify video processed correctly

---

## Glossary

**Active Learning** - AI improves based on expert corrections

**Confidence Score** - How certain the AI is (0.0 = uncertain, 1.0 = very certain)

**Correction** - Expert feedback differs from AI prediction

**Frame Interval** - Extract every Nth frame

**Retraining** - Updating AI model with new feedback

**Threshold** - Minimum samples needed before retraining (typically 50)

---

**Version:** 2.0
**Last Updated:** 2025
