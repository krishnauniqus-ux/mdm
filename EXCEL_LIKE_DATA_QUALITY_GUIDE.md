# Excel-Like Data Quality - User Guide

## 🎯 Overview

The Data Quality tab has been completely redesigned to work like Excel with a clean, tabular interface. Select columns from the grid and apply transformations instantly!

---

## 🖥️ Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Metrics Bar: Total Rows | Total Columns | Rejected | Undo  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │   DATA GRID (Left)   │  │  TRANSFORMATIONS (Right) │    │
│  │                      │  │                          │    │
│  │  ☑ Column Checkboxes │  │  🤖 AI Suggestions      │    │
│  │  ☑ Select/Clear/     │  │  🔤 Regex Patterns      │    │
│  │     Invert Buttons   │  │  📏 Length Validation   │    │
│  │                      │  │  🔠 Case Transform      │    │
│  │  📊 Data Preview     │  │                          │    │
│  │     (100 rows)       │  │  Quick Apply Buttons    │    │
│  │                      │  │                          │    │
│  │  Selected columns    │  │                          │    │
│  │  highlighted in blue │  │                          │    │
│  │                      │  │                          │    │
│  └──────────────────────┘  └──────────────────────────┘    │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│  📜 Applied Rules History (Expandable)                       │
│  🗑️ Rejected Records (Expandable)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Workflow

### Step 1: Select Columns
1. Check the boxes for columns you want to transform
2. Or use quick buttons:
   - **Select All**: Check all columns
   - **Clear All**: Uncheck all columns
   - **Invert**: Flip selection

### Step 2: Choose Transformation
Switch between tabs on the right:
- **🤖 AI**: Get smart suggestions
- **🔤 Regex**: Pattern matching
- **📏 Length**: Validate length
- **🔠 Case**: Change case

### Step 3: Apply
- Click the **Apply** button
- Changes happen instantly
- Invalid rows go to Rejected Records
- Use **Undo** if needed

---

## 🤖 AI Suggestions Tab

### How to Use:
1. Select columns from grid
2. Go to **AI** tab
3. Describe what you want:
   ```
   "Remove special characters"
   "Replace underscores with spaces"
   "Extract only numbers"
   "Validate emails"
   ```
4. Click **Get Suggestion**
5. Review the pattern and explanation
6. Click **Apply** or **Clear**

### Example Prompts:
- "Remove special chars"
- "Replace _ with space"
- "Extract only numbers"
- "Validate emails"
- "Keep only letters"
- "Remove spaces"

### What You Get:
- ✅ Recommended mode
- ✅ Regex pattern
- ✅ Replacement text (if needed)
- ✅ Clear explanation
- ✅ One-click apply

---

## 🔤 Regex Pattern Tab

### Modes:
1. **Clean (Remove)**: Remove matching characters
2. **Replace**: Find and replace
3. **Extract**: Keep only matching characters
4. **Validate**: Keep only rows that match

### Quick Workflow:
1. Select mode from dropdown
2. Enter pattern (or use Common Patterns)
3. If Replace mode: Enter replacement text
   - Quick buttons: Space, Empty
4. Click **Preview** to see results (optional)
5. Click **Apply to Selected**

### Common Patterns (One-Click):
- **Remove special chars**: `[^a-zA-Z0-9\s]`
- **Extract digits**: `[0-9]`
- **Validate email**: Email regex pattern

### Replace Mode Quick Buttons:
- **Space**: Replace with space
- **Empty**: Remove (replace with nothing)

---

## 📏 Length Validation Tab

### Exact Length:
1. Select **Exact** mode
2. Enter exact length (e.g., 10 for phone)
3. Click **Apply**
4. Rows not matching exact length → Rejected

### Range:
1. Select **Range** mode
2. Enter Min and Max values
3. Click **Apply**
4. Rows outside range → Rejected

### Use Cases:
- Phone numbers: Exact 10
- Zip codes: Exact 5 or 6
- Names: Range 2-50
- Descriptions: Range 10-500

---

## 🔠 Case Transform Tab

### Options:
- **UPPERCASE**: ALL CAPS
- **lowercase**: all small
- **Title Case**: First Letter Of Each Word
- **Sentence case**: First letter only

### Workflow:
1. Select case type
2. Click **Apply**
3. All selected columns transformed instantly

---

## 💡 Real-World Examples

### Example 1: Clean Name Column
**Goal**: Remove special chars and fix spacing

**Steps**:
1. Check **Name** column
2. Go to **AI** tab
3. Type: "Remove special characters"
4. Click **Get Suggestion**
5. Click **Apply**
6. Done! `John@Doe#123` → `JohnDoe123`

---

### Example 2: Standardize Phone Numbers
**Goal**: Keep only 10-digit phones

**Steps**:
1. Check **Phone** column
2. Go to **Length** tab
3. Select **Exact**
4. Enter: 10
5. Click **Apply**
6. Invalid phones → Rejected Records

---

### Example 3: Fix Column with Underscores
**Goal**: `first_name_last` → `first name last`

**Steps**:
1. Check column
2. Go to **Regex** tab
3. Select **Replace** mode
4. Pattern: `_`
5. Click **Space** button (or type space)
6. Click **Apply to Selected**
7. Done!

---

### Example 4: Multiple Columns at Once
**Goal**: Uppercase 5 name columns

**Steps**:
1. Check all 5 name columns
2. Go to **Case** tab
3. Select **UPPERCASE**
4. Click **Apply**
5. All 5 columns transformed instantly!

---

### Example 5: Complex Cleaning
**Goal**: Remove special chars AND replace underscores

**Steps**:
1. Check column
2. **First transformation**:
   - Go to **Regex** tab
   - Mode: **Replace**
   - Pattern: `_`
   - Replace: (space)
   - Click **Apply**
3. **Second transformation**:
   - Mode: **Clean (Remove)**
   - Pattern: `[^a-zA-Z0-9\s]`
   - Click **Apply**
4. Done! `hello_world@test` → `hello world test`

---

## 🎯 Key Features

### ✅ Multi-Column Selection
- Select multiple columns at once
- Apply same transformation to all
- Saves tons of time!

### ✅ Visual Feedback
- Selected columns highlighted in blue
- See sample values before applying
- Preview transformations (Regex tab)

### ✅ Undo Support
- Made a mistake? Click **Undo**
- Restores previous state
- Works for all transformations

### ✅ Rejected Records Tracking
- Invalid rows automatically moved
- View in expandable section
- Download as Excel
- Includes rejection reason and timestamp

### ✅ Applied Rules History
- See all transformations applied
- Timestamp and details
- Rejected count per rule
- Expandable section at bottom

---

## 📊 Metrics Bar

Always visible at top:
- **Total Rows**: Current row count
- **Total Columns**: Number of columns
- **Rejected Rows**: How many rejected
- **Undo Button**: Undo last action

---

## 🗑️ Rejected Records

### What Gets Rejected:
- Length validation failures
- Regex validation failures
- Extract mode with no matches

### Rejected Records Include:
- Original row data
- Rejection reason
- Column that failed
- Timestamp

### Actions:
- View in expandable section
- Download as Excel
- Review and fix manually

---

## ⚡ Performance Tips

### For Large Datasets:
1. **Select fewer columns** at once
2. **Use Preview** before applying (Regex tab)
3. **Apply in batches** if needed
4. **Check Rejected Records** after each batch

### For Speed:
1. **Use AI suggestions** for quick patterns
2. **Use Common Patterns** (one-click)
3. **Select multiple columns** for same transformation
4. **Use quick buttons** (Space, Empty, etc.)

---

## 🎨 UI Highlights

### Color Coding:
- **Blue background**: Selected columns
- **White background**: Unselected columns
- **Metrics**: Green for positive, Red for negative

### Layout:
- **2:1 ratio**: Data grid (larger) + Transformations (smaller)
- **Tabs**: Organized by transformation type
- **Expandable sections**: History and Rejected Records

### Buttons:
- **Primary (Blue)**: Main actions (Apply, Get Suggestion)
- **Secondary (Gray)**: Preview, Undo
- **Small buttons**: Quick actions (Space, Empty, etc.)

---

## 🔄 Workflow Comparison

### Old Way (Sequential):
1. Select ONE column
2. Choose validation type
3. Configure settings
4. Preview
5. Apply
6. Repeat for each column ❌ SLOW

### New Way (Excel-like):
1. Select MULTIPLE columns ✅
2. Choose transformation ✅
3. Apply to ALL at once ✅
4. DONE! ⚡ FAST

---

## 📝 Best Practices

### 1. Start with AI
- Let AI suggest patterns
- Saves time learning regex
- Usually accurate

### 2. Preview First (Regex)
- Use Preview button
- Check before/after
- Verify results

### 3. Work in Batches
- Group similar columns
- Apply same transformation
- More efficient

### 4. Check Rejected Records
- Review after each transformation
- Understand why rows failed
- Adjust patterns if needed

### 5. Use Undo Freely
- Don't be afraid to experiment
- Undo is always available
- Try different approaches

---

## 🆘 Troubleshooting

### "No columns selected"
- Check at least one column checkbox
- Or use "Select All" button

### "Pattern not working"
- Try AI suggestion first
- Check Common Patterns
- Use Preview to test

### "Too many rejected"
- Review rejection reasons
- Adjust pattern/length
- Use Undo and try again

### "Transformation too slow"
- Select fewer columns
- Work in smaller batches
- Close other applications

---

## 🎓 Learning Path

### Beginner:
1. Start with **Case Transform** (easiest)
2. Try **Length Validation** (simple)
3. Use **AI Suggestions** (guided)

### Intermediate:
1. Learn **Common Patterns** (Regex tab)
2. Use **Replace mode** (practical)
3. Combine multiple transformations

### Advanced:
1. Write custom regex patterns
2. Use **Extract** and **Validate** modes
3. Handle complex data cleaning

---

## 🚀 Summary

The new Excel-like interface makes data quality:
- ✅ **Faster**: Multi-column selection
- ✅ **Easier**: Visual grid interface
- ✅ **Smarter**: AI suggestions
- ✅ **Safer**: Undo support
- ✅ **Cleaner**: Organized tabs
- ✅ **Better**: Real-time feedback

**Try it now and experience the difference!**
