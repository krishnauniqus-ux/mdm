# Simple Data Quality - Single Table View

## 🎯 Overview

Super simple interface - all columns in one table with transformations configured inline.

## 📊 Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Metrics: Rows | Columns | Rejected | Undo                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ☑ | Column Name      | Mode    | Pattern      | Replace | Case | AI | Quick│
│  ☑ | Sample Values    |         |              |         |      |    |      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  ☑ | Column Name      | Mode    | Pattern      | Replace | Case | AI | Quick│
│  ☑ | Sample Values    |         |              |         |      |    |      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  ...                                                                          │
│                                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Apply All Enabled] [Preview Changes] [Enable All] [Disable All]           │
├─────────────────────────────────────────────────────────────────────────────┤
│  📜 History | 🗑️ Rejected Records                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 How to Use

### Step 1: Enable Columns
- Check the ☑ box for columns you want to transform
- Or click "Enable All" to select all

### Step 2: Configure Each Row
For each enabled column:

1. **Mode**: Select transformation type
   - Clean: Remove matching characters
   - Replace: Find and replace
   - Extract: Keep only matching
   - Validate: Keep only rows that match
   - Case: Change case
   - Length: Validate length

2. **Pattern**: Enter regex pattern (for Clean/Replace/Extract/Validate)
   - Or enter length (for Length mode)

3. **Replace With**: Enter replacement text (for Replace mode only)

4. **Case**: Select case type (for Case mode only)

5. **AI Button**: Click to get smart suggestion

6. **Quick**: Select common pattern from dropdown

### Step 3: Apply
- Click "Preview Changes" to see what will happen
- Click "Apply All Enabled" to apply transformations
- Use "Undo" if needed

## 💡 Examples

### Example 1: Remove Special Characters
```
☑ Name Column
Mode: Clean
Pattern: [^a-zA-Z0-9\s]
Click: Apply All Enabled
```

### Example 2: Replace Underscores
```
☑ Title Column
Mode: Replace
Pattern: _
Replace With: (space)
Click: Apply All Enabled
```

### Example 3: Validate Emails
```
☑ Email Column
Mode: Validate
Pattern: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
Click: Apply All Enabled
```

### Example 4: Use AI
```
☑ Phone Column
Click: 🤖 AI button
(AI suggests: Extract digits)
Click: Apply All Enabled
```

### Example 5: Multiple Columns
```
☑ First Name
Mode: Case
Case: Title Case

☑ Last Name
Mode: Case
Case: Title Case

☑ Email
Mode: Validate
Pattern: (email regex)

Click: Apply All Enabled
(All 3 columns transformed at once!)
```

## 🎯 Quick Patterns

Select from dropdown for instant configuration:
- **Remove special**: Removes special characters
- **Extract digits**: Keeps only numbers
- **Remove spaces**: Removes all spaces
- **Validate email**: Email format validation

## 🤖 AI Suggestions

Click the AI button for any column to get smart suggestions based on the data:
- Detects underscores → Suggests replace with space
- Detects special chars → Suggests clean
- Detects emails → Suggests validation
- Detects numbers → Suggests extract

## 📏 Modes Explained

### Clean
- Removes characters matching the pattern
- Example: `[^a-zA-Z0-9\s]` removes special chars

### Replace
- Finds pattern and replaces with your text
- Example: `_` → ` ` replaces underscores with spaces

### Extract
- Keeps only characters matching the pattern
- Example: `[0-9]` keeps only digits
- Rows with no matches → Rejected

### Validate
- Keeps only rows where value matches pattern
- Example: `^\d{10}$` keeps only 10-digit values
- Non-matching rows → Rejected

### Case
- Changes text case
- Options: UPPERCASE, lowercase, Title Case

### Length
- Validates string length
- Enter: `10` for exact length
- Enter: `5-20` for range

## ✅ Features

- **Single Table View**: Everything in one place
- **Inline Configuration**: No switching between screens
- **Multi-Column**: Enable multiple columns, apply all at once
- **AI Suggestions**: Smart pattern detection
- **Quick Patterns**: One-click common patterns
- **Preview**: See changes before applying
- **Undo**: Revert last transformation
- **History**: Track all applied rules
- **Rejected Records**: Download invalid rows

## 🎨 Column Layout

Each row shows:
1. ☑ Enable checkbox
2. Column name + sample values
3. Mode dropdown
4. Pattern/Length input
5. Replace input (Replace mode only)
6. Case dropdown (Case mode only)
7. AI button
8. Quick patterns dropdown

## 📊 Action Buttons

- **Apply All Enabled**: Apply transformations to all checked columns
- **Preview Changes**: See before/after samples
- **Enable All**: Check all columns
- **Disable All**: Uncheck all columns

## 🗑️ Rejected Records

Invalid rows are automatically:
- Removed from main data
- Added to Rejected Records
- Include rejection reason
- Can be downloaded as Excel

## ↩️ Undo

- Click Undo to revert last transformation
- Restores both main data and rejected records
- Works for all transformation types

## 💪 Best Practices

1. **Start Small**: Enable 1-2 columns first
2. **Use Preview**: Check results before applying
3. **Use AI**: Let AI suggest patterns
4. **Use Quick Patterns**: Save time with presets
5. **Check Rejected**: Review why rows failed
6. **Use Undo**: Don't be afraid to experiment

## ⚡ Speed Tips

- Enable multiple similar columns
- Use Quick Patterns dropdown
- Click AI for instant suggestions
- Apply all at once instead of one by one

## 🎓 Learning Path

1. Try Case transformation (easiest)
2. Use Quick Patterns (pre-configured)
3. Click AI button (guided)
4. Write custom patterns (advanced)

---

**Simple, fast, and powerful!**
